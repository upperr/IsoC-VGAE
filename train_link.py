import os
import time
import random
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score

parser = argparse.ArgumentParser(description = 'parameters')
# experimental options
parser.add_argument('--dataset', type = str, default = 'Cora', help = "Dataset name: Cora, CiteSeer, PubMed, ogbl-collab")
parser.add_argument('--GNN', type = str, default = 'GCN', help = "GNN name for encoder: GCN, GIN")
parser.add_argument('--learning_rate', type = float, default = 0.01, help = 'Learning rate.')
parser.add_argument('--epochs_train', type = int, default = 30, help = 'Max number of epochs for training. Training may stop early after convergence.')
parser.add_argument('--epochs_task', type = int, default = 50, help = 'Max number of epochs for tasks. Training may stop early after convergence.')
parser.add_argument('--early_stopping', type = int, default = 10, help = "Number of epochs to run after last best validation.")
parser.add_argument('--gpu', type = str, default = '0')
parser.add_argument('--batch_size', type = int, default = 512, help = 'Batch size.')
parser.add_argument('--num_workers', type = int, default = 0, help = "Number of workers.")
parser.add_argument('--seed', type = int, default = None, help = 'Random seed.')
# model options
parser.add_argument('--hidden_dim', type = str, default = '512_512', help = 'Layer sizes.')
parser.add_argument('--dropout_train', type = float, default = 0., help = 'Dropout rate (1 - keep probability).')
parser.add_argument('--dropout', type = float, default = 0.5, help = 'Dropout rate (1 - keep probability).')
parser.add_argument('--bn', type = str, default = 'BatchNorm1d', help = 'Batch Normalization: BatchNorm1d, PairNorm.')
parser.add_argument('--lambda_self', type = float, default = 1)
parser.add_argument('--lambda_neighbor', type = float, default = 0.1)
parser.add_argument('--lambda_degree', type = float, default = 1)
parser.add_argument('--classifier_hidden_dim', type = int, default = 256, help = 'Layer size for classifier.')
parser.add_argument('--classifier_num_hidden', type = int, default = 4, help = 'Layer number for classifier.')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected, degree
from load_data import load_data
from model import ISOC_VGAE, LinkPredictor

class LinkDataset(Dataset):
    def __init__(self, data, split):
        if split == 'train':
            self.edge_index = data.edge_index
            self.neg_edge_index = data.neg_edge_index
        elif split == 'valid':
            self.edge_index = data.val_edge_index
            self.neg_edge_index = data.val_neg_edge_index
        elif split == 'test':
            self.edge_index = data.test_edge_index
            self.neg_edge_index = data.test_neg_edge_index

    def __getitem__(self, index):
        return self.edge_index.T[index], self.neg_edge_index.T[index]

    def __len__(self):
        return self.edge_index.size(1)

# Training
def train(data):
    '''
     Main training function
    '''
    node_features = data.x.to(device)
    adj = data.adj_t.to(device)
    edges = to_undirected(data.edge_index)
    degrees = degree(edges[0]).to(device)
    print('Building neighbor dictionary...')
    neighbor_dict = {x: [] for x in range(data.num_nodes)}
    for source_node, target_node in edges.T:
        neighbor_dict[source_node.item()].append(target_node.item())
    num_features = node_features.shape[1]
    hidden_dim = [int(x) for x in args.hidden_dim.split('_')]
    model = ISOC_VGAE(num_features = num_features, 
                    hidden_dim = hidden_dim, 
                    device = device, 
                    GNN_name = args.GNN, 
                    dropout = args.dropout_train,
                    bn = args.bn,
                    lambda_self = args.lambda_self,
                    lambda_neighbor = args.lambda_neighbor, 
                    lambda_degree = args.lambda_degree).to(device)
    degree_params = list(map(id, model.reconstruct_degree.parameters()))
    base_params = filter(lambda p: id(p) not in degree_params, model.parameters())
    optimizer = torch.optim.Adam([{'params': base_params}, {'params': model.reconstruct_degree.parameters(), 'lr': 1e-2}], lr = 5e-6, weight_decay = 0.0003)
    best_loss = float('inf')
    last_best_epoch = 0
    model.train()
    for epoch in range(args.epochs_train):
        time_start = time.time()
        optimizer.zero_grad()
        loss, embeddings = model(adj, node_features, degrees, neighbor_dict)
        loss.backward()
        optimizer.step()
        t = time.time() - time_start
        print(f"Epoch: {epoch+1:04d} train_loss={loss:.3f} self_loss={model.loss_self:.3f} neighbor_loss={model.kl_neighbor:.3f} degree_loss={model.loss_degree:.3f} time={t:.2f}")
        if loss < best_loss:
            print('Save model!')
            torch.save(model.state_dict(), 'best_model.pkl')
            best_embeddings = embeddings.cpu().detach()
            best_loss = loss
            last_best_epoch = 0
        # early stop
        if last_best_epoch > args.early_stopping:
            break
        else:
            last_best_epoch += 1
    return best_embeddings

def run(data, embeddings):
    input_dims = embeddings.shape[1]
    # data split
    train_data = LinkDataset(data, 'train')
    val_data = LinkDataset(data, 'valid')
    test_data = LinkDataset(data, 'test')
    # construct predictor for link prediction
    predictor = LinkPredictor(input_dim = input_dims, hidden_dim = args.classifier_hidden_dim, num_hidden = args.classifier_num_hidden, dropout = args.dropout).to(device)
    loss_function = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(predictor.parameters())
    best_val = 0
    last_best_epoch = 0
    print('Begin training classifier!')
    for epoch in range(args.epochs_task):
        time_start = time.time()
        # training
        train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
        predictor.train()
        total_preds = []
        total_labels = []
        for batch, (pos_edges, neg_edges) in enumerate(train_loader):
            optimizer.zero_grad()
            nodes_source = torch.cat([pos_edges[:, 0], neg_edges[:, 0]])
            nodes_target = torch.cat([pos_edges[:, 1], neg_edges[:, 1]])
            embeddings_source = embeddings[nodes_source, :].to(device)
            embeddings_target = embeddings[nodes_target, :].to(device)
            labels = torch.cat([torch.ones([pos_edges.size(0), 1]), torch.zeros([neg_edges.size(0), 1])]).to(device)
            preds = predictor(embeddings_source, embeddings_target) # z(N, K, T)
            loss = loss_function(preds, labels)
            loss.backward()
            optimizer.step()
            total_preds.append(preds.detach().cpu())
            total_labels.append(labels.cpu())
        auc_train = roc_auc_score(torch.cat(total_labels), torch.cat(total_preds))
        ap_train = average_precision_score(torch.cat(total_labels), torch.cat(total_preds))
        # validation
        val_loader = DataLoader(val_data, batch_size = args.batch_size, num_workers = args.num_workers)
        predictor.eval()
        with torch.no_grad():
            total_preds = []
            total_labels = []
            for batch, (pos_edges, neg_edges) in enumerate(val_loader):
                nodes_source = torch.cat([pos_edges[:, 0], neg_edges[:, 0]])
                nodes_target = torch.cat([pos_edges[:, 1], neg_edges[:, 1]])
                embeddings_source = embeddings[nodes_source, :].to(device)
                embeddings_target = embeddings[nodes_target, :].to(device)
                labels = torch.cat([torch.ones(pos_edges.size(0)), torch.zeros(neg_edges.size(0))]).to(device)
                preds = predictor(embeddings_source, embeddings_target)
                total_preds.append(preds.cpu())
                total_labels.append(labels.cpu())
        auc_val = roc_auc_score(torch.cat(total_labels), torch.cat(total_preds))
        ap_val = average_precision_score(torch.cat(total_labels), torch.cat(total_preds))
        t = time.time() - time_start
        print(f"Epoch: {epoch+1:04d} auc_train={auc_train:.3f} ap_train={ap_train:.3f} auc_val={auc_val:.3f} ap_val={ap_val:.3f} time={t:.2f}")
        # testing
        if auc_val > best_val:
            print('Save predictor!')
            torch.save(predictor.state_dict(), 'best_predictor.pkl')
            best_val = auc_val
            last_best_epoch = 0
            test_loader = DataLoader(test_data, batch_size = args.batch_size, num_workers = args.num_workers)
            predictor.eval()
            with torch.no_grad():
                total_preds = []
                total_labels = []
                for batch, (pos_edges, neg_edges) in enumerate(test_loader):
                    nodes_source = torch.cat([pos_edges[:, 0], neg_edges[:, 0]])
                    nodes_target = torch.cat([pos_edges[:, 1], neg_edges[:, 1]])
                    embeddings_source = embeddings[nodes_source, :].to(device)
                    embeddings_target = embeddings[nodes_target, :].to(device)
                    labels = torch.cat([torch.ones(pos_edges.size(0)), torch.zeros(neg_edges.size(0))]).to(device)
                    preds = predictor(embeddings_source, embeddings_target)
                    total_preds.append(preds.cpu())
                    total_labels.append(labels.cpu())
            auc_test = roc_auc_score(torch.cat(total_labels), torch.cat(total_preds))
            ap_test = average_precision_score(torch.cat(total_labels), torch.cat(total_preds))
        # early stop
        if last_best_epoch > args.early_stopping:
            break
        else:
            last_best_epoch += 1
    print(f'Best Valid AUC: {best_val:.4f}')
    print(f'Test AUC score: {auc_test:.4f}')
    print(f'Test AP score: {ap_test:.4f}')

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(args.dataset, link_prediction = True, seed = 1234)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
    print('Begin training model!')
    embeddings = train(data)
    run(data, embeddings)
