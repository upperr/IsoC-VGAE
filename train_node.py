import os
import time
import random
import numpy as np
import argparse
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser(description = 'parameters')
# experimental options
parser.add_argument('--dataset', type = str, default = 'Cora', help = "Dataset name: Cora, CiteSeer, PubMed, Flickr")
parser.add_argument('--GNN', type = str, default = 'GCN', help = "GNN name for encoder: GCN, GIN")
parser.add_argument('--learning_rate', type = float, default = 5e-6, help = 'Learning rate.')
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
parser.add_argument('--lambda_self', type = float, default = 1)
parser.add_argument('--lambda_neighbor', type = float, default = 0.1)
parser.add_argument('--lambda_degree', type = float, default = 1)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from load_data import load_data
from model import ISOC_VGAE
from layers import MLP

class NodeDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.len = embeddings.shape[0]
        self.x = embeddings
        self.y = labels.long()

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len

# Training
def train(data):
    '''
     Main training function
    '''
    node_features = data.x.to(device)
    adj = data.adj_t.to(device)
    neighbor_dict = {x: [] for x in range(data.num_nodes)}
    for source_node, target_node in data.edge_index.T:
        neighbor_dict[source_node.item()].append(target_node.item())
    num_features = node_features.shape[1]
    hidden_dim = [int(x) for x in args.hidden_dim.split('_')]
    degrees = adj.sum(0)
    model = ISOC_VGAE(num_features = num_features, 
                    hidden_dim = hidden_dim, 
                    device = device, 
                    GNN_name = args.GNN, 
                    dropout = args.dropout_train,
                    lambda_self = args.lambda_self,
                    lambda_neighbor = args.lambda_neighbor, 
                    lambda_degree = args.lambda_degree).to(device)
    degree_params = list(map(id, model.reconstruct_degree.parameters()))
    base_params = filter(lambda p: id(p) not in degree_params, model.parameters())
    optimizer = torch.optim.Adam([{'params': base_params}, {'params': model.reconstruct_degree.parameters(), 'lr': 1e-2}], lr = args.learning_rate, weight_decay = 0.0003)
    best_loss = float('inf')
    last_best_epoch = 0
    model.train()
    for epoch in range(args.epochs_train):
        time_start = time.time()
        loss, embeddings = model(adj, node_features, degrees, neighbor_dict)
        optimizer.zero_grad()
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

def node_classification(data, embeddings):
    input_dims = embeddings.shape[1]
    num_classes = int(max(data.y)) + 1
    # data split
    train_dataset = NodeDataset(embeddings[~(data.val_mask + data.test_mask), :], data.y[~(data.val_mask + data.test_mask)])
    val_dataset = NodeDataset(embeddings[data.val_mask, :], data.y[data.val_mask])
    test_dataset = NodeDataset(embeddings[data.test_mask, :], data.y[data.test_mask])
    # construct classifier
    classifier = MLP(input_dim = input_dims, hidden_dim = input_dims // 2, output_dim = num_classes, num_hidden = 4, dropout = args.dropout).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters())
    
    best_val = 0
    last_best_epoch = 0
    print('Begin training classifier!')
    for epoch in range(args.epochs_task):
        time_start = time.time()
        # training
        train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
        classifier.train()
        for batch, (inputs, labels) in enumerate(train_loader, 0):
            inputs = inputs.to(device)
            labels = labels.to(device)
            preds = classifier(inputs)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # validation
        val_loader = DataLoader(val_dataset, batch_size = args.batch_size, num_workers = args.num_workers)
        classifier.eval()
        with torch.no_grad():
            total_preds = []
            total_labels = []
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = classifier(inputs)
                _, preds = torch.max(outputs.data, 1)
                total_preds.append(preds.cpu())
                total_labels.append(labels.cpu())
        val_acc = accuracy_score(torch.cat(total_labels), torch.cat(total_preds))
        t = time.time() - time_start
        print(f"Epoch: {epoch+1:04d} train_loss={loss:.3f} val_acc={val_acc:.3f} time={t:.2f}")
        # testing
        if val_acc > best_val:
            print('Save classifier!')
            torch.save(classifier.state_dict(), 'best_mlp.pkl')
            best_val = val_acc
            last_best_epoch = 0
            test_loader = DataLoader(test_dataset, batch_size = args.batch_size, num_workers = args.num_workers)
            classifier.eval()
            with torch.no_grad():
                total_preds = []
                total_labels = []
                for inputs, labels in test_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = classifier(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    total_preds.append(preds.cpu())
                    total_labels.append(labels.cpu())
            test_acc = accuracy_score(torch.cat(total_labels), torch.cat(total_preds))
        # early stop
        if last_best_epoch > args.early_stopping:
            break
        else:
            last_best_epoch += 1
    print(f'Best Validation: {best_val:.4f}')
    print(f'Test Accuracy: {test_acc:.4f}')

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(args.dataset, seed = 1234)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
    print('Begin training model!')
    embeddings = train(data)
    node_classification(data, embeddings)
