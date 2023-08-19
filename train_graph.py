import os
import time
import random
import numpy as np
import argparse
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser(description = 'parameters')
# experimental options
parser.add_argument('--dataset', type = str, default = 'Cora', help = "Dataset name: MUTAG, PTC_MR, IMDB-BINARY, COLLAB")
parser.add_argument('--GNN', type = str, default = 'GIN', help = "GNN name for encoder: GCN, GIN")
parser.add_argument('--learning_rate', type = float, default = 5e-6, help = 'Learning rate.')
parser.add_argument('--epochs_train', type = int, default = 30, help = 'Max number of epochs for training. Training may stop early after convergence.')
parser.add_argument('--epochs_task', type = int, default = 50, help = 'Max number of epochs for tasks. Training may stop early after convergence.')
parser.add_argument('--early_stopping', type = int, default = 10, help = "Number of epochs to run after last best validation.")
parser.add_argument('--batch_size', type = int, default = 64, help = 'Batch size.')
parser.add_argument('--num_workers', type = int, default = 0, help = "Number of workers.")
parser.add_argument('--seed', type = int, default = None, help = 'Random seed.')
parser.add_argument('--seed_data', type = int, default = None, help = 'Random seed.')
parser.add_argument('--gpu', type = str, default = '0')
# model options
parser.add_argument('--hidden_dim', type = str, default = '512_512', help = 'Layer sizes.')
parser.add_argument('--dropout_train', type = float, default = 0.5, help = 'Dropout rate (1 - keep probability).')
parser.add_argument('--dropout', type = float, default = 0.5, help = 'Dropout rate (1 - keep probability).')
parser.add_argument('--bn', type = str, default = 'PairNorm', help = 'Batch Normalization: BatchNorm1d, PairNorm.')
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
from torch_geometric.nn import global_add_pool
from load_data import load_graph_list
from model import ISOC_VGAE
from layers import MLP

class GraphDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        return self.data[index], index

    def __len__(self):
        return len(self.data)

class TaskDataset(Dataset):
    def __init__(self, inputs, labels):
        
        self.inputs = inputs
        self.labels = labels
    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

def construct_neighbor_dict(adj):
    neighbor_dict = {x: [] for x in range(adj.size(0))}
    source_nodes, target_nodes, _ = adj.coo()
    edges = torch.cat([torch.reshape(source_nodes, [-1, 1]), torch.reshape(target_nodes, [-1, 1])], 1)
    for source_node, target_node in edges:
        neighbor_dict[source_node.item()].append(target_node.item())
    return neighbor_dict
# Training
def train(dataset):
    '''
     Main training function
    '''
    dataset = dataset['train'] + dataset['valid'] + dataset['test']
    if dataset[0].x is None:
        for i in range(len(dataset)):
            dataset[i].x = torch.ones((dataset[i].num_nodes, 1))
    num_features = dataset[0].x.shape[1]
    dataset = GraphDataset(dataset)
    hidden_dim = [int(x) for x in args.hidden_dim.split('_')]
    # bulid model
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
    optimizer = torch.optim.Adam([{'params': base_params}, {'params': model.reconstruct_degree.parameters(), 'lr': 1e-2}], lr = args.learning_rate, weight_decay = 0.0003)
    # unsupervised training
    best_loss = float('inf')
    last_best_epoch = 0
    model.train()
    for epoch in range(args.epochs_train):
        time_start = time.time()
        data_loader = DataLoader(dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
        loss = 0.
        self_loss = 0.
        neighbor_loss = 0.
        degree_loss = 0.
        graph_embeddings = torch.zeros([len(dataset), hidden_dim[-1]])
        for batch, (batch_data, indices) in enumerate(data_loader):
            adj_block = batch_data.adj_t.to(device)
            inputs = batch_data.x.to(device)
            neighbor_dict = construct_neighbor_dict(adj_block)
            degrees = adj_block.sum(0).to(device)
            loss_curr, embeddings_curr = model(adj_block, inputs, degrees, neighbor_dict)
            self_loss += model.loss_self
            neighbor_loss += model.kl_neighbor
            degree_loss+= model.loss_degree
            optimizer.zero_grad()
            loss_curr.backward()
            optimizer.step()
            loss += loss_curr
            graph_embeddings_curr = global_add_pool(embeddings_curr, batch_data.batch.to(device))
            graph_embeddings[indices, :] = graph_embeddings_curr.cpu().detach()
        loss /= (batch + 1)
        self_loss /= (batch + 1)
        neighbor_loss /= (batch + 1)
        degree_loss /= (batch + 1)
        t = time.time() - time_start
        print(f"Epoch: {epoch+1:04d} train_loss={loss:.3f} self_loss={self_loss:.3f} neighbor_loss={neighbor_loss:.3f} degree_loss={degree_loss:.3f} time={t:.2f}")
        if loss < best_loss:
            print('Save model!')
            torch.save(model.state_dict(), 'best_model.pkl')
            best_embeddings = graph_embeddings
            best_loss = loss
            last_best_epoch = 0
        # early stop
        if last_best_epoch > args.early_stopping:
            break
        else:
            last_best_epoch += 1
    return best_embeddings
    
def graph_classification(dataset, embeddings):
    input_dims = embeddings.shape[1]
    label_train = torch.tensor([i.y for i in dataset['train']])
    label_val = torch.tensor([i.y for i in dataset['valid']])
    label_test = torch.tensor([i.y for i in dataset['test']])
    train_dataset = TaskDataset(embeddings[:label_train.shape[0], :], label_train)
    val_dataset = TaskDataset(embeddings[label_train.shape[0]: label_train.shape[0] + label_val.shape[0], :], label_val)
    test_dataset = TaskDataset(embeddings[label_train.shape[0] + label_val.shape[0]:, :], label_test)
    # construct classifier
    classifier = MLP(input_dim = input_dims, 
                     hidden_dim = args.classifier_hidden_dim, 
                     output_dim = dataset['num_classes'], 
                     num_hidden = args.classifier_num_hidden, 
                     dropout = args.dropout).to(device)
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
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            preds = classifier(inputs)
            labels = labels.to(device)
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
                preds = classifier(inputs)
                preds = torch.argmax(preds, 1)
                total_preds.append(preds.cpu())
                total_labels.append(labels)
        val_acc = accuracy_score(torch.cat(total_labels), torch.cat(total_preds))
        t = time.time() - time_start
        print(f"Epoch: {epoch+1:04d} train_loss={loss:.3f} val_acc={val_acc:.3f} time={t:.2f}")
        # testing
        if val_acc >= best_val:
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
                    preds = classifier(inputs)
                    preds = torch.argmax(preds, 1)
                    total_preds.append(preds.cpu())
                    total_labels.append(labels)
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
    dataset = load_graph_list(args.dataset, seed = args.seed_data)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
    print('Begin training model!')
    embeddings = train(dataset)
    graph_classification(dataset, embeddings)
