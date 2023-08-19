import os
import numpy as np
import torch
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.transforms import ToSparseTensor
from torch_geometric.utils import negative_sampling, add_self_loops, train_test_split_edges
from torch_geometric.data import Data
from torch_sparse import from_scipy
from ogb.linkproppred import PygLinkPropPredDataset
import scipy.sparse as sp
import json

def load_data(dataset, seed = None, link_prediction = False):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    if dataset in ['Cora', 'CiteSeer', 'PubMed']:
        path = os.path.join('dataset', dataset)
        dataset = Planetoid(path, dataset)
        data = dataset[0]
        if link_prediction:
            # edge splits for link prediction
            split_edge = do_edge_split(dataset)
            data.edge_index = split_edge['train']['edge']
            # adjacency matrix
            data = ToSparseTensor(remove_edge_index = False)(data)
            data.neg_edge_index = split_edge['train']['edge_neg']
            data.val_edge_index = split_edge['valid']['edge']
            data.val_neg_edge_index = split_edge['valid']['edge_neg']
            data.test_edge_index = split_edge['test']['edge']
            data.test_neg_edge_index = split_edge['test']['edge_neg']
        else:
            # adjacency matrix
            data = ToSparseTensor(remove_edge_index = False)(data)
    
    elif dataset in ['Flickr']:
        adj_full = sp.load_npz('dataset/{}/adj_full.npz'.format(dataset))
        role = json.load(open('dataset/{}/role.json'.format(dataset)))
        feats = np.load('dataset/{}/feats.npy'.format(dataset))
        class_map = json.load(open('dataset/{}/class_map.json'.format(dataset)))
        data = Data(edge_index = from_scipy(adj_full)[0], 
                    x = torch.tensor(feats, dtype = torch.float),
                    y = torch.tensor([class_map[str(i)] for i in range(adj_full.shape[0])]),
                    num_nodes = adj_full.shape[0])
        # adjacency matrix
        data = ToSparseTensor(remove_edge_index = False)(data)
        data.train_mask = torch.tensor([False] * data.num_nodes, dtype = torch.bool)
        data.val_mask = torch.tensor([False] * data.num_nodes, dtype = torch.bool)
        data.test_mask = torch.tensor([False] * data.num_nodes, dtype = torch.bool)
        data.train_mask[role['tr']] = True
        data.val_mask[role['va']] = True
        data.test_mask[role['te']] = True
    
    elif dataset in ['ogbl-collab']:
        dataset = PygLinkPropPredDataset(dataset)
        data = dataset[0]
        # edge splits for link prediction
        split_edge = dataset.get_edge_split()
        data.edge_index = split_edge['train']['edge'].t() # exist unsymmetric edges
        # adjacency matrix
        data = ToSparseTensor(remove_edge_index = False, attr = None)(data)
        data.neg_edge_index = get_neg_edges(split_edge['train']['edge'].t(), data.num_nodes)
        data.val_edge_index = split_edge['valid']['edge'].t()
        data.val_neg_edge_index = split_edge['valid']['edge_neg'].t()
        data.test_edge_index = split_edge['test']['edge'].t()
        data.test_neg_edge_index = split_edge['test']['edge_neg'].t()
        # node features
        data.x = data.x.type(torch.float32)
    
    else:
        raise Exception('Unrecognized dataset name.')
    return data

def do_edge_split(dataset, val_ratio = 0.05, test_ratio = 0.1):
    data = dataset[0]
    data = train_test_split_edges(data, val_ratio, test_ratio)
    edge_index, _ = add_self_loops(data.train_pos_edge_index)
    data.train_neg_edge_index = negative_sampling(edge_index,
                                                  num_nodes = data.num_nodes,
                                                  num_neg_samples = data.train_pos_edge_index.size(1))
    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = data.train_pos_edge_index
    split_edge['train']['edge_neg'] = data.train_neg_edge_index
    split_edge['valid']['edge'] = data.val_pos_edge_index
    split_edge['valid']['edge_neg'] = data.val_neg_edge_index
    split_edge['test']['edge'] = data.test_pos_edge_index
    split_edge['test']['edge_neg'] = data.test_neg_edge_index
    return split_edge

def get_neg_edges(edge, num_nodes):
    """
    Parameters
    ----------
    edge: edge indices [2, num_edges]
    num_nodes: number of nodes

    Returns
    -------
    neg_edge: tensor with shape [2, num_edges]
    """
    new_edge_index, _ = add_self_loops(edge)
    neg_edge = negative_sampling(new_edge_index, num_nodes = num_nodes, num_neg_samples = edge.size(1))
    return neg_edge

def load_graph_list(dataset, seed = None, val_ratio = 0.2, test_ratio = 0.3):
    if seed is not None:
        torch.manual_seed(seed)
    if dataset in ['MUTAG', 'PTC_MR', 'IMDB-BINARY', 'COLLAB']:
        dataset = TUDataset(root = 'dataset', name = dataset)
        # data split
        split_graph = do_graph_split(dataset)
        data = {'train': [], 'valid': [], 'test': []}
        labels = set([])
        for i in range(len(split_graph['train'])):
            data['train'].append(ToSparseTensor(remove_edge_index = False)(split_graph['train'][i]))
            data['train'][-1].num_nodes = data['train'][-1].adj_t.size(0)
            labels.add(split_graph['train'][i].y.item())
        for i in range(len(split_graph['valid'])):
            data['valid'].append(ToSparseTensor(remove_edge_index = False)(split_graph['valid'][i]))
            data['valid'][-1].num_nodes = data['valid'][-1].adj_t.size(0)
            labels.add(split_graph['train'][i].y.item())
        for i in range(len(split_graph['test'])):
            data['test'].append(ToSparseTensor(remove_edge_index = False)(split_graph['test'][i]))
            data['test'][-1].num_nodes = data['test'][-1].adj_t.size(0)
            labels.add(split_graph['train'][i].y.item())
        data['num_classes'] = len(labels)
    else:
        raise Exception('Unrecognized dataset name.')
    return data

def do_graph_split(dataset, val_ratio = 0.2, test_ratio = 0.3):        
    num_graphs = len(dataset)
    val_num_graphs = int(val_ratio * num_graphs)
    test_num_graphs = int(test_ratio * num_graphs)
    dataset = dataset.shuffle()
    # mask test and validation graphs
    split_graph = {}
    split_graph['valid'] = dataset[: val_num_graphs]
    split_graph['test'] = dataset[val_num_graphs: val_num_graphs + test_num_graphs]
    split_graph['train'] = dataset[val_num_graphs + test_num_graphs: ]
    return split_graph