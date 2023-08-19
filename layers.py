import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv

# MLP with linear outputs (without softmax)
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden = 1, act = nn.ReLU(), bias = True, dropout = 0., bn = True):
        '''
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            num_hidden: number of hidden layers in the multi-layer perceptron.
        '''
        super(MLP, self).__init__()
        
        self.num_hidden = num_hidden
        self.act = act
        self.dropout = dropout
        self.bn = None
        
        self.linears = torch.nn.ModuleList()
        for layer in range(num_hidden):
            self.linears.append(nn.Linear(input_dim, hidden_dim, bias = bias))
            input_dim = hidden_dim
        self.linears.append(nn.Linear(input_dim, output_dim, bias = False))
        if bn:
            self.bn = torch.nn.ModuleList()
            for layer in range(num_hidden):
                self.bn.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.dropout:
            x = F.dropout(x, self.dropout, training = self.training)
        for layer in range(self.num_hidden):
            x = self.linears[layer](x)
            if self.bn is not None:
                x = self.bn[layer](x)
            x = self.act(x)   
        return self.linears[-1](x)

class PairNorm(nn.Module):
    def __init__(self, mode='PN', scale=10):
        """
            mode:
              'None' : No normalization
              'PN'   : Original version
              'PN-SI'  : Scale-Individually version
              'PN-SCS' : Scale-and-Center-Simultaneously version

            ('SCS'-mode is not in the paper but we found it works well in practice,
              especially for GCN and GAT.)
            PairNorm is typically used after each graph convolution operation.
        """
        assert mode in ['None', 'PN', 'PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

        # Scale can be set based on origina data, and also the current feature lengths.
        # We leave the experiments to future. A good pool we used for choosing scale:
        # [0.1, 1, 10, 50, 100]
    def forward(self, x):
        if self.mode == 'None':
            return x
        col_mean = x.mean(dim=0)
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = self.scale * x / rownorm_mean
        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual
        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean
        return x

class GCNLayer(nn.Module):
    """
    Graph convolution layer
    """
    def __init__(self, input_dim, output_dim, act = nn.ReLU(), add_self_loops = True, normalize = True, bias = False, dropout = 0., bn = None):
        super(GCNLayer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.GraphConv = GCNConv(input_dim, output_dim, add_self_loops = add_self_loops, normalize = normalize, bias = bias)
        self.act = act
        self.dropout = dropout
        if bn == 'BatchNorm1d':
            self.bn = nn.BatchNorm1d(output_dim)
        elif bn == 'PairNorm':
            self.bn = PairNorm(mode = "PN-SCS", scale = 20)
        else:
            self.bn = None

    def forward(self, inputs):
        x, adj = inputs
        if self.dropout:
            x = F.dropout(x, self.dropout, training = self.training)
        outputs = self.GraphConv(x, adj)  # (N, K)
        if self.bn is not None:
            outputs = self.bn(outputs)
        return self.act(outputs)

class GINLayer(nn.Module):
    """
    Graph isomorphism network layer
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden = 2, act = nn.ReLU(), bias = False, dropout = 0., bn = None):
        super(GINLayer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.MLP = MLP(input_dim, hidden_dim, output_dim, num_hidden = num_hidden)
        self.GraphConv = GINConv(nn = self.MLP, aggr = 'add')
        self.act = act
        self.dropout = dropout
        if bn == 'BatchNorm1d':
            self.bn = nn.BatchNorm1d(output_dim)
        elif bn == 'PairNorm':
            self.bn = PairNorm(mode = "PN-SCS", scale = 20)
        else:
            self.bn = None

    def forward(self, inputs):
        x, adj = inputs
        if self.dropout:
            x = F.dropout(x, self.dropout, training = self.training)
        outputs = self.GraphConv(x, adj)  # (N, K)
        if self.bn is not None:
            outputs = self.bn(outputs)
        return self.act(outputs)

class InnerProduct(nn.Module):
    """
    Decoder model for link prediction
    """
    def __init__(self, input_dim, hidden_dim, num_hidden = 1, dropout = 0.):
        super(InnerProduct, self).__init__()
        
        self.dropout = dropout
        self.linear = MLP(input_dim, hidden_dim, hidden_dim, num_hidden = num_hidden)

    def forward(self, x_s, x_t, act = nn.Sigmoid()):
        x = torch.cat([x_s, x_t])
        num_edges = x_s.size(0)
        if self.dropout:
            x = F.dropout(x, self.dropout, training = self.training)
        x = self.linear(x)
        x_s = x[:num_edges, :].clone()
        x_t = x[num_edges:, :].clone()
        outputs = torch.sum(x_s * x_t, dim = 1)
        return act(outputs)
