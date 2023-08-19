from layers import MLP, GCNLayer, GINLayer
import torch
import torch.nn as nn
import torch.nn.functional as F

# Main Autoencoder structure here
class ISOC_VGAE(nn.Module):
    def __init__(self, num_features, hidden_dim, device, GNN_name = "GIN", 
                 dropout = 0., bn = 'BatchNorm1d', lambda_self = 1, lambda_neighbor = 0.0001, lambda_degree = 10):
        '''
         Main Autoencoder structure
         INPUT:
         -----------------------
         input_dim    :    input graph feature dimension
         hidden_dim     :   latent variable feature dimension
         layer_num    :    GIN encoder, number of MLP layer
         sample_size     :    number of neighbors sampled
         device     :   CPU or GPU
         neighbor_num_list    :    number of neighbors for a specific node
         neighbor_dict    :    specific neighbors a node have
         norm   :   Pair Norm from https://openreview.net/forum?id=rkecl1rtwB
         lambda_loss    :   Trade-off between degree loss and neighborhood reconstruction loss
        '''
        super(ISOC_VGAE, self).__init__()
        self.input_dim = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = len(hidden_dim)
        self.lambda_self = lambda_self
        self.lambda_neighbor = lambda_neighbor
        self.lambda_degree = lambda_degree
        self.device = device
        
        GNNEncoder = []
        decode_mean = []
        decode_std = []
        reconstruct_self = []
        reconstruct_degree = []
        input_dim = num_features
        for layer in range(self.num_layers):
            # encoder layers
            if GNN_name == "GIN":
                GNNEncoder.append(GINLayer(input_dim = input_dim, 
                                           hidden_dim = hidden_dim[layer], 
                                           output_dim = hidden_dim[layer], 
                                           num_hidden = 2,
                                           act = lambda x: x if layer == self.num_layers - 1 else nn.ReLU(),
                                           dropout = 0. if layer == 0 else dropout,
                                           bn = None if layer == self.num_layers - 1 else bn))
            else:
                GNNEncoder.append(GCNLayer(input_dim = input_dim, 
                                           output_dim = hidden_dim[layer], 
                                           act = lambda x: x if layer == self.num_layers - 1 else nn.ReLU(), 
                                           dropout = 0. if layer == 0 else dropout,
                                           bn = None if layer == self.num_layers - 1 else bn))
            input_dim = hidden_dim[layer]
            # decoder layers
            if layer == self.num_layers - 1:
                output_dim = num_features
            else:
                output_dim = hidden_dim[self.num_layers - layer - 2]
            if layer > 0:
                decode_mean.append(MLP(hidden_dim[self.num_layers - layer - 1], hidden_dim[self.num_layers - layer - 1], output_dim, num_hidden = 2))
            decode_std.append(MLP(hidden_dim[self.num_layers - layer - 1], hidden_dim[self.num_layers - layer - 1], output_dim, num_hidden = 2))
            reconstruct_self.append(MLP(hidden_dim[self.num_layers - layer - 1], hidden_dim[self.num_layers - layer - 1], output_dim, num_hidden = 2))
            reconstruct_degree.append(MLP(hidden_dim[self.num_layers - layer - 1], hidden_dim[self.num_layers - layer - 1], 1, num_hidden = 3))
        self.GNN = nn.Sequential(*GNNEncoder)
        self.decode_mean = nn.Sequential(*decode_mean)
        self.decode_std = nn.Sequential(*decode_std)
        self.reconstruct_self = nn.Sequential(*reconstruct_self)
        self.reconstruct_degree = nn.Sequential(*reconstruct_degree)
        self.degree_loss_function = nn.MSELoss()
        self.self_loss_function = nn.MSELoss()
        self.neighbor_loss_function = nn.MSELoss()

    def encoder(self, h0, adj):
        '''
        GNN encoder
        '''
        h_list = []
        h = h0
        for layer in range(self.num_layers):
            h = self.GNN[layer]([h, adj])
            h_list.append(h)
        return h_list
    
    def neighborhood_distribution(self, embedding, neighbor_dict, degree):
        mean = torch.zeros(embedding.shape, device = self.device)
        for node in neighbor_dict.keys():
            neighbors = neighbor_dict[node]
            embedding_neighbor = torch.reshape(embedding[node, :], [1, -1])
            embedding_neighbor = torch.cat([embedding_neighbor, embedding[neighbors, :]], dim = 0)
            mean[node, :] = torch.mean(embedding_neighbor, dim = 0)
        return mean
    
    def decoder(self, h_list, h0, degree, neighbor_dict):
        '''
         Inv-GNN decoder
        '''
        self.loss_self = 0.
        self.kl_neighbor = 0.
        self.loss_degree = 0.
        mean_prior = 0.
        z_list = []
        for layer in range(self.num_layers - 1, -1, -1):
            if layer == 0:
                target_embedding = h0
            else:
                target_embedding = h_list[layer - 1]
            # reconstruct self node
            mean = self.reconstruct_self[self.num_layers - layer - 1](h_list[layer])
            if layer < self.num_layers - 1:
                mean_prior = self.decode_mean[self.num_layers - layer - 2](mean_prior)
            mean_posterior = mean + mean_prior
            log_std = self.decode_std[self.num_layers - layer - 1](h_list[layer])
            z = mean_posterior + torch.randn(mean.shape).to(self.device) * log_std.exp() # [q, c]
            self.loss_self += self.self_loss_function(target_embedding, z)
            # reconstruct neighbors
            h_mean = self.neighborhood_distribution(target_embedding, neighbor_dict, degree)
            self.kl_neighbor += self.kl_normal(mean_posterior, log_std, h_mean)
            # reconstruct degree
            reconstruction_degree = F.relu(self.reconstruct_degree[self.num_layers - layer - 1](h_list[layer])) # non-negative transformation
            self.loss_degree += self.degree_loss_function(reconstruction_degree, torch.unsqueeze(degree, dim = 1).float())
            
            mean_prior = z
            z_list.append(z)
        self.loss_self = self.loss_self / self.num_layers
        self.kl_neighbor = self.kl_neighbor / self.num_layers
        self.loss_degree = self.loss_degree / self.num_layers
        loss = self.lambda_self * self.loss_self + self.lambda_neighbor * self.kl_neighbor + self.lambda_degree * self.loss_degree
        return loss

    def forward(self, adj, h0, degree, neighbor_dict):
        # Generate GNN embeddings
        h_list = self.encoder(h0, adj)
        # Decoding and generating the latent representation by decoder
        loss = self.decoder(h_list, h0, degree, neighbor_dict)
        return loss, h_list[-1]
    
    def kl_normal(self, mean_source, log_sigma_source, mean_target):
        kl = 0.5 * torch.mean(-1 - 2 * log_sigma_source + torch.square(mean_source - mean_target) + torch.exp(2 * log_sigma_source))
        return kl

class LinkPredictor(nn.Module):
    """
    Decoder model for link prediction
    """
    def __init__(self, input_dim, hidden_dim, num_hidden = 1, dropout = 0.):
        super(LinkPredictor, self).__init__()
        
        self.linear = MLP(input_dim, hidden_dim, 1, num_hidden = num_hidden, dropout = dropout)
    def forward(self, x_s, x_t, act = nn.Sigmoid()):
        x = x_s * x_t
        x = self.linear(x)
        return act(x)
