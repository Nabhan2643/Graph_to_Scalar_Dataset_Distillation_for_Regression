import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from itertools import product
import numpy as np

class PGE(nn.Module):

    def __init__(self, nfeat, nnodes, nhid=128, nlayers=3, device=None, args=None):
        super(PGE, self).__init__()

        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(nfeat*2, nhid))
        self.bns = torch.nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(nhid))
        for i in range(nlayers-2):
            self.layers.append(nn.Linear(nhid, nhid))
            self.bns.append(nn.BatchNorm1d(nhid))
        self.layers.append(nn.Linear(nhid, 1))

        edge_index = np.array(list(product(range(nnodes), range(nnodes))))
        self.edge_index = edge_index.T
        self.nnodes = nnodes
        self.device = device
        self.reset_parameters()
        self.cnt = 0
        self.args = args
        self.nnodes = nnodes

    def forward(self, x):
        edge_index = self.edge_index
        edge_embed = torch.cat([x[edge_index[0]],
                x[edge_index[1]]], axis=1)
        for ix, layer in enumerate(self.layers):
            edge_embed = layer(edge_embed)
            if ix != len(self.layers) - 1:
                edge_embed = self.bns[ix](edge_embed)
                edge_embed = F.relu(edge_embed)

        adj = edge_embed.reshape(self.nnodes, self.nnodes)

        adj = (adj + adj.T)/2
        adj = torch.sigmoid(adj)
        adj = adj - torch.diag(torch.diag(adj, 0))
        return adj

    @torch.no_grad()
    def inference(self, x):
        # self.eval()
        adj_syn = self.forward(x)
        return adj_syn

    def reset_parameters(self):
        def weight_reset(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            if isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()
        self.apply(weight_reset)

class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()

        # GraphSAGE layers
        self.lin_self_1 = nn.Linear(in_dim, hidden_dim)
        self.lin_neigh_1 = nn.Linear(in_dim, hidden_dim)

        self.lin_self_2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin_neigh_2 = nn.Linear(hidden_dim, hidden_dim)

        # Graph-level readout → scalar
        self.readout = nn.Linear(hidden_dim, 1)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def sage_layer(self, X, A, lin_self, lin_neigh):
        """
        One GraphSAGE mean-aggregation layer
        """
        deg = A.sum(dim=1, keepdim=True).clamp(min=1.0)
        neigh_mean = (A @ X) / deg

        h = lin_self(X) + lin_neigh(neigh_mean)
        return F.relu(h)

    def forward(self, X, A):
        """
        X: (N, F)
        A: (N, N)
        returns: tensor
        """
        h = self.sage_layer(X, A, self.lin_self_1, self.lin_neigh_1)
        h = self.sage_layer(h, A, self.lin_self_2, self.lin_neigh_2)

        # Graph-level pooling
        h_graph = h.mean(dim=0)  # (hidden_dim,)

        # Scalar output
        out = self.readout(h_graph)  # (1,)
        return out # tensor
