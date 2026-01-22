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

    def __init__(self, nfeat, nhid=128, nlayers=3, device=None, args=None):
        super(PGE, self).__init__()

        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(nfeat*2, nhid))
        self.bns = torch.nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(nhid))
        for i in range(nlayers-2):
            self.layers.append(nn.Linear(nhid, nhid))
            self.bns.append(nn.BatchNorm1d(nhid))
        self.layers.append(nn.Linear(nhid, 1))

        
        # edge_index = np.array(list(product(range(nnodes), range(nnodes))))
        # self.edge_index = edge_index.T
        # self.nnodes = nnodes
        self.device = device
        self.reset_parameters()
        # self.cnt = 0
        # self.args = args
        # self.nnodes = nnodes

    def forward(self, x):
        nnodes = x.shape[0]
        idx = torch.arange(nnodes, device=x.device)
        edge_index = torch.cartesian_prod(idx, idx).T
        edge_embed = torch.cat([x[edge_index[0]],
                x[edge_index[1]]], axis=1)
        for ix, layer in enumerate(self.layers):
            edge_embed = layer(edge_embed)
            if ix != len(self.layers) - 1:
                edge_embed = self.bns[ix](edge_embed)
                edge_embed = F.relu(edge_embed)

        adj = edge_embed.reshape(nnodes, nnodes)

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
        One GraphSAGE mean-aggregation layer using sparse edge-index based aggregation.
        
        OPTIMIZATION: Replaced dense matrix multiplication (A @ X) with sparse aggregation
        using scatter_add_ operations. This is crucial for scalability.
        
        Time complexity:  O(E * F) instead of O(N^2 * F) where E=edges, N=nodes, F=features
        Space complexity: O(E) instead of O(N^2) for adjacency representation
        
        Example: For 100-node graphs with ~500 edges (5% density):
        - Dense: 10,000 entries × 4 bytes = 40 KB
        - Sparse: 500 edges × 2 indices × 8 bytes = 8 KB (80% memory savings)
        
        Supports both input formats:
        - Dense adjacency matrix A: (N, N) - will be converted to edge_index internally
        - Edge index tuple (src, dst): already sparse, processed directly
        """
        # Handle both dense adjacency matrix and edge_index inputs
        if isinstance(A, tuple):  # Edge index format (src, dst)
            src, dst = A
        elif hasattr(A, 'dim') and A.dim() == 2:  # Dense adjacency matrix (N, N)
            # Extract edges from dense adjacency using nonzero
            edge_index = A.nonzero(as_tuple=True)  # Returns (src, dst) as separate tensors
            src, dst = edge_index
        else:
            raise ValueError("A must be either a dense adjacency matrix or (src, dst) edge index tuple")
        
        # Compute degree for each node using scatter_add_
        # This counts in-degree for each node efficiently
        num_nodes = X.shape[0]
        deg = torch.zeros(num_nodes, device=X.device, dtype=X.dtype)
        deg.scatter_add_(0, src, torch.ones(src.shape[0], device=X.device, dtype=X.dtype))
        deg = deg.clamp(min=1.0).unsqueeze(1)  # (N, 1), clamped to avoid division by zero
        
        # Sparse neighbor aggregation: sum neighbor features using scatter_add_
        # Instead of dense matrix multiply, gather features from destinations and scatter to sources
        neigh_sum = torch.zeros_like(X)  # (N, F)
        neigh_sum.scatter_add_(0, src.unsqueeze(1).expand(-1, X.shape[1]), X[dst])
        
        # Normalize by degree: divide each aggregated feature by node degree
        neigh_mean = neigh_sum / deg  # (N, F)
        
        # Apply GraphSAGE transformation: combine self and neighbor features
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
