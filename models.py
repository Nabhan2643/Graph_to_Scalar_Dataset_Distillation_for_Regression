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

    def __init__(self, nfeat, nhid=128, nlayers=3, device=None, args=None, threshold=0.5):
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
        self.threshold = threshold
        self.reset_parameters()
        # self.cnt = 0
        # self.args = args
        # self.nnodes = nnodes

    def forward(self, x, edge_index=None):
        """
        Forward pass: only predict on upper triangle edges.
        Symmetry is enforced by mirroring predictions.
        """
        nnodes = x.shape[0]
        
        if edge_index is None:
            idx = torch.arange(nnodes, device=x.device)
            edge_index = torch.cartesian_prod(idx, idx).T
            # Keep only upper triangle (i < j)
            mask = edge_index[0] < edge_index[1]
            edge_index = edge_index[:, mask]
        
        src, dst = edge_index[0], edge_index[1]
        edge_embed = torch.cat([x[src], x[dst]], dim=1)
        
        for ix, layer in enumerate(self.layers):
            edge_embed = layer(edge_embed)
            if ix != len(self.layers) - 1:
                edge_embed = self.bns[ix](edge_embed)
                edge_embed = F.relu(edge_embed)
        
        edge_logits = edge_embed.squeeze(-1)
        edge_probs = torch.sigmoid(edge_logits)
        
        return edge_probs, edge_index

    @torch.no_grad()
    def inference(self, x, edge_index=None):
        """
        Inference: predict on upper triangle, then mirror to lower triangle.
        This ensures perfect symmetry.
        """
        edge_probs, edge_index = self.forward(x, edge_index)
        
        # Filter by threshold
        mask = edge_probs > self.threshold
        filtered_edges = edge_index[:, mask]
        
        # Create symmetric edge index by adding reverse edges
        src, dst = filtered_edges[0], filtered_edges[1]
        reverse_edges = torch.stack([dst, src])
        
        # Combine upper and lower triangle
        symmetric_edge_index = torch.cat([filtered_edges, reverse_edges], dim=1)
        
        return symmetric_edge_index

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

    def sage_layer(self, X, edge_index, lin_self, lin_neigh):
        """
        One GraphSAGE mean-aggregation layer using sparse edge-index.
        
        Args:
            X: Node features (N, F)
            edge_index: Edge indices (2, E) where edge_index[0] = src, edge_index[1] = dst
            lin_self: Linear layer for self features
            lin_neigh: Linear layer for neighbor features
        
        Returns:
            h: Updated node features (N, F)
        """
        src, dst = edge_index[0], edge_index[1]
        num_nodes = X.shape[0]
        
        # Compute in-degree for each node using scatter_add_
        deg = torch.zeros(num_nodes, device=X.device, dtype=X.dtype)
        deg.scatter_add_(0, src, torch.ones(src.shape[0], device=X.device, dtype=X.dtype))
        deg = deg.clamp(min=1.0).unsqueeze(1)  # (N, 1), avoid division by zero
        
        # Sparse neighbor aggregation using scatter_add_
        # Sum features from source nodes, scatter to destination nodes
        neigh_sum = torch.zeros_like(X)  # (N, F)
        neigh_sum.scatter_add_(0, src.unsqueeze(1).expand(-1, X.shape[1]), X[dst])
        
        # Normalize by degree
        neigh_mean = neigh_sum / deg  # (N, F)
        
        # GraphSAGE: combine self and neighbor features
        h = lin_self(X) + lin_neigh(neigh_mean)
        return F.relu(h)

    def forward(self, X, edge_index, batch=None):
        """
        Forward pass using edge index representation.
        
        Args:
            X: Node features (N, F)
            edge_index: Edge indices (2, E)
            batch: Optional batch vector (N,) with graph indices for each node
        
        Returns:
            out: Graph-level prediction(s). Scalar if single graph, vector if batched.
        """
        h = self.sage_layer(X, edge_index, self.lin_self_1, self.lin_neigh_1)
        h = self.sage_layer(h, edge_index, self.lin_self_2, self.lin_neigh_2)

        # If no batch vector provided, keep old behavior (single-graph)
        if batch is None:
            h_graph = h.mean(dim=0)  # (hidden_dim,)
            out = self.readout(h_graph)  # (1,)
            return out.squeeze(-1)  # scalar

        # Batched graphs: compute per-graph mean pooling using scatter_add_
        num_graphs = int(batch.max().item()) + 1

        # Sum node features per graph
        h_sum = torch.zeros((num_graphs, h.size(1)), device=h.device, dtype=h.dtype)
        h_sum.scatter_add_(0, batch.unsqueeze(1).expand(-1, h.size(1)), h)

        # Count nodes per graph
        counts = torch.zeros((num_graphs,), device=h.device, dtype=h.dtype)
        counts.scatter_add_(0, batch, torch.ones(batch.size(0), device=h.device, dtype=h.dtype))
        counts = counts.clamp(min=1.0).unsqueeze(1)

        h_mean = h_sum / counts  # (num_graphs, hidden_dim)

        # Graph-level readout for each graph
        out = self.readout(h_mean)  # (num_graphs, 1)
        return out.squeeze(-1)  # (num_graphs,)