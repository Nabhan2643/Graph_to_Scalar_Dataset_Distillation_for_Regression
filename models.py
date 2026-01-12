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
        if args.dataset in ['ogbn-arxiv', 'arxiv', 'flickr']:
           nhid = 256
        if args.dataset in ['reddit']:
           nhid = 256
           if args.reduction_rate==0.01:
               nhid = 128
           nlayers = 3
           # nhid = 128

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

    def forward(self, x, inference=False):
        if self.args.dataset == 'reddit' and self.args.reduction_rate >= 0.01:
            edge_index = self.edge_index
            n_part = 5
            splits = np.array_split(np.arange(edge_index.shape[1]), n_part)
            edge_embed = []
            for idx in splits:
                tmp_edge_embed = torch.cat([x[edge_index[0][idx]],
                        x[edge_index[1][idx]]], axis=1)
                for ix, layer in enumerate(self.layers):
                    tmp_edge_embed = layer(tmp_edge_embed)
                    if ix != len(self.layers) - 1:
                        tmp_edge_embed = self.bns[ix](tmp_edge_embed)
                        tmp_edge_embed = F.relu(tmp_edge_embed)
                edge_embed.append(tmp_edge_embed)
            edge_embed = torch.cat(edge_embed)
        else:
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
        adj_syn = self.forward(x, inference=True)
        return adj_syn

    def reset_parameters(self):
        def weight_reset(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            if isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()
        self.apply(weight_reset)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch_sparse


# --------------------------------------------------
# GraphSAGE Convolution
# --------------------------------------------------
class SageConvolution(Module):
    def __init__(self, in_features, out_features, root_weight=True):
        super(SageConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.root_weight = root_weight

        self.weight_l = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_r = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_l.size(1))
        self.weight_l.data.uniform_(-stdv, stdv)
        self.weight_r.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, size=None):
        # neighbor aggregation
        h_neigh = x @ self.weight_l

        if isinstance(adj, torch_sparse.SparseTensor):
            out = torch_sparse.matmul(adj, h_neigh)
        else:
            out = torch.spmm(adj, h_neigh)

        # root/self contribution
        if self.root_weight:
            if size is not None:
                out = out + x[:size[1]] @ self.weight_r
            else:
                out = out + x @ self.weight_r

        return out + self.bias


# --------------------------------------------------
# GraphSAGE for GRAPH-LEVEL REGRESSION
# --------------------------------------------------
class GraphSageRegressor(nn.Module):
    def __init__(
        self,
        nfeat,
        nhid=128,
        nlayers=3,
        dropout=0.5,
        with_bn=True,
        readout="mean",   # mean | sum | max
        device=None
    ):
        super(GraphSageRegressor, self).__init__()
        assert device is not None
        self.device = device
        self.readout = readout
        self.dropout = dropout
        self.with_bn = with_bn

        # ----------- GNN layers -----------
        self.layers = nn.ModuleList()
        self.layers.append(SageConvolution(nfeat, nhid))

        for _ in range(nlayers - 1):
            self.layers.append(SageConvolution(nhid, nhid))

        if with_bn:
            self.bns = nn.ModuleList([nn.BatchNorm1d(nhid) for _ in range(nlayers - 1)])

        # ----------- Graph-level MLP head -----------
        self.regressor = nn.Sequential(
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Linear(nhid, 1)
        )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()
        for m in self.regressor:
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    # --------------------------------------------------
    # Full-batch forward (single graph)
    # --------------------------------------------------
    def forward(self, x, adj):
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            if i != len(self.layers) - 1:
                if self.with_bn:
                    x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        # ----------- READOUT -----------
        if self.readout == "mean":
            g = x.mean(dim=0)
        elif self.readout == "sum":
            g = x.sum(dim=0)
        elif self.readout == "max":
            g = x.max(dim=0)[0]
        else:
            raise ValueError("Invalid readout")

        # ----------- SCALAR OUTPUT -----------
        y = self.regressor(g)
        return y.squeeze()   # scalar

    # --------------------------------------------------
    # Neighbor-sampled forward (single graph)
    # --------------------------------------------------
    def forward_sampler(self, x, adjs):
        for i, (adj, _, size) in enumerate(adjs):
            x = self.layers[i](x, adj, size=size)
            if i != len(self.layers) - 1:
                if self.with_bn:
                    x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        # only target nodes
        x_target = x[:size[1]]

        if self.readout == "mean":
            g = x_target.mean(dim=0)
        elif self.readout == "sum":
            g = x_target.sum(dim=0)
        elif self.readout == "max":
            g = x_target.max(dim=0)[0]

        y = self.regressor(g)
        return y.squeeze()

