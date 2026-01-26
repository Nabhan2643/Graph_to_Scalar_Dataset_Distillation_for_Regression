#assumes X,A,y are pytorch tensors

import torch

class GraphData:
    def __init__(self, X, edge_index, y, requires_grad=False):
        self.X = X
        self.edge_index = edge_index
        self.y = y

        if requires_grad:
            self.X.requires_grad_(True)
            self.y.requires_grad_(True)


class BatchedGraphData:
    """
    Pre-batched graph data for efficient processing.
    Concatenates multiple graphs into a single batched representation.
    
    Attributes:
        X: Concatenated node features (N, F) where N = sum of all nodes
        edge_index: Concatenated edge indices (2, E) with offsets applied
        y: Concatenated targets (B,) where B = number of graphs
        batch: Graph assignment vector (N,) with graph index per node
        num_graphs: Number of graphs in the batch
    """
    def __init__(self, graph_list, requires_grad=False):
        """
        Create a batched representation from a list of GraphData objects.
        
        Args:
            graph_list: List of GraphData objects
            requires_grad: Whether tensors require gradients
        """
        if not graph_list:
            raise ValueError("graph_list cannot be empty")
        
        self.num_graphs = len(graph_list)
        device = graph_list[0].X.device
        
        # Concatenate node features, offset edge indices, build batch vector
        Xs = []
        edges = []
        batch_idx = []
        n_nodes_running = 0
        
        for i, g in enumerate(graph_list):
            n = g.X.shape[0]
            Xs.append(g.X)
            edges.append(g.edge_index.long() + n_nodes_running)
            batch_idx.append(torch.full((n,), i, dtype=torch.long, device=device))
            n_nodes_running += n
        
        self.X = torch.cat(Xs, dim=0)
        self.edge_index = torch.cat(edges, dim=1)
        self.batch = torch.cat(batch_idx, dim=0)
        self.y = torch.stack([g.y.view(-1) for g in graph_list], dim=0).squeeze(-1).to(device)
        
        if requires_grad:
            self.X.requires_grad_(True)
            self.y.requires_grad_(True)
    
    def to(self, device):
        """Move all tensors to device."""
        self.X = self.X.to(device)
        self.edge_index = self.edge_index.to(device)
        self.batch = self.batch.to(device)
        self.y = self.y.to(device)
        return self
    
    def __repr__(self):
        return (f"BatchedGraphData(num_graphs={self.num_graphs}, "
                f"num_nodes={self.X.shape[0]}, "
                f"num_edges={self.edge_index.shape[1]}, "
                f"features={self.X.shape[1]})")
