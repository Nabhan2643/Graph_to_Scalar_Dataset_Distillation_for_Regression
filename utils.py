import torch
from class_definition import GraphData
import matplotlib.pyplot as plt
import os

## MAKES GRAPH UNDIRECTED

def _is_symmetric(edge_index):
    """
    Check if edge_index represents a symmetric (undirected) graph.
    
    Args:
        edge_index: (2, E) tensor
    
    Returns:
        bool: True if graph is symmetric
    """
    src, dst = edge_index
    reverse_edges = torch.stack([dst, src])
    
    # Convert to set for comparison
    edges = set(map(tuple, torch.t(edge_index).tolist()))
    reverse = set(map(tuple, torch.t(reverse_edges).tolist()))
    
    return edges == reverse


def pyg_to_graphdata( 
    pyg_graph_list,
    target_key="stiffness",
    requires_grad=False
):
    """
    Convert a list of PyG Data objects to GraphData objects.

    Args:
        pyg_graph_list (list): list of torch_geometric.data.Data
        target_key (str): name of target attribute (e.g. 'stiffness' or 'strength')
        requires_grad (bool): whether X and y require gradients (True for synthetic data)

    Returns:
        list[GraphData]
    """
    graphdata_list = []

    for data in pyg_graph_list:
        # Node features
        X = data.x

        # Extract edge_index directly from PyG Data object
        # PyG edge_index is already in format (2, E)
        edge_index = data.edge_index

        # For undirected graphs, ensure symmetric edges (both directions)
        if not _is_symmetric(edge_index):
            src, dst = edge_index
            reverse_edges = torch.stack([dst, src])
            edge_index = torch.cat([edge_index, reverse_edges], dim=1)
            # Remove duplicates if any
            edge_index = torch.unique(edge_index, dim=1)

        # Graph-level target
        y = getattr(data, target_key)

        graphdata_list.append(
            GraphData(X=X, edge_index=edge_index, y=y, requires_grad=requires_grad)
        )

    return graphdata_list

def save_scatter_preds_vs_targets(
    preds: torch.Tensor,
    ys: torch.Tensor,
    save_path: str,
    title: str = "Predictions vs Ground Truth",
    xlabel: str = "Ground Truth",
    ylabel: str = "Predictions"
):
    preds = preds.detach().cpu().view(-1)
    ys = ys.detach().cpu().view(-1)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.scatter(ys, preds, alpha=0.7)

    min_val = min(ys.min().item(), preds.min().item())
    max_val = max(ys.max().item(), preds.max().item())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
