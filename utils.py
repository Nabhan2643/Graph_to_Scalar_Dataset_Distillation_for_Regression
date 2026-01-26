import torch
from class_definition import BatchedGraphData, GraphData
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


def batch_real_data(real_list, batch_size, shuffle=True, seed=None):
    """
    Create batches of real graphs using BatchedGraphData.
    
    Shuffles the real data before batching and returns a list of 
    disconnected batched graphs (each batch is independent).
    
    Args:
        real_list: List of GraphData objects
        batch_size: Number of graphs per batch
        shuffle: Whether to shuffle before batching (default: True)
        seed: Random seed for reproducibility (optional)
    
    Returns:
        batches: List of BatchedGraphData objects, one per batch
        num_batches: Number of batches created
    
    Example:
        batches, num_batches = batch_real_data(train_real, batch_size=32, seed=42)
        for batch in batches:
            y_pred = gnn(batch.X, batch.edge_index, batch.batch)
    """
    
    # Shuffle if requested
    if shuffle:
        if seed is not None:
            torch.manual_seed(seed)
        indices = torch.randperm(len(real_list)).tolist()
        shuffled_list = [real_list[i] for i in indices]
    else:
        shuffled_list = real_list
    
    # Create batches
    batches = []
    num_full_batches = len(shuffled_list) // batch_size
    
    for i in range(num_full_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_graphs = shuffled_list[start_idx:end_idx]
        
        batches.append(BatchedGraphData(batch_graphs, requires_grad=False))
    
    # Handle remainder graphs (if any)
    remainder = len(shuffled_list) % batch_size
    if remainder > 0:
        batches.append(BatchedGraphData(shuffled_list[-remainder:], requires_grad=False))
    
    return batches, len(batches)


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
