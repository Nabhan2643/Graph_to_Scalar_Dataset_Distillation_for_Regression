import torch
from class_definition import GraphData
import matplotlib.pyplot as plt
import os

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

        # Build adjacency matrix from edge_index (ignore edge_weight)
        num_nodes = X.size(0)
        A = torch.zeros(
            (num_nodes, num_nodes),
            device=X.device,
            dtype=X.dtype
        )

        src, dst = data.edge_index
        A[src, dst] = 1.0
        A[dst, src] = 1.0  # assume undirected graph

        # Graph-level target
        y = getattr(data, target_key)

        graphdata_list.append(
            GraphData(X=X, A=A, y=y, requires_grad=requires_grad)
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
    """
    Save scatter plot of predicted vs true values.

    Args:
        preds (torch.Tensor): shape (N,)
        ys (torch.Tensor): shape (N,)
        save_path (str): full path including filename (.png, .pdf, etc.)
    """

    preds = preds.detach().cpu()
    ys = ys.detach().cpu()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure()
    plt.scatter(ys, preds)

    # y = x reference line
    min_val = min(ys.min().item(), preds.min().item())
    max_val = max(ys.max().item(), preds.max().item())
    plt.plot([min_val, max_val], [min_val, max_val])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()   # VERY important to avoid memory leaks
