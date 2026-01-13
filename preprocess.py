from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import torch

def preprocess_features(X, strategy='mean', scaler=None, device = "cpu"):

    # Ensure input is in numpy format
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    # Convert to torch tensor and move to device
    return torch.tensor(X_scaled, dtype=torch.float32, device=device), scaler


def preprocess_graph_list_inplace(graph_list, strategy='mean', device='cpu'):
    
    x_all = torch.cat([g.x.to(device) for g in graph_list], dim=0)

    # 1) Preprocess node features (returns X_proc of same shape, plus scaler)
    X_proc, x_scaler = preprocess_features(x_all, strategy=strategy, device=device)

    # 2) Write back into each graph x, leave edge_weight alone
    offset = 0
    for g in graph_list:
        n_nodes = g.x.size(0)
        g.x = X_proc[offset : offset + n_nodes]
        offset += n_nodes

    return x_scaler
