#train_real_list and train_syn_list contain objects having X, edge_index, y
#access using train_real_list[i].X, train_real_list[i].edge_index, train_real_list[i].y
#these are pytorch tensors 

import torch
from class_definition import BatchedGraphData

def _to_batched(data):
    """Convert list to BatchedGraphData if not already batched."""
    if isinstance(data, BatchedGraphData):
        return data
    return BatchedGraphData(data)

def l_syn(gnn, syn_list):
    """
    Compute loss on synthetic data.
    
    Args:
        gnn: Graph neural network model
        syn_list: List of GraphData objects OR BatchedGraphData
    
    Returns:
        Mean squared error loss (scalar)
    """
    batched = _to_batched(syn_list)
    
    y_pred = gnn(batched.X, batched.edge_index, batch=batched.batch)
    if isinstance(y_pred, tuple):
        y_pred = y_pred[0]
    
    y_pred = y_pred.view(-1)
    loss = torch.mean((y_pred - batched.y) ** 2)
    return loss

def l_q(gnn, real_list):
    """
    Compute loss on real data.
    
    Args:
        gnn: Graph neural network model
        real_list: List of GraphData objects OR BatchedGraphData
    
    Returns:
        Mean squared error loss (scalar)
    """
    batched = _to_batched(real_list)
    
    y_pred = gnn(batched.X, batched.edge_index, batch=batched.batch)
    if isinstance(y_pred, tuple):
        y_pred = y_pred[0]
    
    y_pred = y_pred.view(-1)
    loss = torch.mean((y_pred - batched.y) ** 2)
    return loss

def l_real(syn_list, lambda_X, lambda_Y, l_q_list):
    loss = torch.tensor(0.0, device=syn_list[0].X.device)

    # ----- average over q -----
    Q = len(l_q_list)
    for q in range(Q):
        loss += l_q_list[q]
    loss = loss / Q

    # ----- Frobenius norm regularization -----
    reg_X = 0.0
    reg_Y = 0.0

    for d in syn_list:
        reg_X += torch.sum(d.X ** 2)
        reg_Y += torch.sum(d.y ** 2)

    loss += (lambda_X / 2.0) * reg_X
    loss += (lambda_Y / 2.0) * reg_Y

    return loss
