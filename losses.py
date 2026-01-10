#train_real_list and train_syn_list contain objects having X,A,y
#access using train_real_list[i].X, train_real_list[i].A, train_real_list[i].y
#these are pytorch tensors 

import torch

def l_syn(gnn, syn_list):
    loss = torch.tensor(0.0, device=syn_list[0].X.device) #CHECK AGAIN
    M = len(syn_list)

    for i in range(M):
        X = syn_list[i].X
        A = syn_list[i].A
        y = syn_list[i].y

        y_pred = gnn(X, A)
        loss += torch.mean((y_pred - y) ** 2)

    loss = loss / M
    return loss

def l_q(gnn, real_list):
    loss = torch.tensor(0.0, device=real_list[0].X.device) #CHECK AGAIN
    B = len(real_list)

    for i in range(B):
        X = real_list[i].X
        A = real_list[i].A
        y = real_list[i].y

        y_pred = gnn(X, A)
        loss += torch.mean((y_pred - y) ** 2)

    loss = loss / B
    return loss

def l_real(syn_list, lambda_X, lambda_Y, l_q_list):
    loss = torch.tensor(0.0, device=syn_list[0].X.device) #CHECK AGAIN

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
