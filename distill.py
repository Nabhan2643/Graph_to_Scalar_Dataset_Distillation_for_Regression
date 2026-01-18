#train_real_list and train_syn_list contain objects having X,A,y
#access using train_real_list[i].X, train_real_list[i].A, train_real_list[i].y
#gnn has a method that reinitialises itself

# def distill(train_real_list,train_syn_list,epochs,batch_size,mlp,Q,T,gnn,lr_gnn,lr_X,lr_y,lr_mlp,l_syn,l_q,l_real):
#     return train_syn_list, mlp, gnn

import torch
import random

def distill(
    train_real_list,
    train_syn_list,
    epochs,
    batch_size,
    mlp,        # g_phi
    Q,
    T,
    gnn,        # theta
    lr_gnn,
    lr_X,
    lr_y,
    lambda_X,
    lambda_Y,
    l_syn,
    l_q,
    l_real
):

    device = next(mlp.parameters()).device  # check again

    # Ensure synthetic data is differentiable
    for g in train_syn_list:
        g.X.requires_grad_(True)
        g.y.requires_grad_(True)

    for epoch in range(epochs):

        # --------------------------------------------------
        # Sample real batch B
        # --------------------------------------------------
        real_batch = random.sample(train_real_list, batch_size)

        # --------------------------------------------------
        # A_tilde = g_phi(X_tilde)  (FIXED in this loop)
        # --------------------------------------------------
        for g in train_syn_list:
            g.A = mlp(g.X)

        # --------------------------------------------------
        # Inner-loop: Q random initialisations
        # --------------------------------------------------
        l_q_list = []

        for q in range(Q):

            # if(q==0):
            #     print(f"BEFORE RESET")
            #     for name, param in gnn.named_parameters():
            #         print(f"{name}:\n{param.data}\n")

            #     print(f"--------------------------------------------------------------------------------------------------------")
            #     print(f"--------------------------------------------------------------------------------------------------------")

            # Sample theta
            gnn.reset_parameters()  # check again the name

            # if(q==0):
            #     print(f"AFTER RESET")
            #     for name, param in gnn.named_parameters():
            #         print(f"{name}:\n{param.data}\n")
                
            #     print(f"--------------------------------------------------------------------------------------------------------")
            #     print(f"--------------------------------------------------------------------------------------------------------")  

            # ----- T steps on synthetic data -----
            for _ in range(T):
                Ls = l_syn(gnn, train_syn_list)

                grads = torch.autograd.grad(
                    Ls,
                    gnn.parameters(),
                    create_graph=True
                )

                with torch.no_grad():
                    for p, g in zip(gnn.parameters(), grads):
                        p -= lr_gnn * g

            # ----- Compute L_q -----
            Lq = l_q(gnn, real_batch)
            l_q_list.append(Lq)

        # --------------------------------------------------
        # Compute L_real
        # --------------------------------------------------
        L_real = l_real(
            train_syn_list,
            lambda_X,
            lambda_Y,
            l_q_list
        )

        # --------------------------------------------------
        # Meta-gradients
        # --------------------------------------------------
        syn_X = [g.X for g in train_syn_list]
        syn_y = [g.y for g in train_syn_list]

        grad_X = torch.autograd.grad(
            L_real,
            syn_X,
            retain_graph=True
        )

        grad_y = torch.autograd.grad(
            L_real,
            syn_y,
            retain_graph=True
        )

        # grad_phi = torch.autograd.grad(
        #     L_real,
        #     mlp.parameters()
        # )

        # --------------------------------------------------
        # Updates
        # --------------------------------------------------
        with torch.no_grad():
            for g, gx, gy in zip(train_syn_list, grad_X, grad_y):
                g.X -= lr_X * gx
                g.y -= lr_y * gy

            # for p, gp in zip(mlp.parameters(), grad_phi):
            #     p -= lr_mlp * gp
    
            # for p, gp in zip(mlp.parameters(), grad_phi):
            #     if gp is not None:
            #         p -= lr_mlp * gp
    
        g = train_syn_list[0]

        
        # print(f"Adjacency: {g.A}")
        # print(f"--------------------------------------------------------------------------------------------------------")
        # print(f"--------------------------------------------------------------------------------------------------------")
        # print(f"Feature Matrix: {g.X}")
        # print(f"--------------------------------------------------------------------------------------------------------")
        # print(f"--------------------------------------------------------------------------------------------------------")
        # print(f"label: {g.y}")
        # print(f"--------------------------------------------------------------------------------------------------------")
        # print(f"--------------------------------------------------------------------------------------------------------")
        print(f"Epoch {epoch} - Completed")
        print(f"--------------------------------------------------------------------------------------------------------")
        print(f"--------------------------------------------------------------------------------------------------------")
        print(f"--------------------------------------------------------------------------------------------------------")
        print(f"--------------------------------------------------------------------------------------------------------")

    return train_syn_list, mlp, gnn

