#train_real_list and train_syn_list contain objects having X, edge_index, y
#access using train_real_list[i].X, train_real_list[i].edge_index, train_real_list[i].y
#gnn has a method that reinitialises itself

import torch
import random
import time

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

    device = next(mlp.parameters()).device

    # Ensure synthetic data is differentiable
    for g in train_syn_list:
        g.X.requires_grad_(True)
        g.y.requires_grad_(True)
        # Move to device
        g.X = g.X.to(device)
        g.y = g.y.to(device)
        g.edge_index = g.edge_index.to(device)

    for epoch in range(epochs):
        epoch_start_time = time.time()

        # --------------------------------------------------
        # Sample real batch B
        # --------------------------------------------------
        real_batch = random.sample(train_real_list, batch_size)

        # --------------------------------------------------
        # edge_index_tilde = g_phi(X_tilde)  (FIXED in this loop)
        # Use mlp.inference() to get filtered edges above threshold
        # --------------------------------------------------
        for g in train_syn_list:
            g.edge_index = mlp.inference(g.X)

        # --------------------------------------------------
        # Inner-loop: Q random initialisations
        # --------------------------------------------------
        l_q_list = []

        for q in range(Q):

            # Sample theta
            gnn.reset_parameters()

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

        # --------------------------------------------------
        # Updates
        # --------------------------------------------------
        with torch.no_grad():
            for g, gx, gy in zip(train_syn_list, grad_X, grad_y):
                g.X -= lr_X * gx
                g.y -= lr_y * gy

        # Calculate elapsed time for epoch
        epoch_elapsed_time = time.time() - epoch_start_time

        print(f"Epoch {epoch + 1}/{epochs} - Completed in {epoch_elapsed_time:.2f}s")
        print(f"--------------------------------------------------------------------------------------------------------")

    return train_syn_list, mlp, gnn

