#gnn has a method that reinitialises itself
import torch
import time

def distill(
    real_batches,
    train_syn_list,
    epochs,
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
    l_real,
    device
):
    """
    Distillation loop for graph neural network dataset distillation.
    
    Args:
        real_batches: List of BatchedGraphData objects (pre-created by batch_real_data)
        train_syn_list: List of GraphData objects (synthetic graphs to distill)
        epochs: Number of distillation epochs
        mlp: PGE model for edge generation
        Q: Number of random GNN initializations per epoch
        T: Number of training steps on synthetic data per initialization
        gnn: GraphSAGE model
        lr_gnn: Learning rate for GNN parameter updates
        lr_X: Learning rate for synthetic node features
        lr_y: Learning rate for synthetic targets
        lambda_X: Regularization weight for node features
        lambda_Y: Regularization weight for targets
        l_syn: Loss function for synthetic data
        l_q: Loss function for real data
        l_real: Meta-loss function
        device: Torch device
    
    Returns:
        train_syn_list: Updated synthetic graphs
        mlp: Updated PGE model
        gnn: Updated GNN model
    """

    # Ensure synthetic data is differentiable
    for g in train_syn_list:
        g.X.requires_grad_(True)
        g.y.requires_grad_(True)
        # Move to device
        g.X = g.X.to(device)
        g.y = g.y.to(device)
        g.edge_index = g.edge_index.to(device)

    num_batches = len(real_batches)

    for epoch in range(epochs):
        epoch_start_time = time.time()

        # --------------------------------------------------
        # Cycle through real batches
        # --------------------------------------------------
        batch_idx = epoch % num_batches
        real_batch = real_batches[batch_idx]

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

        print(f"Epoch {epoch + 1}/{epochs} - Batch {batch_idx + 1}/{num_batches} - Completed in {epoch_elapsed_time:.2f}s")
        print(f"--------------------------------------------------------------------------------------------------------")

    return train_syn_list, mlp, gnn
