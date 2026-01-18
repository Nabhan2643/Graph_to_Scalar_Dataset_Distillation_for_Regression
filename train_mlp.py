import torch
import torch.nn.functional as F

def train_pge(
    mlp,
    train_real,
    optimizer,
    epochs,
    device
):
    mlp.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for g in train_real:
            # --------------------------------------------------
            # Move data to device
            # --------------------------------------------------
            x = g.X.to(device)          # (N, F)
            adj_gt = g.A.to(device)     # (N, N)

            # --------------------------------------------------
            # Forward pass
            # --------------------------------------------------
            adj_pred = mlp(x)           # (N, N), sigmoid already applied

            # --------------------------------------------------
            # Mask: upper triangle, no diagonal
            # --------------------------------------------------
            mask = torch.triu(
                torch.ones_like(adj_gt, dtype=torch.bool),
                diagonal=1
            )

            # --------------------------------------------------
            # Loss: BCE on edges
            # --------------------------------------------------
            loss = F.binary_cross_entropy(
                adj_pred[mask],
                adj_gt[mask]
            )

            # --------------------------------------------------
            # Backprop
            # --------------------------------------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_real)
        print(f"[Epoch {epoch+1:03d}] PGE Loss: {avg_loss:.6f}")
