"""
MLP (PGE) model training module: Initialize, train, and save the MLP model.
Can be run as a standalone script or imported as a module.
"""

import os
import torch
from models import PGE
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


def train_and_save_mlp(
    train_real,
    feat_dim: int,
    save_dir: str,
    device: str,
    pge_hidden_dim: int = 128,
    pge_layers: int = 3,
    lr_mlp: float = 1.0,
    pge_epochs: int = 15,
    pge_wd: float = 0.0,
):
    """
    Initialize, train, and save the MLP model.
    
    Args:
        train_real: Training data (list of GraphData objects)
        feat_dim: Feature dimension
        save_dir: Directory to save the trained model
        device: Device to use (cpu/cuda)
        pge_hidden_dim: Hidden dimension for PGE
        pge_layers: Number of layers for PGE
        lr_mlp: Learning rate for MLP training
        pge_epochs: Number of epochs to train
        pge_wd: Weight decay for optimizer
    
    Returns:
        mlp: Trained MLP model
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    # ============================================================
    # INITIALIZE MLP
    # ============================================================
    
    mlp = PGE(
        nfeat=feat_dim,
        nhid=pge_hidden_dim,
        nlayers=pge_layers,
        device=device
    ).to(device)
    
    print("✔ MLP initialized")
    
    # ============================================================
    # TRAIN MLP
    # ============================================================
    
    optimizer = torch.optim.Adam(
        mlp.parameters(),
        lr=lr_mlp,
        weight_decay=pge_wd
    )
    
    train_pge(
        mlp=mlp,
        train_real=train_real,
        optimizer=optimizer,
        epochs=pge_epochs,
        device=device
    )
    
    # ============================================================
    # SAVE MLP
    # ============================================================
    
    torch.save(
        mlp.state_dict(),
        os.path.join(save_dir, "pge_mlp.pt")
    )
    
    print("✔ MLP Trained and saved")
    
    return mlp


def load_or_train_mlp(
    train_real,
    feat_dim: int,
    save_dir: str,
    device: str,
    pge_hidden_dim: int = 128,
    pge_layers: int = 3,
    lr_mlp: float = 1.0,
    pge_epochs: int = 15,
    pge_wd: float = 0.0,
):
    """
    Load MLP from saved checkpoint if it exists, otherwise train and save it.
    
    Args:
        train_real: Training data (list of GraphData objects)
        feat_dim: Feature dimension
        save_dir: Directory where model is/will be saved
        device: Device to use (cpu/cuda)
        pge_hidden_dim: Hidden dimension for PGE
        pge_layers: Number of layers for PGE
        lr_mlp: Learning rate for MLP training
        pge_epochs: Number of epochs to train
        pge_wd: Weight decay for optimizer
    
    Returns:
        mlp: MLP model (loaded or newly trained)
    """
    
    mlp_path = os.path.join(save_dir, "pge_mlp.pt")
    
    # Try to load existing model
    if os.path.exists(mlp_path):
        mlp = PGE(
            nfeat=feat_dim,
            nhid=pge_hidden_dim,
            nlayers=pge_layers,
            device=device
        ).to(device)
        
        mlp.load_state_dict(torch.load(mlp_path, weights_only=True))
        print("✔ MLP loaded from saved checkpoint")
        return mlp
    
    # Train new model if checkpoint doesn't exist
    print("⚠ MLP checkpoint not found. Training new model...")
    return train_and_save_mlp(
        train_real=train_real,
        feat_dim=feat_dim,
        save_dir=save_dir,
        device=device,
        pge_hidden_dim=pge_hidden_dim,
        pge_layers=pge_layers,
        lr_mlp=lr_mlp,
        pge_epochs=pge_epochs,
        pge_wd=pge_wd,
    )


if __name__ == "__main__":
    # ============================================================
    # CONFIGURATION
    # ============================================================
    
    CFG = dict(
        device="cpu",
        target_key="stiffness",
        pge_hidden_dim=128,
        pge_layers=3,
        lr_mlp=1.0,
        pge_epochs=1,
        pge_wd=0.0,
        seed=42
    )
    
    # ============================================================
    # PATHS
    # ============================================================
    
    save_dir = "saved_data"
    device = CFG["device"]
    
    # ============================================================
    # LOAD TRAINING DATA
    # ============================================================
    
    training_graphdata_path = os.path.join(save_dir, "training_graphdata.pt")
    
    if os.path.exists(training_graphdata_path):
        train_real = torch.load(training_graphdata_path, weights_only=False)
        print("✔ Training data loaded")
    else:
        print("⚠ Training data not found. Please run prepare_data.py first.")
        exit(1)
    
    # ============================================================
    # TRAIN AND SAVE MLP
    # ============================================================
    
    print("=" * 60)
    print("Starting MLP Training")
    print("=" * 60)
    
    feat_dim = train_real[0].X.shape[1]
    
    mlp = train_and_save_mlp(
        train_real=train_real,
        feat_dim=feat_dim,
        save_dir=save_dir,
        device=device,
        pge_hidden_dim=CFG["pge_hidden_dim"],
        pge_layers=CFG["pge_layers"],
        lr_mlp=CFG["lr_mlp"],
        pge_epochs=CFG["pge_epochs"],
        pge_wd=CFG["pge_wd"]
    )
    
    print("=" * 60)
    print("✔ MLP training complete!")
    print(f"  - Model saved to: {os.path.join(save_dir, 'pge_mlp.pt')}")
    print("=" * 60)
