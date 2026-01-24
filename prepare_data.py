"""
Data preparation module: Load, preprocess, and convert PyG data to GraphData format.
Can be run as a standalone script to prepare data before running main.py
"""

import os
import torch
from load_data import load_test_data, load_train_data
from preprocess import preprocess_graph_list_inplace
from utils import pyg_to_graphdata


def seed_worker(worker_id):
    """Seed worker for DataLoader reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    torch.manual_seed(worker_seed)


def prepare_graphdata(
    train_data_path: str,
    test_data_path: str,
    save_dir: str,
    target_key: str,
    device: str,
):
    """
    Load raw data, preprocess, and convert to GraphData format.
    
    Args:
        train_data_path: Path to training PyG data
        test_data_path: Path to test PyG data
        save_dir: Directory to save converted GraphData
        target_key: Target variable key
        device: Device to use (cpu/cuda)
    
    Returns:
        train_real: List of training GraphData objects
        test_data: List of test GraphData objects
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    # ============================================================
    # 4. LOAD DATA
    # ============================================================
    
    train_pyg = load_train_data(train_data_path, 'cpu')
    test_pyg = load_test_data(test_data_path, 'cpu')
    print("✔ Data loaded")
    
    # ============================================================
    # 5. PREPROCESS
    # ============================================================
    
    preprocess_graph_list_inplace(train_pyg, strategy="mean", device=device)
    preprocess_graph_list_inplace(test_pyg, strategy="mean", device=device)
    print("✔ Data preprocessed")
    
    # ============================================================
    # 6. CONVERT PyG → GraphData
    # ============================================================
    
    train_real = pyg_to_graphdata(
        train_pyg,
        target_key=target_key,
        requires_grad=False
    )
    
    test_data = pyg_to_graphdata(
        test_pyg,
        target_key=target_key,
        requires_grad=False
    )
    
    # Save the converted data
    torch.save(train_real, os.path.join(save_dir, "training_graphdata.pt"))
    torch.save(test_data, os.path.join(save_dir, "test_graphdata.pt"))
    
    print("✔ Converted & saved GraphData")
    
    return train_real, test_data


if __name__ == "__main__":
    # ============================================================
    # CONFIGURATION
    # ============================================================
    
    CFG = dict(
        device="cuda",
        target_key="stiffness",
        seed=42
    )
    
    # ============================================================
    # PATHS
    # ============================================================
    
    train_data_path = "data/train_dataset.pt"
    test_data_path = "data/test_dataset_2.pt"
    save_dir = "saved_data"
    
    device = CFG["device"]
    
    # ============================================================
    # PREPARE DATA
    # ============================================================
    
    print("=" * 60)
    print("Starting Data Preparation")
    print("=" * 60)
    
    train_real, test_data = prepare_graphdata(
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        save_dir=save_dir,
        target_key=CFG["target_key"],
        device=device
    )
    
    print("=" * 60)
    print(f"✔ Data preparation complete!")
    print(f"  - Training graphs: {len(train_real)}")
    print(f"  - Test graphs: {len(test_data)}")
    print(f"  - Feature dimension: {train_real[0].X.shape[1]}")
    print(f"  - Saved to: {save_dir}")
    print("=" * 60)
