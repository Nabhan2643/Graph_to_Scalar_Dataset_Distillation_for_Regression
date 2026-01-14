# LOAD DATA -> PREPROCESS -> CONVERT PYG TO DEFINED DATA CLASS USING UTILS -> INITIALIZATIONS FOR SYN DATA AND MODELS
#  AND SET ALL HYPERPARAMETERS AT ONE PLACE -> DISTILL DATA -> EVALUATE

# main.py
# LOAD DATA -> PREPROCESS -> CONVERT -> INIT SYN DATA & MODELS
# -> DISTILL -> EVALUATE

import os
import random
import torch

from evaluate import detach_syn_data, evaluate, train_gnn_on_syn
from load_data import load_test_data, load_train_data
from losses import l_q, l_real, l_syn
from preprocess import preprocess_graph_list_inplace
from utils import pyg_to_graphdata
from distill import distill
from models import PGE, GraphSAGE

# ============================================================
# 1. GLOBAL CONFIG / HYPERPARAMETERS (ONE PLACE)
# ============================================================

CFG = dict(
    device="cpu",

    # data
    target_key="stiffness",
    syn_graphs=10,          # number of synthetic graphs
    syn_nodes=20,           # nodes per synthetic graph

    # model
    sage_hidden_dim=64,
    pge_hidden_dim=128,
    pge_layers=3,

    # distillation
    epochs=200,
    batch_size=8,
    Q=3,
    T=5,

    # learning rates
    lr_gnn=1e-2,
    lr_X=1e-2,
    lr_y=1e-2,
    lr_mlp=1e-3,

    # loss weights
    lambda_X=1.0,
    lambda_Y=1.0,

    seed=42
)

# ============================================================
# 2. SET SEED
# ============================================================

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    torch.manual_seed(worker_seed)

# ============================================================
# 3. PATHS
# ============================================================

train_data_path = r"data\train_dataset.pt"
test_data_path  = r"data\test_dataset_2.pt"
save_dir = r"saved_data"

os.makedirs(save_dir, exist_ok=True)

device = CFG["device"]

# ============================================================
# 4. LOAD DATA
# ============================================================

train_pyg = load_train_data(train_data_path, device)
test_pyg  = load_test_data(test_data_path, device)
print("✔ Data loaded")

# ============================================================
# 5. PREPROCESS
# ============================================================

preprocess_graph_list_inplace(train_pyg, strategy="mean", device=device)
preprocess_graph_list_inplace(test_pyg,  strategy="mean", device=device)
print("✔ Data preprocessed")

# ============================================================
# 6. CONVERT PyG → GraphData
# ============================================================

train_real = pyg_to_graphdata(
    train_pyg,
    target_key=CFG["target_key"],
    requires_grad=False
)

test_data = pyg_to_graphdata(
    test_pyg,
    target_key=CFG["target_key"],
    requires_grad=False
)

torch.save(train_real, os.path.join(save_dir, "training_graphdata.pt"))
torch.save(test_data, os.path.join(save_dir, "test_graphdata.pt"))

print("✔ Converted & saved GraphData")

# ============================================================
# 7. INITIALIZE SYNTHETIC DATA (SEEDED ONLY HERE)
# ============================================================

syn_gen = torch.Generator(device=device)
syn_gen.manual_seed(CFG["seed"])

feat_dim = train_real[0].X.shape[1]

train_syn = []
for _ in range(CFG["syn_graphs"]):
    X_syn = torch.randn(
        CFG["syn_nodes"],
        feat_dim,
        device=device,
        generator=syn_gen,
        requires_grad=True
    )

    y_syn = torch.randn(
        1,
        device=device,
        generator=syn_gen,
        requires_grad=True
    )

    A_syn = torch.zeros(
        CFG["syn_nodes"],
        CFG["syn_nodes"],
        device=device
    )

    from class_definition import GraphData
    train_syn.append(
        GraphData(X=X_syn, A=A_syn, y=y_syn, requires_grad=True)
    )


print("✔ Synthetic graphs initialized")

# ============================================================
# 8. INITIALIZE MODELS
# ============================================================

mlp = PGE(
    nfeat=feat_dim,
    nnodes=CFG["syn_nodes"],
    nhid=CFG["pge_hidden_dim"],
    nlayers=CFG["pge_layers"],
    device=device
).to(device)

gnn = GraphSAGE(
    in_dim=feat_dim,
    hidden_dim=CFG["sage_hidden_dim"]
).to(device)

print("✔ Models initialized")

# ============================================================
# 9. DISTILLATION
# ============================================================

train_syn, mlp, gnn = distill(
    train_real_list=train_real,
    train_syn_list=train_syn,
    epochs=CFG["epochs"],
    batch_size=CFG["batch_size"],
    mlp=mlp,
    Q=CFG["Q"],
    T=CFG["T"],
    gnn=gnn,
    lr_gnn=CFG["lr_gnn"],
    lr_X=CFG["lr_X"],
    lr_y=CFG["lr_y"],
    lr_mlp=CFG["lr_mlp"],
    lambda_X=CFG["lambda_X"],
    lambda_Y=CFG["lambda_Y"],
    l_syn=l_syn,
    l_q=l_q,
    l_real=l_real
)

print("✔ Distillation complete")

# ============================================================
# 10. SAVE DISTILLED SYN DATA & MODELS
# ============================================================

torch.save(
    train_syn,
    os.path.join(save_dir, "train_syn_distilled.pt")
)

torch.save(
    mlp.state_dict(),
    os.path.join(save_dir, "pge_mlp_distilled.pt")
)

torch.save(
    gnn.state_dict(),
    os.path.join(save_dir, "gnn_distilled.pt")
)

print("✔ Distilled synthetic data and models saved")

# ============================================================
# 11. EVALUATION ON TEST DATA
# ============================================================
train_syn_eval = detach_syn_data(train_syn)

gnn_eval = GraphSAGE(
    in_dim=feat_dim,
    hidden_dim=CFG["sage_hidden_dim"]
).to(device)

gnn_eval = train_gnn_on_syn(
    gnn=gnn_eval,
    syn_data=train_syn_eval,
    epochs=300,
    lr=1e-2
)

test_mse = evaluate(gnn_eval, test_data)
print(f"✔ Test MSE (trained on synthetic data): {test_mse:.6f}")

