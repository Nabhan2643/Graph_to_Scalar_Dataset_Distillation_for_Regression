# LOAD DATA -> PREPROCESS -> CONVERT PYG TO DEFINED DATA CLASS USING UTILS -> INITIALIZATIONS FOR SYN DATA AND MODELS
#  AND SET ALL HYPERPARAMETERS AT ONE PLACE -> DISTILL DATA -> EVALUATE

# main.py
# LOAD DATA -> PREPROCESS -> CONVERT -> INIT SYN DATA & MODELS
# -> DISTILL -> EVALUATE

import os
import torch

from evaluate import detach_syn_data, evaluate, train_gnn_on_syn
from losses import l_q, l_real, l_syn
from utils import save_scatter_preds_vs_targets
from distill import distill
from models import PGE, GraphSAGE
from prepare_data import prepare_graphdata
from train_mlp import load_or_train_mlp
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
    epochs=15,
    batch_size=32,
    Q=30,
    T=150,

    # learning rates
    lr_gnn=1e-2,
    lr_X=100,
    lr_y=0.1,
    lr_mlp=1,
    pge_epochs=1,

    # loss weights
    lambda_X=0.001,
    lambda_Y=0.001,

    seed=42
)

# ============================================================
# 2. SET SEED
# ============================================================

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    torch.manual_seed(worker_seed)

# ============================================================
# 3. PATHS & LOAD DATA
# ============================================================

save_dir = "saved_data"

os.makedirs(save_dir, exist_ok=True)

device = CFG["device"]

# Try to load precomputed GraphData
training_graphdata_path = os.path.join(save_dir, "training_graphdata.pt")
test_graphdata_path = os.path.join(save_dir, "test_graphdata.pt")

if os.path.exists(training_graphdata_path) and os.path.exists(test_graphdata_path):
    train_real = torch.load(training_graphdata_path, weights_only=False)
    test_data = torch.load(test_graphdata_path, weights_only=False)
    print("✔ GraphData loaded from saved files")
else:
    print("⚠ No precomputed GraphData exists. Please run data preparation first.")
    train_real = None
    test_data = None

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
# 8. LOAD OR TRAIN MLP
# ============================================================

mlp = load_or_train_mlp(
    train_real=train_real,
    feat_dim=feat_dim,
    save_dir=save_dir,
    device=device,
    pge_hidden_dim=CFG["pge_hidden_dim"],
    pge_layers=CFG["pge_layers"],
    lr_mlp=CFG["lr_mlp"],
    pge_epochs=CFG["pge_epochs"],
    pge_wd=CFG.get("pge_wd", 0.0)
)

# ============================================================
# 9. INITIALIZE GNN
# ============================================================

gnn = GraphSAGE(
    in_dim=feat_dim,
    hidden_dim=CFG["sage_hidden_dim"]
).to(device)

print("✔ GNN initialized")

# ============================================================
# 10. DISTILLATION
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
    lambda_X=CFG["lambda_X"],
    lambda_Y=CFG["lambda_Y"],
    l_syn=l_syn,
    l_q=l_q,
    l_real=l_real
)

print("✔ Distillation complete")

# ============================================================
# 11. SAVE DISTILLED SYN DATA & MODELS
# ============================================================

torch.save(
    train_syn,
    os.path.join(save_dir, "train_syn_distilled.pt")
)

torch.save(
    gnn.state_dict(),
    os.path.join(save_dir, "gnn_distilled.pt")
)

print("✔ Distilled synthetic data and models saved")

# ============================================================
# 12. EVALUATION ON TEST DATA
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

train_real_mse, train_preds, train_ys = evaluate(gnn_eval, train_real)
save_scatter_preds_vs_targets(
    train_preds,
    train_ys,
    save_path="/Users/syednabhan/Documents/Graph to Scalar/modular_code/saved_data/plots/train_real_scatter.png",
    title="Synthetic-trained GNN on Train Data"
)
print(f"✔ Train Real MSE (trained on synthetic data): {train_real_mse:.6f}")
# for i in range(len(train_preds)):
#     print(
#         f"✔ Train Real (Predictions, Ground Truth): "
#         f"{train_preds[i].item():.6f} --------- {train_ys[i].item():.6f}"
#     )



test_mse, test_preds, test_ys = evaluate(gnn_eval, test_data)
save_scatter_preds_vs_targets(
    test_preds,
    test_ys,
    save_path="/Users/syednabhan/Documents/Graph to Scalar/modular_code/saved_data/plots/test_scatter.png",
    title="Synthetic-trained GNN on Test Data"
)
print(f"✔ Test MSE (trained on synthetic data): {test_mse:.6f}")
# for i in range(len(test_preds)):
#     print(
#         f"✔ Test Real (Predictions, Ground Truth): "
#         f"{test_preds[i].item():.6f} --------- {test_ys[i].item():.6f}"
#     )



