
import torch 

@torch.no_grad()
def evaluate(gnn, data_list):
    gnn.eval()
    preds, ys = [], []

    for g in data_list:
        pred = gnn(g.X, g.A)
        preds.append(pred.item())
        ys.append(g.y.item())

    preds = torch.tensor(preds)
    ys = torch.tensor(ys)

    mse = torch.mean((preds - ys) ** 2)
    return mse.item()

def train_gnn_on_syn(
    gnn,
    syn_data,
    epochs=300,
    lr=1e-2
):
    gnn.train()
    optimizer = torch.optim.Adam(gnn.parameters(), lr=lr)

    for _ in range(epochs):
        total_loss = 0.0

        for g in syn_data:
            optimizer.zero_grad()

            pred = gnn(g.X, g.A)
            loss = (pred - g.y).pow(2).mean()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    return gnn

def detach_syn_data(syn_list):
    syn_detached = []

    from class_definition import GraphData

    for g in syn_list:
        syn_detached.append(
            GraphData(
                X=g.X.detach().clone(),
                A=g.A.detach().clone(),
                y=g.y.detach().clone(),
                requires_grad=False
            )
        )

    return syn_detached
