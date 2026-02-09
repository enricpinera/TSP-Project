from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv
from torch_geometric.loader import DataLoader


def load_train_graphs(root: Path):
    graphs = []
    for pt_file in sorted(root.glob("*.pt"))[278:428]:
        data = torch.load(pt_file, weights_only=False)

        # Eliminar node_id perquè PyG el tracta com a node feature
        if hasattr(data, "node_id"):
            del data.node_id

        graphs.append(data)
    return graphs

root = Path("Datasets/train_pyg")
graphs = load_train_graphs(root)
print(f"Total graphs loaded: {len(graphs)}")

random.shuffle(graphs)
val_ratio = 0.2
val_size = int(len(graphs) * val_ratio)
train_size = len(graphs) - val_size
graphs_train = graphs[:train_size]
graphs_val = graphs[train_size:]
print(f"Train set: {len(graphs_train)} graphs")
print(f"Val set: {len(graphs_val)} graphs")

train_loader = DataLoader(graphs_train, batch_size=32, shuffle=True)
val_loader = DataLoader(graphs_val, batch_size=32, shuffle=False)

class TSPGNN(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=3):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # MLP per a la primera capa GINEConv
        self.mlp_in = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Capa GINE inicial
        self.convs = nn.ModuleList()
        self.convs.append(
            GINEConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                ),
                edge_dim=1
            )
        )

        # Capes GINE intermèdies
        for _ in range(num_layers - 1):
            self.convs.append(
                GINEConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim)
                    ),
                    edge_dim=1
                )
            )

        # Capa final: score per node
        self.node_classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.mlp_in(x)

        for conv in self.convs:
            h = conv(h, edge_index, edge_attr)
            h = F.relu(h)

        logits = self.node_classifier(h).squeeze(-1)
        return logits


torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = TSPGNN(
    hidden_dim=16,   # 32 o 64
    num_layers=2
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

criterion = nn.CrossEntropyLoss()

best_val_acc = 0.0
patience = 20
patience_counter = 0

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        losses = []
        correct = 0

        for i in range(len(batch.y)):
            start = batch.ptr[i].item()
            end = batch.ptr[i+1].item()

            logits_i = logits[start:end]          # [num_nodes_graph_i]
            target_i = batch.y[i].item()          # scalar

            # CrossEntropyLoss expects [1, num_classes] and [1]
            loss_i = F.cross_entropy(
                logits_i.view(1, -1),
                torch.tensor([target_i], device=device)
            )
            losses.append(loss_i)

            pred_i = logits_i.argmax().item()
            if pred_i == target_i:
                correct += 1

        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += correct
        total_graphs += len(batch.y)

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_graphs

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        losses = []
        correct = 0

        for i in range(len(batch.y)):
            start = batch.ptr[i].item()
            end = batch.ptr[i+1].item()

            logits_i = logits[start:end]
            target_i = batch.y[i].item()

            loss_i = F.cross_entropy(
                logits_i.view(1, -1),
                torch.tensor([target_i], device=device)
            )
            losses.append(loss_i)

            pred_i = logits_i.argmax().item()
            if pred_i == target_i:
                correct += 1

        loss = torch.stack(losses).mean()

        total_loss += loss.item()
        total_correct += correct
        total_graphs += len(batch.y)

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_graphs

    return avg_loss, accuracy



num_epochs = 200

for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, device)

    print(
        f"Epoch {epoch:03d} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )

    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pt")
        print("  → New best model saved")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break