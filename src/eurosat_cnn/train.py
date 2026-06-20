"""Train LandCoverCNN on an ImageFolder dataset (real or synthetic)."""
from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .data import EuroSATDataset, make_synthetic_dataset, train_val_split, CLASSES
from .model import LandCoverCNN, count_parameters


def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def train(data_dir: str, epochs: int = 5, batch_size: int = 32, lr: float = 1e-3,
          seed: int = 0, out: str = "landcover_cnn.pt", device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    ds = EuroSATDataset(data_dir)
    tr_idx, va_idx = train_val_split(len(ds), 0.2, seed)
    tr = DataLoader(Subset(ds, tr_idx), batch_size=batch_size, shuffle=True)
    va = DataLoader(Subset(ds, va_idx), batch_size=batch_size)

    model = LandCoverCNN(num_classes=len(CLASSES)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = torch.nn.CrossEntropyLoss()
    print(f"Model params: {count_parameters(model):,} | device: {device}")

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        acc = evaluate(model, va, device)
        print(f"epoch {ep:02d} | loss {np.mean(losses):.4f} | val_acc {acc:.3f}")

    torch.save({"state_dict": model.state_dict(), "classes": CLASSES}, out)
    print(f"Saved -> {out}")
    return model


def main():
    p = argparse.ArgumentParser(description="Train EuroSAT land-cover CNN")
    p.add_argument("--data-dir", default=None,
                   help="ImageFolder root. If omitted, a synthetic set is built.")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", default="landcover_cnn.pt")
    args = p.parse_args()

    data_dir = args.data_dir
    if data_dir is None:
        data_dir = os.path.join(tempfile.mkdtemp(), "synthetic_eurosat")
        print("No --data-dir given; generating synthetic dataset...")
        make_synthetic_dataset(data_dir, per_class=80)
    train(data_dir, args.epochs, args.batch_size, args.lr, out=args.out)


if __name__ == "__main__":
    main()
