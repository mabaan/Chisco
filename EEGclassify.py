# -*- coding: utf-8 -*-
import os, math
import argparse
import pickle as pkl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from eegcnn import EEGNetClassifier, EEGNetGRUClassifier, EEGNetAttnClassifier
import torch.nn.functional as F
import random

def time_shift(x, max_shift=8):
    # x: [B, C, T], circular shift in time by up to ±max_shift samples
    if max_shift <= 0:
        return x
    B, C, T = x.shape
    s = torch.randint(low=-max_shift, high=max_shift+1, size=(B,), device=x.device)
    idx = (torch.arange(T, device=x.device).unsqueeze(0) - s.unsqueeze(1)) % T
    # gather per batch
    return x.gather(2, idx.unsqueeze(1).expand(B, C, T))

def mixup(x, y, alpha=0.2, n_classes=16):
    if alpha <= 0:
        return x, y, None
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1 - lam) * x[perm]
    y1, y2 = y, y[perm]
    # return “soft” targets for loss
    y_soft = (lam, y1, y2)
    return x_mixed, y, y_soft



def load_split(root: str, split: str):
    path = os.path.join(root, "preprocessed_pkl", f"{split}.pkl")
    with open(path, "rb") as f:
        obj = pkl.load(f)
    X = obj["X"].astype(np.float32)  # [N, 14, T]
    y = obj["y"].astype(np.int64)    # [N]
    return X, y, obj


def make_loaders(root: str, batch: int):
    Xtr, ytr, _ = load_split(root, "train")
    Xva, yva, _ = load_split(root, "val")
    Xte, yte, _ = load_split(root, "test")

    tr_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    va_ds = TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva))
    te_ds = TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))

    tr_dl = DataLoader(tr_ds, batch_size=batch, shuffle=True, drop_last=False)
    va_dl = DataLoader(va_ds, batch_size=batch, shuffle=False)
    te_dl = DataLoader(te_ds, batch_size=batch, shuffle=False)

    info = {
        "in_ch": Xtr.shape[1],
        "n_classes": int(max(ytr.max(), yva.max(), yte.max()) + 1),
        "sizes": {"train": len(tr_ds), "val": len(va_ds), "test": len(te_ds)},
        "train_labels": ytr,
        "T": Xtr.shape[2],
    }
    return tr_dl, va_dl, te_dl, info


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    correct1 = correct3 = total = 0
    loss_sum = 0.0
    for xb, yb in loader:
        xb = xb.to(device).float()
        # per-window, per-channel z-score
        xb = xb - xb.mean(dim=2, keepdim=True)
        xb = xb / (xb.std(dim=2, keepdim=True) + 1e-6)
        yb = yb.to(device).long()

        logits = model(xb)
        loss = ce(logits, yb)
        loss_sum += float(loss.item()) * yb.size(0)

        pred1 = logits.argmax(1)
        correct1 += int((pred1 == yb).sum().item())
        top3 = logits.topk(k=min(3, logits.size(1)), dim=1).indices
        correct3 += int((top3 == yb.unsqueeze(1)).any(dim=1).sum().item())
        total += int(yb.numel())

    acc1 = correct1 / max(total, 1)
    acc3 = correct3 / max(total, 1)
    return acc1, acc3, loss_sum / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/AREEG_Words")
    ap.add_argument("--cls", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--save", type=str, default="best_areeg_eegnet.pt")
    ap.add_argument("--head", type=str, default="attn", choices=["avg", "gru", "attn"])

    # EEGNet backbone widths and shapes
    ap.add_argument("--F1", type=int, default=16)
    ap.add_argument("--D", type=int, default=2)
    ap.add_argument("--F2", type=int, default=64)
    ap.add_argument("--k1", type=int, default=15)
    ap.add_argument("--k2", type=int, default=7)
    ap.add_argument("--P1", type=int, default=2)
    ap.add_argument("--P2", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--warmup", type=int, default=10)  # warmup epochs

    # GRU head params
    ap.add_argument("--rnn_hidden", type=int, default=64)
    ap.add_argument("--rnn_layers", type=int, default=1)

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tr_dl, va_dl, te_dl, info = make_loaders(args.root, args.batch)
    if info["in_ch"] != 14:
        raise SystemExit(f"Error: Expected 14 channels, got {info['in_ch']}. Rebuild PKLs.")
    n_classes = args.cls if args.cls is not None else info["n_classes"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"sizes: {info['sizes']}  classes={n_classes}  T={info['T']}")

    if args.head == "gru":
        model = EEGNetGRUClassifier(
            n_classes=n_classes, Chans=14,
            kernLength1=args.k1, kernLength2=args.k2,
            F1=args.F1, D=args.D, F2=args.F2,
            P1=args.P1, P2=args.P2, dropoutRate=args.dropout,
            rnn_hidden=args.rnn_hidden, rnn_layers=args.rnn_layers
        ).to(device)
    elif args.head == "attn":
        model = EEGNetAttnClassifier(
            n_classes=n_classes, Chans=14,
            kernLength1=args.k1, kernLength2=args.k2,
            F1=args.F1, D=args.D, F2=args.F2,
            P1=args.P1, P2=args.P2, dropoutRate=args.dropout
        ).to(device)
    else:
        model = EEGNetClassifier(
            n_classes=n_classes, Chans=14,
            kernLength1=args.k1, kernLength2=args.k2,
            F1=args.F1, D=args.D, F2=args.F2,
            P1=args.P1, P2=args.P2, dropoutRate=args.dropout
        ).to(device)

    # class weights from training labels
    ytr = info["train_labels"]
    counts = np.bincount(ytr, minlength=n_classes).astype(np.float32)
    weights = (counts.sum() / np.maximum(counts, 1.0))
    weights = torch.tensor(weights, dtype=torch.float, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # warmup + cosine decay
    def lr_lambda(epoch):
        if epoch < args.warmup:
            return (epoch + 1) / max(1, args.warmup)
        t = (epoch - args.warmup) / max(1, args.epochs - args.warmup)
        return 0.5 * (1.0 + math.cos(math.pi * t))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    # class-weighted CE with light label smoothing
    ce = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    best_va = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for xb, yb in tr_dl:
            xb = xb.to(device).float()
            # per-window, per-channel z-score
            xb = xb - xb.mean(dim=2, keepdim=True)
            xb = xb / (xb.std(dim=2, keepdim=True) + 1e-6)
            yb = yb.to(device).long()

            # ---- start of augmented training step ----
            opt.zero_grad()

            # simple augmentations
            xb = time_shift(xb, max_shift=8)  # ~62 ms at 128 Hz
            xb, yb_mixed_target, ysoft = mixup(xb, yb, alpha=0.2, n_classes=n_classes)

            logits = model(xb)
            if ysoft is None:
                loss = ce(logits, yb_mixed_target)
            else:
                lam, y1, y2 = ysoft
                loss = lam * ce(logits, y1) + (1 - lam) * ce(logits, y2)

            loss.backward()
            opt.step()
            # ---- end of augmented training step ----

            loss_sum += float(loss.item()) * yb.size(0) 
            count += int(yb.numel())

        tr_loss = loss_sum / max(count, 1)

        va_acc1, va_acc3, va_loss = evaluate(model, va_dl, device)
        if va_acc1 > best_va:
            best_va = va_acc1
            torch.save(model.state_dict(), args.save)

        if epoch % 5 == 0 or epoch == 1:
            lr_now = opt.param_groups[0]["lr"]
            print(f"epoch {epoch}  lr {lr_now:.2e}  train_loss {tr_loss:.4f}  "
                  f"val_loss {va_loss:.4f}  val@1 {va_acc1:.4f}  val@3 {va_acc3:.4f}  best {best_va:.4f}")

        sched.step()

    model.load_state_dict(torch.load(args.save, map_location=device))
    te_acc1, te_acc3, te_loss = evaluate(model, te_dl, device)
    print(f"test_loss {te_loss:.4f}  test@1 {te_acc1:.4f}  test@3 {te_acc3:.4f}")


if __name__ == "__main__":
    main()
