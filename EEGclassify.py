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
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             norm: str = "per", extra_norm: str = "none",
             global_mean: torch.Tensor | None = None, global_std: torch.Tensor | None = None):
    """Evaluate model with normalization consistent with training.

    norm: 'per' or 'global'. If 'global', global_mean/std must be provided.
    extra_norm: handling for extra channels when C>14.
    """
    model.eval()
    ce = nn.CrossEntropyLoss()
    correct1 = correct3 = total = 0
    loss_sum = 0.0
    for xb, yb in loader:
        xb = xb.to(device).float()
        if norm == "per":
            if xb.size(1) > 14:
                eeg_part = xb[:, :14, :]
                eeg_part = eeg_part - eeg_part.mean(dim=2, keepdim=True)
                eeg_part = eeg_part / (eeg_part.std(dim=2, keepdim=True) + 1e-6)
                extra = xb[:, 14:, :]
                if extra_norm == "per":
                    extra = extra - extra.mean(dim=2, keepdim=True)
                    extra = extra / (extra.std(dim=2, keepdim=True) + 1e-6)
                elif extra_norm == "global" and global_mean is not None:
                    extra_mean = global_mean[:, 14:, :]
                    extra_std = global_std[:, 14:, :]
                    extra = (extra - extra_mean) / (extra_std + 1e-6)
                xb = torch.cat([eeg_part, extra], dim=1)
            else:
                xb = xb - xb.mean(dim=2, keepdim=True)
                xb = xb / (xb.std(dim=2, keepdim=True) + 1e-6)
        else:  # global
            xb = (xb - global_mean) / (global_std + 1e-6)
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
    ap.add_argument("--dataset", type=str, default="areeg", choices=["areeg", "bci"], 
                    help="Dataset type: 'areeg' for AREEG_Words, 'bci' for BCI Competition 2020")
    ap.add_argument("--cls", type=int, default=None, help="Number of classes (auto-detect if None)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--save", type=str, default=None, help="Model save path (auto-generated if None)")
    ap.add_argument("--head", type=str, default="attn", choices=["avg", "gru", "attn", "gruattn", "transformer"])

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
    ap.add_argument("--sched", type=str, default="cosine", choices=["cosine", "onecycle", "multistep"], help="LR scheduler type")
    ap.add_argument("--milestones", type=str, default="", help="Comma-separated epoch milestones for multistep scheduler (e.g. 100,150)")
    ap.add_argument("--gamma", type=float, default=0.5, help="LR decay factor for multistep")
    ap.add_argument("--max_lr", type=float, default=None, help="Max LR for OneCycle (defaults to --lr * 10 if None)")
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--early_stop", type=int, default=0, help="Early stopping patience (0 disables)")
    ap.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping max norm (<=0 disables)")
    ap.add_argument("--mixup_alpha", type=float, default=0.2, help="Mixup alpha (0 disables)")
    ap.add_argument("--time_shift", type=int, default=8, help="Max +/- samples for time shift (0 disables)")
    ap.add_argument("--min_delta", type=float, default=0.0, help="Minimum improvement to reset patience")
    ap.add_argument("--early_metric", type=str, default="acc", choices=["acc","loss"], help="Metric for early stopping (acc=val@1, loss=val_loss)")
    ap.add_argument("--mixup_start", type=int, default=0, help="Epoch to start applying mixup (curriculum)")
    ap.add_argument("--channel_dropout", type=float, default=0.0, help="Probability of dropping (zeroing) each EEG channel per sample")
    ap.add_argument("--channel_dropout_start", type=int, default=0, help="Epoch to start applying channel dropout")
    ap.add_argument("--gaussian_noise", type=float, default=0.0, help="Std factor of additive Gaussian noise relative to per-channel std")
    ap.add_argument("--gaussian_noise_start", type=int, default=0, help="Epoch to start applying gaussian noise")
    ap.add_argument("--no_class_weights", action="store_true", help="Disable class weighting in loss (use uniform)")
    ap.add_argument("--norm", type=str, default="per", choices=["per","global"], help="Normalization strategy: per (per-sample z), global (dataset stats)")
    ap.add_argument("--time_cutmix_alpha", type=float, default=0.0, help="Temporal CutMix Beta alpha (0 disables). Applied if mixup inactive this batch.")
    ap.add_argument("--time_cutmix_start", type=int, default=0, help="Epoch to start temporal CutMix (0 = immediate)")
    ap.add_argument("--swa_start", type=int, default=0, help="Epoch to start SWA (0 disables)")
    ap.add_argument("--swa_lr", type=float, default=None, help="LR used during SWA phase (defaults to current LR if None)")
    ap.add_argument("--ls_decay_start", type=int, default=0, help="Epoch to start decaying label smoothing toward 0")
    ap.add_argument("--extra_norm", type=str, default="none", choices=["none","global","per"], help="Normalization strategy for extra (non-EEG) channels when C>14")

    # GRU head params
    ap.add_argument("--rnn_hidden", type=int, default=64)
    ap.add_argument("--rnn_layers", type=int, default=1)
    ap.add_argument("--rnn_layernorm", action="store_true", help="Apply LayerNorm after GRU outputs (stabilizes training)")
    ap.add_argument("--use_gru_proj", action="store_true", help="Enable residual MLP projection after GRU (gruattn head only)")
    ap.add_argument("--proj_hidden", type=int, default=0, help="Hidden size for GRU projection MLP (requires --use_gru_proj)")

    # Augmentation toggles
    ap.add_argument("--no_mixup", action="store_true", help="Disable mixup augmentation")
    ap.add_argument("--no_timeshift", action="store_true", help="Disable time shift augmentation")

    # Transformer specific arguments
    ap.add_argument('--num_layers', type=int, default=2)
    ap.add_argument('--dim_feedforward', type=int, default=128)
    ap.add_argument('--ensemble', type=int, default=1)

    args = ap.parse_args()

    # Set dataset-specific defaults
    if args.dataset == "bci":
        if args.root == "data/AREEG_Words":  # Default wasn't changed
            args.root = "data/BCI_Words"
        if args.save is None:
            args.save = f"best_bci_{args.head}.pt"
    else:  # areeg
        if args.save is None:
            args.save = f"best_areeg_{args.head}.pt"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tr_dl, va_dl, te_dl, info = make_loaders(args.root, args.batch)
    # Support both 14 (without band powers) and 19 (with band powers) channels
    if info["in_ch"] not in [14, 19]:
        raise SystemExit(f"Error: Expected 14 or 19 channels, got {info['in_ch']}. Rebuild PKLs.")
    
    # Extract the actual number of EEG channels (14) and total channels
    n_eeg_channels = 14  # Always 14 EEG channels
    n_total_channels = info["in_ch"]  # 14 or 19 total channels
    n_classes = args.cls if args.cls is not None else info["n_classes"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Data path: {args.root}")
    print(f"Device: {device}")
    print(f"Sizes: {info['sizes']}  Classes: {n_classes}  Channels: {n_total_channels}  Time points: {info['T']}")

    if args.head == "gru":
        model = EEGNetGRUClassifier(
            n_classes=n_classes, Chans=n_total_channels,
            kernLength1=args.k1, kernLength2=args.k2,
            F1=args.F1, D=args.D, F2=args.F2,
            P1=args.P1, P2=args.P2, dropoutRate=args.dropout,
            rnn_hidden=args.rnn_hidden, rnn_layers=args.rnn_layers, use_layernorm=args.rnn_layernorm
        ).to(device)
    elif args.head == "attn":
        model = EEGNetAttnClassifier(
            n_classes=n_classes, Chans=n_total_channels,
            kernLength1=args.k1, kernLength2=args.k2,
            F1=args.F1, D=args.D, F2=args.F2,
            P1=args.P1, P2=args.P2, dropoutRate=args.dropout
        ).to(device)
    elif args.head == "gruattn":
        from eegcnn import EEGNetGRUAttnClassifier
        model = EEGNetGRUAttnClassifier(
            n_classes=n_classes, Chans=n_total_channels,
            kernLength1=args.k1, kernLength2=args.k2,
            F1=args.F1, D=args.D, F2=args.F2,
            P1=args.P1, P2=args.P2, dropoutRate=args.dropout,
            rnn_hidden=args.rnn_hidden, rnn_layers=args.rnn_layers,
            use_layernorm=args.rnn_layernorm,
            use_proj=args.use_gru_proj, proj_hidden=args.proj_hidden
        ).to(device)
    elif args.head == "transformer":
        from eegcnn import EEGNetTransformerClassifier
        model = EEGNetTransformerClassifier(
            n_classes=n_classes, Chans=n_total_channels,
            kernLength1=args.k1, kernLength2=args.k2,
            F1=args.F1, D=args.D, F2=128,
            P1=4, P2=4, dropoutRate=args.dropout,  # Use smaller pooling for transformer
            num_layers=args.num_layers, dim_feedforward=args.dim_feedforward, ensemble=args.ensemble
        ).to(device)
    else:
        model = EEGNetClassifier(
            n_classes=n_classes, Chans=n_total_channels,
            kernLength1=args.k1, kernLength2=args.k2,
            F1=args.F1, D=args.D, F2=args.F2,
            P1=args.P1, P2=args.P2, dropoutRate=args.dropout
        ).to(device)

    # class weights from training labels
    ytr = info["train_labels"]
    counts = np.bincount(ytr, minlength=n_classes).astype(np.float32)
    if args.no_class_weights:
        weights = torch.ones(n_classes, dtype=torch.float, device=device)
    else:
        weights = (counts.sum() / np.maximum(counts, 1.0))
        weights = torch.tensor(weights, dtype=torch.float, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    # Scheduler selection (warmup + chosen policy)
    steps_per_epoch = len(tr_dl)
    if args.sched == "cosine":
        def lr_lambda(epoch):
            if epoch < args.warmup:
                return (epoch + 1) / max(1, args.warmup)
            # Single cosine decay (no restarts)
            progress = (epoch - args.warmup) / max(1, (args.epochs - args.warmup))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    elif args.sched == "onecycle":
        max_lr = args.max_lr if args.max_lr is not None else args.lr * 10.0
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=max_lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch, pct_start=0.1, anneal_strategy='cos', div_factor=max_lr/args.lr
        )
    else:  # multistep
        milestones = [int(m.strip()) for m in args.milestones.split(',') if m.strip().isdigit()]
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=args.gamma) if milestones else torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda e:1.0)

    # Class-weighted CE with configurable label smoothing
    ce = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    kl = nn.KLDivLoss(reduction='batchmean')

    best_va = 0.0
    patience_counter = 0
    # Pre-compute global mean/std if requested
    global_mean = None
    global_std = None
    if args.norm == "global":
        with torch.no_grad():
            s_count = 0
            s_sum = 0.0
            s_sumsq = 0.0
            for xb, _ in tr_dl:
                xb = xb.float()
                s_sum += xb.sum(dim=(0,2))  # per-channel sum over batch+time
                s_sumsq += (xb**2).sum(dim=(0,2))
                s_count += xb.shape[0]*xb.shape[2]
            global_mean = (s_sum / s_count).view(1,-1,1)
            var = (s_sumsq / s_count).view(1,-1,1) - global_mean**2
            global_std = torch.sqrt(torch.clamp(var, min=1e-6))
            global_mean = global_mean.to(device)
            global_std = global_std.to(device)

    # SWA state
    swa_model = None
    swa_n = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        correct_train = 0
        total_train = 0
        for batch_idx, (xb, yb) in enumerate(tr_dl):
            xb = xb.to(device).float()
            # Normalization
            if args.norm == "per":
                if xb.size(1) > 14:
                    eeg_part = xb[:, :14, :]
                    eeg_part = eeg_part - eeg_part.mean(dim=2, keepdim=True)
                    eeg_part = eeg_part / (eeg_part.std(dim=2, keepdim=True) + 1e-6)
                    extra = xb[:, 14:, :]
                    if args.extra_norm == "per":
                        extra = extra - extra.mean(dim=2, keepdim=True)
                        extra = extra / (extra.std(dim=2, keepdim=True) + 1e-6)
                    elif args.extra_norm == "global" and global_mean is not None:
                        extra_mean = global_mean[:, 14:, :]
                        extra_std = global_std[:, 14:, :]
                        extra = (extra - extra_mean) / (extra_std + 1e-6)
                    xb = torch.cat([eeg_part, extra], dim=1)
                else:
                    xb = xb - xb.mean(dim=2, keepdim=True)
                    xb = xb / (xb.std(dim=2, keepdim=True) + 1e-6)
            else:  # global
                xb = (xb - global_mean) / (global_std + 1e-6)
            yb = yb.to(device).long()

            opt.zero_grad()

            # Augmentations (toggleable)
            if not args.no_timeshift and args.time_shift > 0:
                xb = time_shift(xb, max_shift=args.time_shift)
            # Channel dropout (drop entire channels)
            if args.channel_dropout > 0 and epoch >= args.channel_dropout_start:
                if args.channel_dropout >= 1.0:
                    raise SystemExit("channel_dropout must be < 1.0")
                mask = (torch.rand(xb.size(0), xb.size(1), 1, device=xb.device) > args.channel_dropout).float()
                xb = xb * mask
            # Add Gaussian noise
            if args.gaussian_noise > 0 and epoch >= args.gaussian_noise_start:
                noise = torch.randn_like(xb) * xb.std(dim=2, keepdim=True) * args.gaussian_noise
                xb = xb + noise
            # Delayed (curriculum) mixup
            # Default (no soft targets)
            yb_mixed_target = yb
            ysoft = None
            used_mix = False
            if epoch >= args.mixup_start and (not args.no_mixup) and args.mixup_alpha > 0:
                xb, yb_mixed_target, ysoft = mixup(xb, yb, alpha=args.mixup_alpha, n_classes=n_classes)
                used_mix = True
            # Temporal CutMix (only if mixup not used this batch and enabled)
            if not used_mix and args.time_cutmix_alpha > 0 and epoch >= args.time_cutmix_start:
                lam = np.random.beta(args.time_cutmix_alpha, args.time_cutmix_alpha)
                perm = torch.randperm(xb.size(0), device=xb.device)
                T = xb.size(2)
                cut_point = int(T * lam)
                if cut_point>0 and cut_point<T:
                    xb_cut = torch.cat([xb[:, :, :cut_point], xb[perm, :, cut_point:]], dim=2)
                    # compute effective lambda = proportion from first sample
                    lam_eff = cut_point / T
                    lam = lam_eff
                    y1 = yb
                    y2 = yb[perm]
                    y1_onehot = F.one_hot(y1, n_classes).float()
                    y2_onehot = F.one_hot(y2, n_classes).float()
                    targets = lam * y1_onehot + (1 - lam) * y2_onehot
                    logits = model(xb_cut)
                    loss = kl(F.log_softmax(logits, dim=1), targets)
                    loss.backward()
                    if args.grad_clip and args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    opt.step()
                    if args.sched == "onecycle":
                        sched.step()
                    loss_sum += float(loss.item()) * yb.size(0)
                    count += int(yb.numel())
                    with torch.no_grad():
                        preds = logits.argmax(1)
                        correct_train += int((preds == yb).sum().item())
                        total_train += int(yb.numel())
                    continue
            # If CutMix not applied (either disabled or degenerate window) we fall through with original targets

            logits = model(xb)
            # Adaptive label smoothing decay
            if args.ls_decay_start>0 and epoch>=args.ls_decay_start and isinstance(ce, nn.CrossEntropyLoss):
                # Rebuild CE with decayed smoothing (linear to 0 by end)
                decay_frac = (epoch - args.ls_decay_start) / max(1, args.epochs - args.ls_decay_start)
                new_ls = max(0.0, args.label_smoothing * (1 - decay_frac))
                if abs(new_ls - ce.label_smoothing) > 1e-4:
                    ce = nn.CrossEntropyLoss(weight=ce.weight, label_smoothing=new_ls)
            if ysoft is None:
                loss = ce(logits, yb_mixed_target)
            else:
                lam, y1, y2 = ysoft
                y1_onehot = F.one_hot(y1, n_classes).float()
                y2_onehot = F.one_hot(y2, n_classes).float()
                targets = lam * y1_onehot + (1 - lam) * y2_onehot
                loss = kl(F.log_softmax(logits, dim=1), targets)

            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            # Step OneCycle per batch if used
            if args.sched == "onecycle":
                sched.step()

            loss_sum += float(loss.item()) * yb.size(0)
            count += int(yb.numel())
            with torch.no_grad():
                preds = logits.argmax(1)
                correct_train += int((preds == yb).sum().item())
                total_train += int(yb.numel())

        tr_loss = loss_sum / max(count, 1)

        va_acc1, va_acc3, va_loss = evaluate(model, va_dl, device, norm=args.norm, extra_norm=args.extra_norm, global_mean=global_mean, global_std=global_std)
        # SWA accumulation
        if args.swa_start>0 and epoch>=args.swa_start:
            if swa_model is None:
                swa_model = {k: v.detach().clone() for k,v in model.state_dict().items()}
                swa_n = 1
            else:
                for k,v in model.state_dict().items():
                    swa_model[k].mul_(swa_n/(swa_n+1)).add_(v.detach()/(swa_n+1))
                swa_n += 1
        prev_best = best_va
        if va_acc1 > best_va:
            best_va = va_acc1
            torch.save(model.state_dict(), args.save)

        train_acc = correct_train / max(total_train, 1)
        if epoch % 5 == 0 or epoch == 1:
            lr_now = opt.param_groups[0]["lr"]
            print(f"epoch {epoch}  lr {lr_now:.2e}  train_loss {tr_loss:.4f}  train@1 {train_acc:.4f}  "
                  f"val_loss {va_loss:.4f}  val@1 {va_acc1:.4f}  val@3 {va_acc3:.4f}  best {best_va:.4f}")

        # Per-epoch scheduler step (not for OneCycle which steps per batch)
        if args.sched != "onecycle":
            sched.step()

        # Early stopping (corrected so improvement epoch does NOT increment counter)
        if args.early_stop > 0:
            if args.early_metric == "acc":
                # Compare against previous best before update
                improve = (va_acc1 - prev_best) > args.min_delta
            else:  # loss metric
                if epoch == 1:
                    best_loss = va_loss
                best_loss = min(locals().get('best_loss', va_loss), va_loss)
                improve = (locals().get('best_loss_prev', va_loss) - va_loss) > args.min_delta
                best_loss_prev = best_loss
            if improve:
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.early_stop:
                    print(f"Early stopping at epoch {epoch} (patience {args.early_stop})")
                    break

    # If SWA was used, optionally evaluate a SWA blended model (does not overwrite best checkpoint)
    if swa_model is not None:
        model.load_state_dict(swa_model, strict=False)
        swa_acc1, swa_acc3, swa_loss = evaluate(model, te_dl, device, norm=args.norm, extra_norm=args.extra_norm, global_mean=global_mean, global_std=global_std)
        print(f"SWA test_loss {swa_loss:.4f}  SWA test@1 {swa_acc1:.4f}  SWA test@3 {swa_acc3:.4f}")
    # Load best validation checkpoint for final test metric
    model.load_state_dict(torch.load(args.save, map_location=device))
    te_acc1, te_acc3, te_loss = evaluate(model, te_dl, device, norm=args.norm, extra_norm=args.extra_norm, global_mean=global_mean, global_std=global_std)
    print(f"test_loss {te_loss:.4f}  test@1 {te_acc1:.4f}  test@3 {te_acc3:.4f}")


if __name__ == "__main__":
    main()
