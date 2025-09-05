import os
import pickle as pkl
import numpy as np
import torch
from eegcnn import EEGNetClassifier

# Load a tiny subset (10 samples, 2 classes)
with open("data/AREEG_Words/preprocessed_pkl/train.pkl", "rb") as f:
    obj = pkl.load(f)
X = obj["X"]
y = obj["y"]
idx_to_label = obj["idx_to_label"]

# Select 2 classes
classes = np.unique(y)[:2]
mask = np.isin(y, classes)
X_small = X[mask][:10]
y_small = y[mask][:10]

# Convert to torch tensors
X_small = torch.from_numpy(X_small).float()
y_small = torch.from_numpy(y_small).long()

# Model
model = EEGNetClassifier(n_classes=16, Chans=14)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
ce = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(1, 101):
    model.train()
    opt.zero_grad()
    xb = X_small
    xb = xb - xb.mean(dim=2, keepdim=True)
    xb = xb / (xb.std(dim=2, keepdim=True) + 1e-6)
    logits = model(xb)
    loss = ce(logits, y_small)
    loss.backward()
    opt.step()
    pred = logits.argmax(1)
    acc = (pred == y_small).float().mean().item()
    if epoch % 10 == 0 or acc == 1.0:
        print(f"Epoch {epoch}: loss={loss.item():.4f}, acc={acc:.4f}")
    if acc == 1.0:
        print("Model can overfit tiny subset.")
        break
else:
    print("Model could NOT overfit tiny subset.")
