# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGcnn(nn.Module):
    """
    Mini-EEGNet style backbone.
    Input:  x [B, C=14, T]
    Output: feat [B, F2, T_out]
    """
    def __init__(
        self,
        Chans: int = 14,
        kernLength1: int = 15,
        kernLength2: int = 7,
        F1: int = 16,
        D: int = 2,
        F2: int = 64,
        P1: int = 2,
        P2: int = 2,
        dropoutRate: float = 0.5,
        dropoutType: str = "Dropout",
    ):
        super().__init__()
        self.F1 = F1
        self.F2 = F2

        Drop = nn.Dropout if dropoutType == "Dropout" else nn.Dropout2d

        # Block 1: temporal conv then depthwise spatial conv across channels
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength1), padding="same", bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, D * F1, (Chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(D * F1),
            nn.ELU(),
            nn.AvgPool2d((1, P1)),
            Drop(p=dropoutRate),
        )

        # Block 2: depthwise temporal conv then pointwise conv
        self.block2 = nn.Sequential(
            nn.Conv2d(D * F1, D * F1, (1, kernLength2), groups=D * F1, padding="same", bias=False),
            nn.Conv2d(D * F1, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, P2)),
            Drop(p=dropoutRate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, Chans, T]
        x = x.unsqueeze(1)          # [B, 1, Chans, T]
        x = self.block1(x)          # [B, D*F1, 1, T/P1]
        x = self.block2(x)          # [B, F2, 1, T/(P1*P2)]
        x = x.squeeze(2)            # [B, F2, T_out]
        return x


class EEGNetClassifier(nn.Module):
    """
    EEGNet backbone + global average pooling head.
    """
    def __init__(
        self,
        n_classes: int = 16,
        Chans: int = 14,
        kernLength1: int = 15,
        kernLength2: int = 7,
        F1: int = 16,
        D: int = 2,
        F2: int = 64,
        P1: int = 2,
        P2: int = 2,
        dropoutRate: float = 0.5,
    ):
        super().__init__()
        self.backbone = EEGcnn(
            Chans=Chans, kernLength1=kernLength1, kernLength2=kernLength2,
            F1=F1, D=D, F2=F2, P1=P1, P2=P2, dropoutRate=dropoutRate
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(F2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)                 # [B, F2, T_out]
        feat = self.pool(feat).squeeze(-1)      # [B, F2]
        return self.fc(feat)


class EEGNetGRUClassifier(nn.Module):
    """
    EEGNet backbone + bidirectional GRU over time.
    Uses GRU to aggregate temporal features from EEGNet.
    """
    def __init__(
        self,
        n_classes: int = 16,
        Chans: int = 14,
        kernLength1: int = 15,
        kernLength2: int = 7,
        F1: int = 16,
        D: int = 2,
        F2: int = 64,
        P1: int = 2,
        P2: int = 2,
        dropoutRate: float = 0.5,
        rnn_hidden: int = 64,
        rnn_layers: int = 1,
    ):
        super().__init__()
        self.backbone = EEGcnn(
            Chans=Chans, kernLength1=kernLength1, kernLength2=kernLength2,
            F1=F1, D=D, F2=F2, P1=P1, P2=P2, dropoutRate=dropoutRate
        )
        self.gru = nn.GRU(
            input_size=F2,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropoutRate)
        self.fc = nn.Linear(2 * rnn_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)                 # [B, F2, T_out]
        feat = feat.permute(0, 2, 1)            # [B, T_out, F2]
        out, _ = self.gru(feat)                 # [B, T_out, 2*H]
        out = out.mean(dim=1)                   # mean over time is more stable
        out = self.dropout(out)
        return self.fc(out)


class EEGNetAttnClassifier(nn.Module):
    """
    EEGNet backbone + temporal attention pooling.
    """
    def __init__(
        self,
        n_classes: int = 16,
        Chans: int = 14,
        kernLength1: int = 15,
        kernLength2: int = 7,
        F1: int = 16,
        D: int = 2,
        F2: int = 64,
        P1: int = 2,
        P2: int = 2,
        dropoutRate: float = 0.5,
    ):
        super().__init__()
        self.backbone = EEGcnn(
            Chans=Chans, kernLength1=kernLength1, kernLength2=kernLength2,
            F1=F1, D=D, F2=F2, P1=P1, P2=P2, dropoutRate=dropoutRate
        )
        self.attn = nn.Conv1d(F2, 1, kernel_size=1)   # [B,1,T]
        self.dropout = nn.Dropout(dropoutRate)
        self.fc = nn.Linear(F2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)                 # [B, F2, T]
        w = self.attn(feat)                     # [B, 1, T]
        w = torch.softmax(w, dim=-1)            # attention over time
        pooled = (feat * w).sum(dim=-1)         # [B, F2]
        pooled = self.dropout(pooled)
        return self.fc(pooled)
