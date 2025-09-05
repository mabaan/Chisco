#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠🤖 EEG Neural Network Models - The Brain Decoders

This file contains the actual AI models that learn to decode brain signals.
Think of these as different types of "brain reading machines" - each with 
their own special way of understanding EEG data.

The Models:
1. EEGNet: The classic brain signal decoder - proven and reliable
2. EEGNetTransformer: The modern approach using attention mechanisms 
3. Ensemble Models: Multiple brain decoders working together for better accuracy

What makes these special:
- Designed specifically for EEG brain signals (not just any data)
- Can handle both time and spatial patterns in brain activity
- Uses advanced techniques like attention to focus on important parts
- Ensemble approach: like having multiple experts vote on the answer

Think of it like this:
- Your brain produces electrical signals when you think
- These models learn the unique "fingerprint" of each thought
- The better the model, the more accurately it can guess what you're thinking

The transformer version is like having a super-smart assistant that can 
pay attention to the most important parts of your brain signals!
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F

class EEGNetTransformerClassifier(nn.Module):
    """
    Enhanced EEGNet backbone + Transformer encoder with advanced features.
    Supports deeper/wider transformer, larger ensembling, and better regularization.
    """
    def __init__(self, n_classes, Chans=14, kernLength1=15, kernLength2=7, F1=16, D=2, F2=128, P1=64, P2=16, dropoutRate=0.3,
                 num_layers=6, dim_feedforward=512, ensemble=5):
        super().__init__()
        self.ensemble = ensemble
        
        # Diverse ensemble with different architectures
        self.backbones = nn.ModuleList([
            EEGcnn(Chans=Chans, kernLength1=kernLength1, kernLength2=kernLength2, 
                   F1=F1 + i*4, D=D, F2=F2, P1=P1, P2=P2, dropoutRate=dropoutRate + i*0.05)
            for i in range(ensemble)
        ])
        
        self.transformers = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=F2, nhead=8, dim_feedforward=dim_feedforward, 
                    dropout=dropoutRate, batch_first=True, activation='gelu'
                ), num_layers=num_layers
            ) for _ in range(ensemble)
        ])
        
        # Multi-layer classifier with residual connections
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(F2, F2 // 2),
                nn.LayerNorm(F2 // 2),
                nn.GELU(),
                nn.Dropout(dropoutRate),
                nn.Linear(F2 // 2, n_classes)
            ) for _ in range(ensemble)
        ])
        
        # Learnable ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(ensemble) / ensemble)

    def forward(self, x):
        # x: [B, C, T]
        outs = []
        for i, (backbone, transformer, classifier) in enumerate(zip(
            self.backbones, self.transformers, self.classifiers
        )):
            feat = backbone(x)  # [B, F2, T_out]
            feat = feat.permute(0, 2, 1)  # [B, T_out, F2] for transformer
            feat = transformer(feat)  # [B, T_out, F2]
            
            # Multi-head attention pooling
            attn_weights = torch.softmax(feat.mean(dim=-1), dim=-1)  # [B, T_out]
            feat = (feat * attn_weights.unsqueeze(-1)).sum(dim=1)  # [B, F2]
            
            out = classifier(feat)  # [B, n_classes]
            outs.append(out)
        
        # Weighted ensemble
        outs = torch.stack(outs)  # [ensemble, B, n_classes]
        weights = torch.softmax(self.ensemble_weights, dim=0)
        return torch.sum(outs * weights.view(-1, 1, 1), dim=0)


class EEGNetGRUAttnClassifier(nn.Module):
    """EEGNet backbone + bidirectional GRU + temporal attention pooling with optional LayerNorm."""
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
        use_layernorm: bool = False,
    use_proj: bool = False,
    proj_hidden: int = 0,
    ) -> None:
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
        self.attn = nn.Conv1d(2 * rnn_hidden, 1, kernel_size=1)
        self.dropout = nn.Dropout(dropoutRate)
        self.fc = nn.Linear(2 * rnn_hidden, n_classes)
        self.layernorm = nn.LayerNorm(2 * rnn_hidden) if use_layernorm else nn.Identity()
        # Optional residual projection MLP to refine temporal features before attention pooling
        in_dim = 2 * rnn_hidden
        if use_proj and proj_hidden > 0:
            self.proj = nn.Sequential(
                nn.Linear(in_dim, proj_hidden),
                nn.GELU(),
                nn.Dropout(dropoutRate),
                nn.Linear(proj_hidden, in_dim),
            )
        else:
            self.proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,C,T]
        feat = self.backbone(x)                 # [B, F2, T_out]
        feat = feat.permute(0, 2, 1)            # [B, T_out, F2]
        out, _ = self.gru(feat)                 # [B, T_out, 2*H]
        out = self.layernorm(out)
        if self.proj is not None:
            out = out + self.proj(out)          # residual refinement
        out = out.permute(0, 2, 1)              # [B, 2*H, T_out]
        w = self.attn(out)                      # [B,1,T_out]
        w = torch.softmax(w, dim=-1)
        pooled = (out * w).sum(dim=-1)          # [B, 2*H]
        pooled = self.dropout(pooled)
        return self.fc(pooled)



class EEGcnn(nn.Module):
    """
    Mini-EEGNet style backbone.
    Input:  x [B, C, T] where C can be 14 (EEG only) or 19 (EEG + band powers)
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
        self.eeg_channels = 14  # Always 14 EEG channels for spatial convolution

        Drop = nn.Dropout if dropoutType == "Dropout" else nn.Dropout2d

        # Block 1: temporal conv then depthwise spatial conv across EEG channels only
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength1), padding="same", bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, D * F1, (self.eeg_channels, 1), groups=F1, bias=False),
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
        
        # If we have extra features (band powers), process them separately
        if Chans > self.eeg_channels:
            extra_features = Chans - self.eeg_channels
            self.feature_processor = nn.Sequential(
                nn.Conv1d(extra_features, F2 // 4, kernel_size=1),
                nn.BatchNorm1d(F2 // 4),
                nn.ELU(),
                Drop(p=dropoutRate),
            )
            # Combine EEG features with band power features
            self.feature_combiner = nn.Conv1d(F2 + F2 // 4, F2, kernel_size=1)
        else:
            self.feature_processor = None
            self.feature_combiner = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] where C is 14 or 19
        B, C, T = x.shape
        
        # Extract EEG channels (first 14) for spatial convolution
        eeg_data = x[:, :self.eeg_channels, :]  # [B, 14, T]
        eeg_data = eeg_data.unsqueeze(1)  # [B, 1, 14, T]
        
        # Process EEG data through EEGNet blocks
        eeg_feat = self.block1(eeg_data)  # [B, D*F1, 1, T/P1]
        eeg_feat = self.block2(eeg_feat)  # [B, F2, 1, T/(P1*P2)]
        eeg_feat = eeg_feat.squeeze(2)    # [B, F2, T_out]
        
        # If we have extra features (band powers), process and combine them
        if C > self.eeg_channels and self.feature_processor is not None:
            extra_data = x[:, self.eeg_channels:, :]  # [B, 5, T]
            # Downsample extra features to match EEG feature temporal dimension
            T_out = eeg_feat.shape[2]
            if extra_data.shape[2] != T_out:
                extra_data = F.adaptive_avg_pool1d(extra_data, T_out)
            
            extra_feat = self.feature_processor(extra_data)  # [B, F2//4, T_out]
            
            # Combine EEG and extra features
            combined_feat = torch.cat([eeg_feat, extra_feat], dim=1)  # [B, F2 + F2//4, T_out]
            final_feat = self.feature_combiner(combined_feat)  # [B, F2, T_out]
            return final_feat
        else:
            return eeg_feat


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
    """EEGNet backbone + bidirectional GRU over time with optional LayerNorm."""
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
        use_layernorm: bool = False,
    ) -> None:
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
        self.layernorm = nn.LayerNorm(2 * rnn_hidden) if use_layernorm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,C,T]
        feat = self.backbone(x)                 # [B, F2, T_out]
        feat = feat.permute(0, 2, 1)            # [B, T_out, F2]
        out, _ = self.gru(feat)                 # [B, T_out, 2*H]
        out = self.layernorm(out)
        out = out.mean(dim=1)                   # [B, 2*H]
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
