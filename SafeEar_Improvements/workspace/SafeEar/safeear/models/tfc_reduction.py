import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalEntropy(nn.Module):
    """
    Estimate temporal importance using variance (proxy for entropy)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: (B, C, T, F)

        # collapse channel + freq → measure variation over time
        energy = x.pow(2).mean(dim=(1, 3))  # (B, T)

        # normalize
        energy = (energy - energy.min(dim=1, keepdim=True)[0]) / (
            energy.max(dim=1, keepdim=True)[0] - energy.min(dim=1, keepdim=True)[0] + 1e-8
        )

        return energy  # (B, T)


class TFCModule(nn.Module):
    """
    Full TFC implementation:
    - multi-resolution
    - entropy-based selection
    """

    def __init__(self, channels):
        super().__init__()

        # Multi-resolution encoders
        self.fine = nn.Identity()
        self.medium = nn.Conv2d(channels, channels, 3, stride=(2, 1), padding=1)
        self.coarse = nn.Conv2d(channels, channels, 3, stride=(4, 1), padding=1)

        # Entropy estimator
        self.entropy = TemporalEntropy()

    def forward(self, x):
        # x: (B, C, T, F)

        B, C, T, F = x.shape

        # -------------------------
        # 1. Multi-resolution features
        # -------------------------
        zf = self.fine(x)
        zm = self.medium(x)
        zc = self.coarse(x)

        # -------------------------
        # 2. Upsample to same length
        # -------------------------
        zm = zm.repeat_interleave(2, dim=2)
        zc = zc.repeat_interleave(4, dim=2)

        zm = zm[:, :, :T, :]
        zc = zc[:, :, :T, :]

        # -------------------------
        # 3. Compute temporal entropy
        # -------------------------
        importance = self.entropy(x)  # (B, T)

        # -------------------------
        # 4. Create masks (adaptive selection)
        # -------------------------
        high_mask = (importance > 0.6).float().unsqueeze(1).unsqueeze(-1)
        mid_mask  = ((importance <= 0.6) & (importance > 0.3)).float().unsqueeze(1).unsqueeze(-1)
        low_mask  = (importance <= 0.3).float().unsqueeze(1).unsqueeze(-1)

        # -------------------------
        # 5. Adaptive fusion
        # -------------------------
        z = (
            zf * high_mask +     # high info → fine
            zm * mid_mask  +     # medium → medium
            zc * low_mask        # low info → coarse
        )

        return z
