import torch
import torch.fft
import torch.nn as nn


class MDCTTransform(nn.Module):
    def __init__(self, n_fft=512, hop_length=256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        # 🔥 Hann window (fix spectral leakage)
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, x):
        """
        x: [B, 1, T]
        """
        x = x.squeeze(1)  # [B, T]

        X = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(x.device),
            return_complex=True,
        )

        # magnitude
        X = torch.abs(X)  # [B, F, T]

        # flatten
        B, F, T = X.shape
        X = X.reshape(B, 1, F * T)

        return X
