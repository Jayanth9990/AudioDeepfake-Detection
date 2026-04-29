import torch
import torch.nn as nn

from mdct import mdct, imdct
from rvq import RVQ


# Encoder

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(40, 128, 3, padding=1),
            nn.ReLU(),

            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU(),

            nn.Conv1d(128, 64, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


# Decoder

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),

            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU(),

            nn.Conv1d(128, 40, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)



# FULL MODEL

class MDCTCodec(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.quantizer = RVQ(dim=64, num_codebooks=8, codebook_size=512)
        self.decoder = Decoder()

    def forward(self, x):
        """
        x: [B, T]
        """

        # MDCT
        X = mdct(x)                 # [B, N, 40]

        # reshape for conv
        X = X.permute(0, 2, 1)      # [B, 40, N]

        # encode
        z = self.encoder(X)         # [B, 64, N]

        # quantize
        z_q, codes = self.quantizer(z)

        # decode
        X_hat = self.decoder(z_q)   # [B, 40, N]

        # reshape back
        X_hat = X_hat.permute(0, 2, 1)  # [B, N, 40]

        # IMDCT
        x_hat = imdct(X_hat)

        return x_hat, codes


#  TEST 
if __name__ == "__main__":
    x = torch.randn(1, 16000)

    model = MDCTCodec()

    x_hat, codes = model(x)

    print("Input:", x.shape)
    print("Output:", x_hat.shape)
    print("Codes:", codes.shape)




import numpy as np

# bitrate calculation
fps = 16000 / 40   # sampling_rate / hop_size
Q = codes.shape[1]
M = 512

bits_per_token = np.log2(M)

bitrate = fps * Q * bits_per_token

print("FPS:", fps)
print("Tokens/sec:", fps * Q)
print("Bitrate (kbps):", bitrate / 1000)
