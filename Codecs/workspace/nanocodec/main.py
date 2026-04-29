import torch
from encoder import Encoder
from fsq_multi import MultiCodebookFSQ
import numpy as np

# model
encoder = Encoder()
fsq = MultiCodebookFSQ(num_codebooks=8)

# input audio
x = torch.randn(1, 1, 16000)

# forward
z = encoder(x)

# reduce channels (32 → 4 for FSQ)
z = z[:, :4, :]

q, codes = fsq(z)

print("Codes shape:", codes.shape)

# ---- bitrate ----
fps = codes.shape[-1]   # frames/sec approx
num_codebooks = codes.shape[1]

levels = [8,7,6,6]

M = np.prod(levels)
bits_per_token = np.log2(M)

tokens_per_sec = fps * num_codebooks
bitrate = tokens_per_sec * bits_per_token

print("FPS:", fps)
print("Tokens/sec:", tokens_per_sec)
print("Bitrate (kbps):", bitrate / 1000)
