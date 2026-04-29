import numpy as np

# from your model output
fps = 9
num_codebooks = 8

# FSQ levels
levels = [8, 7, 6, 6]

# bits per token
M = np.prod(levels)
bits_per_token = np.log2(M)

# tokens/sec
tokens_per_sec = fps * num_codebooks

# bitrate
bitrate = tokens_per_sec * bits_per_token

print("Bits per token:", bits_per_token)
print("Tokens/sec:", tokens_per_sec)
print("Bitrate (bps):", bitrate)
print("Bitrate (kbps):", bitrate / 1000)
