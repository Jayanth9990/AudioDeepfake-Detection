import torch
from snac import SNAC

# Load pretrained model (speech)
device = "cpu"

model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()
audio = torch.randn(1, 1, 24000).to(device)

with torch.inference_mode():
    codes = model.encode(audio)

print("Token shapes:")
for i, c in enumerate(codes):
    print(f"Level {i}: {c.shape}")
