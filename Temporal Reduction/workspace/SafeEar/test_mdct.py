import torch
from safeear.models.decouple import SpeechTokenizer

model = SpeechTokenizer(
    n_filters=32,
    dimension=128,
    strides=[8, 5, 4, 2],
    lstm_layers=2,
    bidirectional=False,
    dilation_base=2,
    residual_kernel_size=3,
    n_residual_layers=1,
    activation="ELU",
    sample_rate=16000,
    n_q=8,
    semantic_dimension=128,
    codebook_size=1024,
)

x = torch.randn(2, 1, 16000)

o, loss, feat, _ = model(x)

print("Output shape:", o.shape)
