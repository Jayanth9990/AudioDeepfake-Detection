import torch
import numpy as np


def mdct(x, frame_length=80, hop_length=40):
    """
    x: [B, T]
    returns: [B, N, K]
    """
    B, T = x.shape

# pad so frames fit exactly
    pad = (hop_length - (T - frame_length) % hop_length) % hop_length
    x = torch.nn.functional.pad(x, (0, pad))

    T = x.shape[1]
    K = hop_length

    frames = []

    for i in range(0, T - frame_length+1, hop_length):
        frame = x[:, i:i+frame_length]  # [B, 80]
        frames.append(frame)

    frames = torch.stack(frames, dim=1)  # [B, N, 80]

    n = torch.arange(frame_length).float()
    k = torch.arange(K).float()

    cos_basis = torch.cos(
        np.pi / K * (n[:, None] + 0.5 + K/2) * (k + 0.5)
    )  # [80, 40]

    cos_basis = cos_basis.to(x.device)

    X = torch.matmul(frames, cos_basis)  # [B, N, K]

    return X


def imdct(X, frame_length=80, hop_length=40):
    """
    X: [B, N, K]
    returns: [B, T]
    """
    B, N, K = X.shape

    n = torch.arange(frame_length).float()
    k = torch.arange(K).float()

    cos_basis = torch.cos(
        np.pi / K * (n[:, None] + 0.5 + K/2) * (k + 0.5)
    )

    cos_basis = cos_basis.to(X.device)

    frames = torch.matmul(X, cos_basis.T) / K  # [B, N, 80]

    T = (N - 1) * hop_length + frame_length
    output = torch.zeros(B, T).to(X.device)

    for i in range(N):
        start = i * hop_length
        output[:, start:start+frame_length] += frames[:, i]

    return output


if __name__ == "__main__":
    x = torch.randn(1, 16000)

    X = mdct(x)
    x_hat = imdct(X)

    print("Input:", x.shape)
    print("MDCT:", X.shape)
    print("Reconstructed:", x_hat.shape)
