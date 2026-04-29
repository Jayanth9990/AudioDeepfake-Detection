import torch
import torch.nn as nn


class FSQCodebook(nn.Module):
    def __init__(self, levels):
        super().__init__()
        self.levels = levels
        self.dim = len(levels)

    def forward(self, x):
        B, D, T = x.shape
        x = torch.sigmoid(x)

        indices = []
        quantized = []

        for i, L in enumerate(self.levels):
            xi = x[:, i, :]
            idx = torch.floor(xi * L).long()
            idx = torch.clamp(idx, 0, L-1)
            qi = idx.float() / (L - 1)

            indices.append(idx)
            quantized.append(qi)

        indices = torch.stack(indices, dim=1)
        quantized = torch.stack(quantized, dim=1)

        return quantized, indices


class MultiCodebookFSQ(nn.Module):
    def __init__(self, num_codebooks=8, levels=[8,7,6,6]):
        super().__init__()

        self.codebooks = nn.ModuleList([
            FSQCodebook(levels) for _ in range(num_codebooks)
        ])

    def forward(self, x):
        all_codes = []
        all_quantized = []

        for cb in self.codebooks:
            q, idx = cb(x)
            all_quantized.append(q)
            all_codes.append(idx)

        # stack across codebooks
        codes = torch.stack(all_codes, dim=1)       # [B, N, D, T]
        quantized = torch.stack(all_quantized, dim=1)

        return quantized, codes


if __name__ == "__main__":
    x = torch.randn(1, 4, 9)

    fsq = MultiCodebookFSQ(num_codebooks=8)

    q, codes = fsq(x)

    print("Quantized:", q.shape)
    print("Codes:", codes.shape)
