import torch
import torch.nn as nn
import torch.nn.functional as F


class Codebook(nn.Module):
    def __init__(self, dim, num_embeddings):
        super().__init__()
        self.dim = dim
        self.num_embeddings = num_embeddings

        self.embedding = nn.Embedding(num_embeddings, dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, x):
        """
        x: [B, C, T]
        """
        B, C, T = x.shape

        # reshape to [B*T, C]
        x_flat = x.permute(0, 2, 1).contiguous().view(-1, C)

        # compute distances
        dist = (
            x_flat.pow(2).sum(1, keepdim=True)
            - 2 * x_flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1)
        )

        # nearest codebook entry
        indices = torch.argmin(dist, dim=1)

        # quantized vectors
        z_q = self.embedding(indices)

        # reshape back
        z_q = z_q.view(B, T, C).permute(0, 2, 1).contiguous()

        indices = indices.view(B, T)

        return z_q, indices


class RVQ(nn.Module):
    def __init__(self, dim, num_codebooks=8, codebook_size=1024):
        super().__init__()

        self.codebooks = nn.ModuleList([
            Codebook(dim, codebook_size)
            for _ in range(num_codebooks)
        ])

    def forward(self, x):
        """
        x: [B, C, T]
        """
        residual = x
        quantized_outputs = []
        all_indices = []

        for codebook in self.codebooks:
            z_q, indices = codebook(residual)

            quantized_outputs.append(z_q)
            all_indices.append(indices)

            # compute residual
            residual = residual - z_q

        # sum quantized outputs
        z_q_total = sum(quantized_outputs)

        # stack indices → [B, N, T]
        codes = torch.stack(all_indices, dim=1)

        return z_q_total, codes


#  TEST
if __name__ == "__main__":
    x = torch.randn(1, 64, 100)

    rvq = RVQ(dim=64, num_codebooks=8, codebook_size=512)

    z_q, codes = rvq(x)

    print("Input:", x.shape)
    print("Quantized:", z_q.shape)
    print("Codes:", codes.shape)
