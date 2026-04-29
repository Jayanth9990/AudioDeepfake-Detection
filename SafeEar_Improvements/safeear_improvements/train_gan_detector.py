import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm # type: ignore


FEATURE_DIR = "./features_hubert/train"
PROTOCOL_FILE = "./SafeEar/datas/ASVSpoof2019/ASVspoof2019.LA.cm.train.trl.txt"
MODEL_DIR = "./models"

BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-4
LATENT_DIM = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

os.makedirs(MODEL_DIR, exist_ok=True)


def load_protocol_labels(protocol_file):
    labels = {}
    with open(protocol_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            utt_id = parts[1]
            label = 0 if parts[-1] == "bonafide" else 1
            labels[utt_id] = label
    return labels

# DATASET

class HubertDataset(Dataset):
    def __init__(self, feature_dir, protocol_labels):
        self.files = []
        self.labels = []

        for fname in os.listdir(feature_dir):
            if not fname.endswith(".npy"):
                continue

            utt_id = fname.replace(".npy", "")
            if utt_id not in protocol_labels:
                continue

            self.files.append(os.path.join(feature_dir, fname))
            self.labels.append(protocol_labels[utt_id])

        print(f"Training samples: {len(self.files)}")
        real = sum(1 for l in self.labels if l == 0)
        fake = sum(1 for l in self.labels if l == 1)
        print(f"Real (bonafide): {real} | Fake (spoof): {fake}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        feat = np.load(self.files[idx])
        feat = torch.tensor(feat, dtype=torch.float32).mean(dim=0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feat, label

# MODELS

class Generator(nn.Module):
    def __init__(self, z_dim, feat_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(),
            nn.Linear(512, feat_dim)
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, feat_dim=768):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2)
        )
        self.adv = nn.Linear(256, 1)
        self.cls = nn.Linear(256, 2)

    def forward(self, x):
        h = self.shared(x)
        return self.adv(h), self.cls(h)

# TRAINING

def train():
    protocol_labels = load_protocol_labels(PROTOCOL_FILE)
    dataset = HubertDataset(FEATURE_DIR, protocol_labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    G = Generator(LATENT_DIM).to(DEVICE)
    D = Discriminator().to(DEVICE)

    opt_G = optim.Adam(G.parameters(), lr=LR)
    opt_D = optim.Adam(D.parameters(), lr=LR)

    adv_loss = nn.BCEWithLogitsLoss()

    #  CLASS-WEIGHTED LOSS 
    real_count = sum(1 for l in dataset.labels if l == 0)
    fake_count = sum(1 for l in dataset.labels if l == 1)

    class_weights = torch.tensor(
        [fake_count / real_count, 1.0],
        dtype=torch.float32
    ).to(DEVICE)

    cls_loss = nn.CrossEntropyLoss(weight=class_weights)

    print("Class weights:", class_weights.tolist())

    for epoch in range(EPOCHS):
        g_loss_total, d_loss_total = 0, 0

        for real_feats, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            real_feats = real_feats.to(DEVICE)
            labels = labels.to(DEVICE)

            # Train Discriminator
            
            z = torch.randn(real_feats.size(0), LATENT_DIM).to(DEVICE)
            fake_feats = G(z)

            real_adv, real_cls = D(real_feats)
            fake_adv, _ = D(fake_feats.detach())

            d_adv_loss = (
                adv_loss(real_adv, torch.ones_like(real_adv)) +
                adv_loss(fake_adv, torch.zeros_like(fake_adv))
            )

            d_cls_loss = cls_loss(real_cls, labels)

            #  BALANCED LOSS
            d_loss = 0.3 * d_adv_loss + 0.7 * d_cls_loss

            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            # Train Generator
            
            fake_adv, _ = D(fake_feats)
            g_loss = adv_loss(fake_adv, torch.ones_like(fake_adv))

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            d_loss_total += d_loss.item()
            g_loss_total += g_loss.item()

        print(
            f"Epoch {epoch+1} | "
            f"D loss: {d_loss_total/len(loader):.4f} | "
            f"G loss: {g_loss_total/len(loader):.4f}"
        )

    torch.save(D.state_dict(), os.path.join(MODEL_DIR, "gyan_discriminator.pt"))
    print("\nGYAN GAN training completed and model saved.")

if __name__ == "__main__":
    train()
