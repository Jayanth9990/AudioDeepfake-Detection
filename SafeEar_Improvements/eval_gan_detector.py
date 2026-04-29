import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


FEATURE_DIR = "./features_hubert/eval"
PROTOCOL_FILE = "./SafeEar/datas/ASVSpoof2019/ASVspoof2019.LA.cm.eval.trl.txt"
MODEL_PATH = "./models/gyan_discriminator.pt"

BATCH_SIZE = 32
PRINT_EVERY = 4000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)


# DISCRIMINATOR
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


def load_protocol(protocol_file):
    labels = {}
    with open(protocol_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            utt_id = parts[1]
            label = 0 if parts[-1] == "bonafide" else 1
            labels[utt_id] = label
    return labels


class HubertEvalDataset(Dataset):
    def __init__(self, feature_dir, protocol_labels):
        self.files = []
        self.labels = []

        for fname in sorted(os.listdir(feature_dir)):
            if not fname.endswith(".npy"):
                continue

            utt_id = fname.replace(".npy", "")
            if utt_id not in protocol_labels:
                continue

            self.files.append(os.path.join(feature_dir, fname))
            self.labels.append(protocol_labels[utt_id])

        print(f"Total eval samples used: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        feat = np.load(self.files[idx])
        feat = torch.tensor(feat, dtype=torch.float32).mean(dim=0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feat, label

# EVALUATION

def evaluate():
    protocol_labels = load_protocol(PROTOCOL_FILE)
    print(f"Loaded {len(protocol_labels)} protocol labels")

    dataset = HubertEvalDataset(FEATURE_DIR, protocol_labels)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = Discriminator().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_labels = []
    all_scores = []

    processed = 0

    with torch.no_grad():
        for feats, labels in loader:
            feats = feats.to(DEVICE)
            labels = labels.to(DEVICE)

            _, cls_out = model(feats)
            probs = torch.softmax(cls_out, dim=1)[:, 1]  # FAKE prob

            all_scores.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            processed += len(labels)
            if processed % PRINT_EVERY == 0 or processed == len(dataset):
                print(f"Processed {processed}/{len(dataset)} samples")

    y_true = np.array(all_labels)
    y_scores = np.array(all_scores)

    print("\n===== GYAN GAN RESULTS =====")

    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred = (y_scores > t).astype(int)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        print(f"\n--- Threshold = {t} ---")
        print(f"Accuracy  : {acc:.4f}")
        print(f"Precision : {prec:.4f}")
        print(f"Recall    : {rec:.4f}")
        print(f"F1-score  : {f1:.4f}")
        print("Confusion Matrix (rows=true, cols=pred):")
        print("           Pred-Real  Pred-Fake")
        print(f"True-Real   {cm[0][0]:6d}     {cm[0][1]:6d}")
        print(f"True-Fake   {cm[1][0]:6d}     {cm[1][1]:6d}")


if __name__ == "__main__":
    evaluate()

