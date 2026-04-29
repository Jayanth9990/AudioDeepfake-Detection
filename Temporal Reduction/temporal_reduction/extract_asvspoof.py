import os
import torch
import torchaudio
from snac import SNAC
from tqdm import tqdm

device = "cpu"

# Load SNAC
model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()

# 🔥 ABSOLUTE PATHS (VERY IMPORTANT)
BASE_PATH = "/home/zeta/Workbenches/KA/Dataset/asvspoof/LA"

DATA_PATH = os.path.join(BASE_PATH, "ASVspoof2019_LA_train/flac")
PROTOCOL_PATH = os.path.join(BASE_PATH, "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt")


import librosa
import numpy as np

def load_audio(path):
    waveform, sr = librosa.load(path, sr=24000)  # resample directly

    waveform = torch.tensor(waveform).unsqueeze(0)  # (1, T)

    return waveform.unsqueeze(0)  # (1,1,T)


def extract_features(audio):
    with torch.inference_mode():
        codes = model.encode(audio)

    feat = torch.cat([
        codes[0].float(),
        codes[1].float()
    ], dim=1)

    return feat.squeeze(0)


def parse_protocol():
    data = []

    with open(PROTOCOL_PATH, "r") as f:
        for line in f:
            parts = line.strip().split()

            file_id = parts[1]
            label = parts[-1]

            label = 0 if label == "bonafide" else 1

            data.append((file_id, label))

    return data


def main():
    entries = parse_protocol()


    features = []
    labels = []

    for file_id, label in tqdm(entries):
        file_path = os.path.join(DATA_PATH, file_id + ".flac")

        if not os.path.exists(file_path):
            print("Missing:", file_path)
            continue

        try:
            audio = load_audio(file_path).to(device)
            feat = extract_features(audio)

            features.append(feat)
            labels.append(label)

        except Exception as e:
            print("Error:", file_id, e)

    torch.save((features, labels), "train_features.pt")

    print("Saved features:", len(features))


if __name__ == "__main__":
    main()
