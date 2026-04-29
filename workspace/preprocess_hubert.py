import os
import torch
import numpy as np
from tqdm import tqdm # type: ignore
from transformers import HubertModel, Wav2Vec2FeatureExtractor # type: ignore
import soundfile as sf
import torchaudio

DATASET_ROOT = "/home/sameera/unofficail_audio_deepfake/Dataset/LA"
OUT_ROOT = "/scratch/sameera/audio_deepfake/features_hubert"

SPLITS = {
    "train": "ASVspoof2019_LA_train/flac",
    "dev":   "ASVspoof2019_LA_dev/flac",
    "eval":  "ASVspoof2019_LA_eval/flac",
}

SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


print(f"Using device: {DEVICE}")

os.makedirs(OUT_ROOT, exist_ok=True)

# Load HuBERT
extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "facebook/hubert-base-ls960"
)
model = HubertModel.from_pretrained(
    "facebook/hubert-base-ls960"
).to(DEVICE)
model.eval()


def load_audio(path):
    wav, sr = sf.read(path)

    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    if sr != SAMPLE_RATE:
        wav = torch.tensor(wav).unsqueeze(0)
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        wav = wav.squeeze(0).numpy()

    return torch.tensor(wav, dtype=torch.float32)


@torch.no_grad()
def extract_features(wav):
    inputs = extractor(
        wav,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    outputs = model(**inputs)
    return outputs.last_hidden_state.squeeze(0).cpu().numpy()

def process_split(split, rel_path):
    in_dir = os.path.join(DATASET_ROOT, rel_path)
    out_dir = os.path.join(OUT_ROOT, split)
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(in_dir) if f.endswith(".flac"))

    print(f"\nProcessing {split} ({len(files)} files)")

    for fname in tqdm(files):
        in_path = os.path.join(in_dir, fname)
        out_path = os.path.join(out_dir, fname.replace(".flac", ".npy"))

        if os.path.exists(out_path):
            continue

        try:
            wav = load_audio(in_path)
            feats = extract_features(wav)
            np.save(out_path, feats)
        except Exception as e:
            print(f"Error with {fname}: {e}")

if __name__ == "__main__":
    for split, path in SPLITS.items():
        process_split(split, path)

    print("\n HuBERT preprocessing completed")
