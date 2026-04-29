import os
import time

import librosa
import torch
from tqdm import tqdm

from snac import SNAC

device = "cuda:0"

model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()

BASE_PATH = "/home/zeta/Workbenches/KA/Dataset/asvspoof/LA"
DATA_PATH = os.path.join(BASE_PATH, "ASVspoof2019_LA_train/flac")

files = os.listdir(DATA_PATH)

results = []
total_time = 0
count = 0


def load_audio(path):
    audio, sr = librosa.load(path, sr=24000)
    return torch.tensor(audio)


for file in tqdm(files):
    try:
        path = os.path.join(DATA_PATH, file)
        audio = load_audio(path)

        audio_len = audio.shape[0]

        audio = audio.unsqueeze(0).unsqueeze(0).to(device)

        # Warm-up
        _ = model.encode(audio)

        # Timing
        start = time.time()
        codes = model.encode(audio)
        end = time.time()

        infer_time = (end - start) * 1000  # ms

        token_lengths = [c.shape[-1] for c in codes]

        reduction = audio_len / token_lengths[0]

        results.append((audio_len, token_lengths, reduction))
        total_time += infer_time
        count += 1

    except Exception as e:
        print("Error:", file, e)


# FINAL RESULTS
print("\n FINAL RESULTS ")

avg_time = total_time / count
avg_reduction = sum(r[2] for r in results) / count

print(f"Total samples processed: {count}")
print(f"Average inference time per sample (ms): {avg_time:.2f}")
print(f"Average reduction ratio: {avg_reduction:.2f}")

print("\nSample outputs:")
for r in results[:5]:
    print(r)
