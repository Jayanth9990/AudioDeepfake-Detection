import librosa
import torch
import l3ac
import time
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
MODELS = ['0k75bps', '1kbps', '1k5bps', '3kbps']
AUDIO_PATH = librosa.example("libri1")  # you can replace later

# -----------------------------
# LOAD AUDIO
# -----------------------------
audio, sr = librosa.load(AUDIO_PATH)
audio = audio[None, :]  # (1, T)

results = []

# -----------------------------
# LOOP OVER MODELS
# -----------------------------
for model_name in MODELS:
    print(f"\n===== MODEL: {model_name} =====")

    codec = l3ac.get_model(model_name)

    # Resample to model SR
    audio_resampled = librosa.resample(
        audio, orig_sr=sr, target_sr=codec.config.sample_rate
    )

    audio_tensor = torch.tensor(audio_resampled, dtype=torch.float32)

    # -----------------------------
    # ENCODE (TEMPORAL REDUCTION)
    # -----------------------------
    start = time.time()
    with torch.inference_mode():
        q_feature, indices = codec.encode_audio(audio_tensor)
    end = time.time()

    # -----------------------------
    # METRICS
    # -----------------------------
    input_len = audio_tensor.shape[-1]
    output_len = q_feature.shape[-1]

    reduction_ratio = input_len / output_len
    time_taken = end - start

    # Tokens per second
    duration_sec = input_len / codec.config.sample_rate
    tokens_per_sec = output_len / duration_sec

    # Feature stats
    mean_val = q_feature.mean().item()
    std_val = q_feature.std().item()

    # Energy proxy
    orig_energy = (audio_tensor ** 2).mean().item()
    feat_energy = (q_feature ** 2).mean().item()
    energy_ratio = feat_energy / orig_energy

    # -----------------------------
    # PRINT
    # -----------------------------
    print(f"Input length      : {input_len}")
    print(f"Output length     : {output_len}")
    print(f"Reduction ratio   : {reduction_ratio:.2f}x")
    print(f"Tokens/sec        : {tokens_per_sec:.2f}")
    print(f"Inference time    : {time_taken:.4f} sec")
    print(f"Feature mean/std  : {mean_val:.4f} / {std_val:.4f}")
    print(f"Energy ratio      : {energy_ratio:.4f}")

    results.append([
        model_name,
        input_len,
        output_len,
        reduction_ratio,
        tokens_per_sec,
        time_taken,
        energy_ratio
    ])

# -----------------------------
# FINAL TABLE
# -----------------------------
print("\n\n FINAL SUMMARY")
print("Model | Reduction | Tokens/sec | Time(s) | Energy")

for r in results:
    print(f"{r[0]:8} | {r[3]:8.2f}x | {r[4]:10.2f} | {r[5]:7.4f} | {r[6]:.4f}")
