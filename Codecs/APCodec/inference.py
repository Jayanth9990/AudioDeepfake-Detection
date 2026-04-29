from __future__ import absolute_import, division, print_function, unicode_literals

import os
import json
import torch
from utils import AttrDict
from dataset import amp_pha_specturm
from models import Encoder
import librosa
import numpy as np

h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def inference(h):
    encoder = Encoder(h).to(device)

    state_dict_encoder = load_checkpoint(h.checkpoint_file_load_Encoder, device)
    encoder.load_state_dict(state_dict_encoder['encoder'])

    filelist = sorted(os.listdir(h.test_input_wavs_dir))
    os.makedirs(h.test_wav_output_dir, exist_ok=True)

    encoder.eval()

    with torch.no_grad():
        for filename in filelist:

            print("\nProcessing:", filename)

            raw_wav, _ = librosa.load(
                os.path.join(h.test_input_wavs_dir, filename),
                sr=h.sampling_rate,
                mono=True
            )

            raw_wav = torch.FloatTensor(raw_wav).to(device)

            logamp, pha, _, _ = amp_pha_specturm(
                raw_wav.unsqueeze(0),
                h.n_fft,
                h.hop_size,
                h.win_size
            )

            # Get tokens
            codes = encoder.encode(logamp, pha)

            # SafeEar-style split
            acoustic_tokens = codes[:, 1:, :]

            # Debug checks
            print("Codes shape:", codes.shape)
            print("Acoustic shape:", acoustic_tokens.shape)
            print("Codes dtype:", codes.dtype)
            print("Codes min/max:", codes.min().item(), codes.max().item())

            unique_vals = torch.unique(codes[0, 0, :])
            print("Unique values in first codebook:", unique_vals.shape[0])

            # Save tokens
            acoustic_tokens = acoustic_tokens.squeeze(0).cpu().numpy()

            save_path = os.path.join(
                h.test_wav_output_dir,
                filename.split('.')[0] + '.npy'
            )

            np.save(save_path, acoustic_tokens)

            print("Saved:", save_path)


def main():
    print("Initializing Inference Process..")

    config_file = "config.json"

    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)

    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    inference(h)


if __name__ == "__main__":
    main()
