import random
from pathlib import Path

import torch
import torchaudio
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


def get_path_iterator(tsv):
    with open(tsv, "r") as f:
        root = f.readline().rstrip()
        lines = [line.rstrip() for line in f]
    return root, lines


class ASVSppof2019(Dataset):
    def __init__(self, tsv_path, protocol_path, feat_dir, max_len=64600, is_train=True):
        super().__init__()

        root, self.lines = get_path_iterator(tsv_path)
        self.root = Path(root)
        self.max_len = max_len
        self.is_train = is_train

        # Get sample rate
        _, self.sr = torchaudio.load(str(self.root / self.lines[0]))

        # Load protocol (labels)
        with open(protocol_path) as file:
            meta_infos = file.readlines()

        self.mapping = {
            meta_info.strip().split(" ")[1]: meta_info.strip().split(" ")[-1]
            for meta_info in meta_infos
        }

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        relative_path = Path(self.lines[index])
        audio_path = self.root / relative_path

        # Load audio
        audio = torchaudio.load(str(audio_path))[0]

        # Label
        key = relative_path.stem
        # waveform_info = self.mapping[key]

        if key not in self.mapping:
            # skip invalid sample by choosing another random one
            return self.__getitem__((index + 1) % len(self.lines))

        waveform_info = self.mapping[key]
        target = 1 if waveform_info == "spoof" else 0

        # -------- AUDIO PROCESSING --------
        if audio.shape[1] > self.max_len:
            if self.is_train:
                st = random.randint(0, audio.shape[1] - self.max_len - 1)
            else:
                st = 0
            audio = audio[:, st : st + self.max_len]

        if audio.shape[1] < self.max_len:
            pad_len = self.max_len - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, pad_len))

        # -------- RETURN --------
        if self.is_train:
            return audio, target
        else:
            return audio, target, str(audio_path)


def pad_sequence(batch):
    batch = [item.permute(1, 0) for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
    return batch.permute(0, 2, 1)


def collate_fn(batch):
    wavs = []
    targets = []

    for item in batch:
        wavs.append(item[0])
        targets.append(item[1])

    wavs = pad_sequence(wavs)
    targets = torch.tensor(targets).long()

    return wavs, targets


class DataClass:
    def __init__(self, train_path, val_path, test_path, max_len=64600):
        self.train = ASVSppof2019(
            train_path[0], train_path[1], train_path[2], max_len, True
        )
        self.val = ASVSppof2019(val_path[0], val_path[1], val_path[2], max_len, True)
        self.test = ASVSppof2019(
            test_path[0], test_path[1], test_path[2], max_len, False
        )

    def __call__(self, mode):
        if mode == "train":
            return self.train
        elif mode == "val":
            return self.val
        elif mode == "test":
            return self.test
        else:
            raise ValueError(f"Unknown mode: {mode}")


class DataModule(LightningDataModule):
    def __init__(self, DataClass_dict, batch_size, num_workers, pin_memory):
        super().__init__()
        self.save_hyperparameters(logger=False)
        DataClass_dict.pop("_target_")
        self.dataset_select = DataClass(**DataClass_dict)

    def setup(self, stage=None):
        self.data_train = self.dataset_select("train")
        self.data_val = self.dataset_select("val")
        self.data_test = self.dataset_select("test")

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
