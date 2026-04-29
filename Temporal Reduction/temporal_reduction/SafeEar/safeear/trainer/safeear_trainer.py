import math
import warnings

import numpy as np
import pytorch_lightning as pl
import torch

from ..losses.loss import compute_eer

warnings.filterwarnings("ignore")


def get_input(x):
    x = x.to(memory_format=torch.contiguous_format)
    return x.float()


class SafeEarTrainer(pl.LightningModule):
    def __init__(self, decouple_model, detect_model, lr_raw_former, save_score_path):
        super().__init__()

        self.decouple_model = decouple_model
        self.detect_model = detect_model
        self.lr_raw_former = lr_raw_former
        self.save_score_path = save_score_path

        self.detect_loss = torch.nn.BCELoss()
        self.automatic_optimization = False

        self.val_index_loader = []
        self.val_score_loader = []
        self.eval_index_loader = []
        self.eval_score_loader = []
        self.eval_filename_loader = []
        self.default_monitor = "val_eer"

    def forward(self, batch, is_train=True):

        # ✅ FIXED INPUT FORMAT
        if is_train:
            x, target = batch
        else:
            if len(batch) == 3:
                x, target, audio_path = batch
            else:
                x, target = batch
                audio_path = None

        x_wav = get_input(x)

        # Tokenizer
        with torch.no_grad():
            self.decouple_model.eval()
            G_x, commit_loss, last_layer, acoustic_tokens = self.decouple_model(
                x_wav, layers=[0, 1, 2, 3, 4, 5, 6, 7]
            )

        # Detector
        raw_logits, raw_feature = self.detect_model(acoustic_tokens)

        if is_train:
            onehot_target = torch.eye(2).to(self.device)[target, :]
            raw_logits = torch.softmax(raw_logits, dim=-1)
            loss = self.detect_loss(raw_logits, onehot_target)
            return loss, raw_logits, target
        else:
            raw_logits = torch.softmax(raw_logits, dim=-1)[:, 0]
            return audio_path, 0, raw_logits, target

    def training_step(self, batch, batch_idx):
        raw_opt = self.optimizers()

        loss, raw_logits, target = self(batch, is_train=True)

        raw_opt.zero_grad()
        self.manual_backward(loss)
        raw_opt.step()

        self.log_dict(
            {"train_loss": loss},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )

    def validation_step(self, batch, batch_idx):
        _, loss, raw_logits, target = self(batch, is_train=False)

        self.val_index_loader.append(target)
        self.val_score_loader.append(raw_logits)

        self.log_dict(
            {"val_loss": loss},
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )

    def on_validation_epoch_end(self):
        all_index = torch.cat(self.val_index_loader).cpu().numpy()
        all_score = torch.cat(self.val_score_loader).cpu().numpy()

        val_eer = compute_eer(all_score[all_index == 0], all_score[all_index == 1])[0]
        other_val_eer = compute_eer(
            -all_score[all_index == 0], -all_score[all_index == 1]
        )[0]
        val_eer = min(val_eer, other_val_eer)

        self.log("val_eer", val_eer, prog_bar=True)

        self.val_index_loader.clear()
        self.val_score_loader.clear()

    def test_step(self, batch, batch_idx):
        audio_path, _, raw_logits, target = self(batch, is_train=False)

        self.eval_index_loader.append(target)
        self.eval_score_loader.append(raw_logits)
        self.eval_filename_loader.append(audio_path)

    def on_test_epoch_end(self):
        all_index = torch.cat(self.eval_index_loader).cpu().numpy()
        all_score = torch.cat(self.eval_score_loader).cpu().numpy()

        eval_eer = compute_eer(all_score[all_index == 0], all_score[all_index == 1])[0]
        other_eval_eer = compute_eer(
            -all_score[all_index == 0], -all_score[all_index == 1]
        )[0]
        eval_eer = min(eval_eer, other_eval_eer)

        self.log("test_eer", eval_eer, prog_bar=True)

        self.eval_index_loader.clear()
        self.eval_score_loader.clear()
        self.eval_filename_loader.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.detect_model.parameters(), lr=self.lr_raw_former)


def adjust_learning_rate(optimizer, epoch, lr, warmup, epochs=100):
    if epoch < warmup:
        lr = lr / (warmup - epoch)
    else:
        lr *= 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup) / (epochs - warmup)))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
