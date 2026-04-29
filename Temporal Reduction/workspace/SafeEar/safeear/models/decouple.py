# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 15:47:55 2023
@author: zhangxin
"""

import torch
import torch.nn as nn
from einops import rearrange

from .modules.mdct import MDCTTransform  # 🔥 NEW
from .modules.quantization import ResidualVectorQuantizer
from .modules.seanet import SEANetDecoder, SEANetEncoder


class SpeechTokenizer(nn.Module):
    def __init__(
        self,
        n_filters,
        dimension,
        strides,
        lstm_layers,
        bidirectional,
        dilation_base,
        residual_kernel_size,
        n_residual_layers,
        activation,
        sample_rate,
        n_q,
        semantic_dimension,
        codebook_size,
    ):
        super().__init__()

        # 🔥 MDCT module (NEW)
        self.mdct = MDCTTransform()

        self.encoder = SEANetEncoder(
            n_filters=n_filters,
            dimension=dimension,
            ratios=strides,
            lstm=lstm_layers,
            bidirectional=bidirectional,
            dilation_base=dilation_base,
            residual_kernel_size=residual_kernel_size,
            n_residual_layers=n_residual_layers,
            activation=activation,
        )

        self.sample_rate = sample_rate
        self.n_q = n_q

        if dimension != semantic_dimension:
            self.transform = nn.Linear(dimension, semantic_dimension)
        else:
            self.transform = nn.Identity()

        self.quantizer = ResidualVectorQuantizer(
            dimension=dimension, n_q=n_q, bins=codebook_size
        )

        self.decoder = SEANetDecoder(
            n_filters=n_filters,
            dimension=dimension,
            ratios=strides,
            lstm=lstm_layers,
            bidirectional=False,
            dilation_base=dilation_base,
            residual_kernel_size=residual_kernel_size,
            n_residual_layers=n_residual_layers,
            activation=activation,
        )

    @classmethod
    def load_from_checkpoint(cls, config_path: str, ckpt_path: str):
        import json

        with open(config_path) as f:
            cfg = json.load(f)

        model = cls(cfg)
        params = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(params)
        return model

    def forward(self, x: torch.tensor, n_q: int = None, layers: list = [0]):
        """
        x: (batch, channels, timesteps)
        """

        # 🔥 APPLY MDCT HERE
        x = self.mdct(x)

        n_q = n_q if n_q else self.n_q

        e = self.encoder(x)

        quantized, codes, commit_loss, quantized_list = self.quantizer(
            e, n_q=n_q, layers=layers
        )

        feature = rearrange(quantized_list[0], "b d t -> b t d")
        feature = self.transform(feature)

        o = self.decoder(quantized)

        return o, commit_loss, feature, quantized_list[1:]

    def forward_feature(self, x: torch.tensor, layers: list = None):
        """
        x: (batch, channels, timesteps)
        """

        # 🔥 APPLY MDCT HERE
        x = self.mdct(x)

        e = self.encoder(x)

        layers = layers if layers else list(range(self.n_q))

        quantized, codes, commit_loss, quantized_list = self.quantizer(e, layers=layers)

        return quantized_list

    def encode(self, x: torch.tensor, n_q: int = None, st: int = None):
        """
        x: (batch, channels, timesteps)
        """

        # 🔥 APPLY MDCT HERE
        x = self.mdct(x)

        e = self.encoder(x)

        if st is None:
            st = 0

        n_q = n_q if n_q else self.n_q

        codes = self.quantizer.encode(e, n_q=n_q, st=st)

        return codes

    def decode(self, codes: torch.tensor, st: int = 0):
        """
        codes: (n_q, batch, timesteps)
        """

        quantized = self.quantizer.decode(codes, st=st)

        o = self.decoder(quantized)

        return o
