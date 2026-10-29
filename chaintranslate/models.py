"""Small, randomly initialized encoder-decoder Transformer baseline."""

import math
import torch
from torch import nn


class BasicTransformer(nn.Module):
    def __init__(self, vocab, dim=128):
        super().__init__()
        self.dim = dim
        self.embedding = nn.Embedding(vocab, dim, padding_idx=0)
        self.transformer = nn.Transformer(
            d_model=dim,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
        )
        self.head = nn.Linear(dim, vocab)
        pos = torch.arange(256).unsqueeze(1)
        freq = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000) / dim))
        pe = torch.zeros(256, dim)
        pe[:, 0::2] = torch.sin(pos * freq)
        pe[:, 1::2] = torch.cos(pos * freq)
        self.register_buffer("position", pe)

    def embed(self, x):
        return self.embedding(x) * math.sqrt(self.dim) + self.position[: x.shape[1]]

    def forward(self, source, target):
        mask = torch.triu(
            torch.ones(
                target.shape[1], target.shape[1], device=source.device, dtype=torch.bool
            ),
            1,
        )
        out = self.transformer(
            self.embed(source),
            self.embed(target),
            tgt_mask=mask,
            src_key_padding_mask=source.eq(0),
            tgt_key_padding_mask=target.eq(0),
            memory_key_padding_mask=source.eq(0),
        )
        return self.head(out)
