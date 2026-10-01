"""Lightweight contextual setwise head shared by training and inference."""

from __future__ import annotations

import json
import os
from typing import Dict

import torch
from torch import Tensor, nn


SETWISE_HEAD_WEIGHTS = "setwise_head.bin"
SETWISE_HEAD_CONFIG = "setwise_head_config.json"


class SetwiseContextualHead(nn.Module):
    """Project document representations, contextualize a set, emit one logit.

    The 3072-dimensional DeAR classifier-position representation is first
    projected to a small set space.  This keeps the requested two-layer
    self-attention head lightweight while every set logit still depends on all
    documents of the same query.
    """

    def __init__(self, input_dim: int, hidden_dim: int, heads: int, layers: int):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("set_hidden_dim must be divisible by set_heads.")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.layers = int(layers)
        self.input_projection = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.heads,
            dim_feedforward=self.hidden_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.layers)
        self.output_projection = nn.Linear(self.hidden_dim, 1)

    def forward(self, hidden: Tensor) -> Tensor:
        """``hidden`` is [query_batch, set_size, input_dim]."""
        contextual = self.encoder(self.input_projection(hidden))
        return self.output_projection(contextual).squeeze(-1)

    def export_config(self) -> Dict[str, int]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "heads": self.heads,
            "layers": self.layers,
        }


def save_setwise_head(head: SetwiseContextualHead, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    torch.save(head.state_dict(), os.path.join(output_dir, SETWISE_HEAD_WEIGHTS))
    with open(os.path.join(output_dir, SETWISE_HEAD_CONFIG), "w", encoding="utf-8") as file:
        json.dump(head.export_config(), file, indent=2, sort_keys=True)


def load_setwise_head(output_dir: str, *, map_location="cpu") -> SetwiseContextualHead:
    config_path = os.path.join(output_dir, SETWISE_HEAD_CONFIG)
    weights_path = os.path.join(output_dir, SETWISE_HEAD_WEIGHTS)
    if not (os.path.isfile(config_path) and os.path.isfile(weights_path)):
        raise FileNotFoundError(
            f"Missing {SETWISE_HEAD_CONFIG} or {SETWISE_HEAD_WEIGHTS} in {output_dir}."
        )
    with open(config_path, encoding="utf-8") as file:
        config = json.load(file)
    head = SetwiseContextualHead(**config)
    head.load_state_dict(torch.load(weights_path, map_location=map_location, weights_only=True))
    return head


def setwise_head_exists(output_dir: str) -> bool:
    return all(
        os.path.isfile(os.path.join(output_dir, name))
        for name in (SETWISE_HEAD_CONFIG, SETWISE_HEAD_WEIGHTS)
    )
