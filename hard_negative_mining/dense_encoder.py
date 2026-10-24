from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class DenseEncoder:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "float16",
        pooling: str = "mean",
        max_length: int = 256,
    ):
        if pooling not in {"mean", "cls"}:
            raise ValueError("pooling must be mean or cls")
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self.device = torch.device(device)
        self.pooling = pooling
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype_map[dtype],
            low_cpu_mem_usage=True,
        ).to(self.device).eval()
        self.dimension = int(self.model.config.hidden_size)

    @torch.inference_mode()
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        hidden = self.model(**encoded, return_dict=True).last_hidden_state
        if self.pooling == "cls":
            embeddings = hidden[:, 0]
        else:
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            embeddings = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return F.normalize(embeddings.float(), p=2, dim=-1).cpu()

