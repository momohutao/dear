from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Sequence

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


SCORE_POSITION_PAD_TOKEN_ID = -1


class RankerScorer:
    """Score query/passage pairs with the same packing used by DeAR training."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "float16",
        q_max_len: int = 32,
        p_max_len: int = 196,
    ):
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self.device = torch.device(device)
        self.dtype = dtype_map[dtype]
        self.q_max_len = q_max_len
        self.p_max_len = p_max_len
        adapter_config = Path(model_path) / "adapter_config.json"
        if adapter_config.is_file():
            config = PeftConfig.from_pretrained(model_path)
            base_path = config.base_model_name_or_path
            self.tokenizer = AutoTokenizer.from_pretrained(base_path)
            base = AutoModelForSequenceClassification.from_pretrained(
                base_path, num_labels=1, torch_dtype=self.dtype, low_cpu_mem_usage=True
            )
            self.model = PeftModel.from_pretrained(base, model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path, num_labels=1, torch_dtype=self.dtype, low_cpu_mem_usage=True
            )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.bos_token_id is None:
            raise ValueError(f"Tokenizer for {model_path} has no bos_token_id")
        self.tokenizer.padding_side = "right"
        self.model.config.pad_token_id = SCORE_POSITION_PAD_TOKEN_ID
        self.model.to(self.device).eval()

    def _pack(self, query: str, passage: dict) -> dict:
        query_ids = self.tokenizer.encode(
            "query: " + query,
            add_special_tokens=False,
            max_length=self.q_max_len,
            truncation=True,
        )
        document = (str(passage.get("title", "")) + " " + str(passage.get("text", ""))).strip()
        passage_ids = self.tokenizer.encode(
            "document: " + document,
            add_special_tokens=False,
            max_length=self.p_max_len - 3,
            truncation=True,
        )
        return self.tokenizer.prepare_for_model(
            [self.tokenizer.bos_token_id] + query_ids,
            [self.tokenizer.bos_token_id] + passage_ids,
            add_special_tokens=False,
            truncation="only_second",
            max_length=self.q_max_len + self.p_max_len,
            padding=False,
            return_token_type_ids=False,
        )

    @torch.inference_mode()
    def score(self, pairs: Sequence[tuple[str, dict]], batch_size: int) -> list[float]:
        results = []
        for start in range(0, len(pairs), batch_size):
            features = [self._pack(query, passage) for query, passage in pairs[start : start + batch_size]]
            batch = self.tokenizer.pad(
                features,
                padding="max_length",
                max_length=self.q_max_len + self.p_max_len,
                return_tensors="pt",
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            context = (
                torch.autocast(device_type="cuda", dtype=self.dtype)
                if self.device.type == "cuda" and self.dtype != torch.float32
                else nullcontext()
            )
            with context:
                logits = self.model(**batch, return_dict=True).logits.reshape(-1)
            results.extend(logits.float().cpu().tolist())
        return results

