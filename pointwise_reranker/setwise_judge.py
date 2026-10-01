"""Setwise relevance judges used by inference-time TS-SetRank.

The pointwise DeAR and RankLLaMA models deliberately do not appear in this
module: evaluating passages independently is not a setwise judgement.  A
judge is optional and is loaded only when ``--use_ts_setrank`` is enabled.
"""

from __future__ import annotations

import abc
import json
import logging
import re
from typing import Callable, Iterable, Mapping, Optional, Sequence, Set, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from setwise_head import load_setwise_head


logger = logging.getLogger(__name__)


class SetwiseJudge(abc.ABC):
    """Returns relevant real document ids, or ``None`` if feedback is invalid."""

    @abc.abstractmethod
    def judge(
        self,
        query: str,
        documents: Sequence[str],
        document_ids: Sequence[str],
        seed: int,
    ) -> Optional[Set[str]]:
        """Judge a whole batch at once; ``None`` means no posterior update."""


def _valid_ids(ids: Iterable[object], allowed_ids: Sequence[str]) -> Set[str]:
    """Keep valid ids only; a set also removes duplicate model output ids."""
    allowed = set(map(str, allowed_ids))
    return {str(value) for value in ids if isinstance(value, str) and str(value) in allowed}


class MockSetwiseJudge(SetwiseJudge):
    """Deterministic test double, not a relevance model or an evaluation method.

    ``relevant`` may be a mapping from query string to ids, a fixed iterable of
    ids, or a callable receiving ``(query, documents, document_ids, seed)``.
    This makes unit tests able to exercise positive, negative and malformed
    judge-feedback paths without accessing DL qrels.
    """

    def __init__(
        self,
        relevant: Optional[
            Union[
                Mapping[str, Iterable[str]],
                Iterable[str],
                Callable[[str, Sequence[str], Sequence[str], int], Optional[Iterable[str]]],
            ]
        ] = None,
    ) -> None:
        self.relevant = relevant

    def judge(self, query, documents, document_ids, seed):
        if callable(self.relevant):
            response = self.relevant(query, documents, document_ids, seed)
        elif isinstance(self.relevant, Mapping):
            response = self.relevant.get(query, [])
        elif self.relevant is None:
            # A default mock has no hidden relevance oracle.  Treat it as no
            # feedback rather than inventing negative labels for every batch.
            response = None
        else:
            response = self.relevant
        if response is None:
            return None
        return _valid_ids(response, document_ids)


class ContextualSetwiseJudge(SetwiseJudge):
    """CPU inference wrapper for the trained DeAR contextual setwise head."""

    def __init__(self, checkpoint_path: str, model_args: object) -> None:
        self.head = load_setwise_head(checkpoint_path, map_location="cpu").float().eval()
        self.temperature = float(getattr(model_args, "setwise_feedback_temperature"))
        self.mode = getattr(model_args, "setwise_feedback_mode")
        self.top_k = int(getattr(model_args, "setwise_feedback_top_k"))
        self._embeddings = None
        if self.temperature <= 0:
            raise ValueError("setwise_feedback_temperature must be positive.")
        if self.mode not in {"topk", "threshold"}:
            raise ValueError("setwise_feedback_mode must be topk or threshold.")

    def set_query_embeddings(self, embeddings_by_docid: Mapping[str, torch.Tensor]) -> None:
        """Install detached CPU classifier-position embeddings for one query."""
        self._embeddings = {
            str(docid): embedding.detach().cpu().float()
            for docid, embedding in embeddings_by_docid.items()
        }

    def judge(self, query, documents, document_ids, seed):
        if self._embeddings is None:
            raise RuntimeError("Contextual setwise judge has no query-local embeddings.")
        if not document_ids or len(documents) != len(document_ids):
            return None
        try:
            hidden = torch.stack([self._embeddings[str(docid)] for docid in document_ids])
        except KeyError as exc:
            raise RuntimeError(f"Missing cached DeAR embedding for {exc.args[0]}.") from exc
        with torch.no_grad():
            logits = self.head(hidden.unsqueeze(0)).squeeze(0)
            probabilities = torch.softmax(logits / self.temperature, dim=-1)
        if self.mode == "topk":
            indices = torch.topk(probabilities, k=min(self.top_k, len(document_ids))).indices.tolist()
        else:
            indices = torch.nonzero(
                probabilities > (1.0 / len(document_ids)), as_tuple=False
            ).flatten().tolist()
            if not indices:
                indices = [int(probabilities.argmax().item())]
        return {str(document_ids[index]) for index in indices}


class GenerativeSetwiseJudge(SetwiseJudge):
    """A batch-wise generative judge with compact temporary-id feedback."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        temperature: float = 0.6,
        load_in_4bit: bool = False,
        max_new_tokens: int = 128,
        max_input_tokens: int = 4096,
        document_max_chars: int = 1600,
        cache_dir: Optional[str] = None,
    ) -> None:
        if not model_name_or_path:
            raise ValueError("A generative setwise judge requires setwise_model_name_or_path.")
        if temperature <= 0:
            raise ValueError("ts_judge_temperature must be positive.")
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        self.document_max_chars = document_max_chars
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        load_kwargs = {"cache_dir": cache_dir, "device_map": "auto"}
        if load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            load_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
        self.model.eval()

    def _prompt(
        self,
        query: str,
        documents: Sequence[str],
        strict_retry: bool,
        document_char_limit: Optional[int] = None,
    ) -> str:
        char_limit = self.document_max_chars if document_char_limit is None else document_char_limit
        passages = "\n".join(
            # Repeating the complete symbolic form before each passage is
            # important for base Llama: `[P0]` often leads it to emit bare
            # numbers, whereas `Passage ID P0:` makes it copy P-prefixed ids.
            f"Passage ID P{index}: {str(document)[:char_limit]}"
            for index, document in enumerate(documents)
        )
        instruction = (
            "Select every passage relevant to the query. Reply with only its ids, "
            "space-separated (example: P0 P3), or exactly NONE. No explanation."
            if strict_retry
            else "Select the passages relevant to the query."
        )
        # Ending on the compact answer cue lets a base CausalLM continue with
        # copied identifiers; verbose JSON instructions often make it emit an
        # immediate EOS token instead.
        return (
            f"Query:\n{query}\n\nPassages:\n{passages}\n\n{instruction}\n"
            "Relevant passage IDs:"
        )

    def _fit_prompt(self, query: str, documents: Sequence[str], strict_retry: bool) -> str:
        """Fit a prompt without silently dropping trailing batch passages.

        Tokenizer-level ``truncation=True`` could remove P8/P9 while their ids
        are still accepted by the parser.  We instead shorten every document
        equally until the full set of temporary ids remains inside context.
        """
        char_limit = self.document_max_chars
        while char_limit >= 32:
            prompt = self._prompt(query, documents, strict_retry, char_limit)
            length = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
            if length <= self.max_input_tokens:
                return prompt
            char_limit = max(31, int(char_limit * self.max_input_tokens / length))
        raise ValueError("Setwise prompt is too long even after balanced document truncation.")

    @staticmethod
    def _parse_temporary_ids(text: str, valid_temp_ids: Set[str]) -> Optional[Set[str]]:
        """Parse compact ``P0 P3`` output, retaining legacy JSON support.

        Compact IDs are the primary protocol because small base CausalLMs are
        much more reliable at copying identifiers than emitting strict JSON.
        JSON remains accepted so an instruction model using the old protocol
        continues to work.  Blank or unrelated output is *not* interpreted as
        "no relevant documents": it is invalid feedback and cannot poison a
        Beta posterior with invented failures.
        """
        # The requested compact protocol is a single answer line.  Looking
        # only at that line avoids treating IDs quoted later in an unsolicited
        # explanation as additional relevant documents.
        first_answer_line = next((line for line in text.splitlines() if line.strip()), "")
        compact_ids = re.findall(r"\bP\d+\b", first_answer_line)
        if compact_ids:
            return {item for item in compact_ids if item in valid_temp_ids}
        if re.fullmatch(r"\s*(?:NONE|NO\s+RELEVANT(?:\s+PASSAGES)?)\s*[.!]?\s*", text, flags=re.IGNORECASE):
            return set()
        candidates = [text.strip()]
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if fenced:
            candidates.append(fenced.group(1))
        object_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if object_match:
            candidates.append(object_match.group(0))
        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            ids = payload.get("relevant_ids") if isinstance(payload, dict) else None
            if not isinstance(ids, list):
                continue
            return {item for item in ids if isinstance(item, str) and item in valid_temp_ids}
        return None

    def _generate(self, prompt: str, seed: int) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False)
        device = self.model.get_input_embeddings().weight.device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        # Older transformers releases used in this project do not accept a
        # ``generator=`` argument to generate().  fork_rng restores the CPU
        # and CUDA RNG states afterwards, so query-local sampling remains
        # reproducible without leaking a seed into DeAR or later queries.
        fork_devices = [device.index] if device.type == "cuda" and device.index is not None else []
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(int(seed))
            if device.type == "cuda":
                torch.cuda.manual_seed(int(seed))
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    # The local base Llama otherwise often chooses EOS before
                    # emitting even its first identifier.  Four tokens are
                    # enough for " P0" plus a delimiter but remain far below
                    # the normal answer budget.
                    min_new_tokens=min(4, self.max_new_tokens),
                    do_sample=True,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        generated = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def judge(self, query, documents, document_ids, seed):
        if len(documents) != len(document_ids) or not document_ids:
            logger.warning("Setwise judge received an empty or misaligned batch.")
            return None
        temporary_ids = [f"P{index}" for index in range(len(document_ids))]
        temporary_to_docid = dict(zip(temporary_ids, map(str, document_ids)))
        for attempt in range(2):
            try:
                response = self._generate(
                    self._fit_prompt(query, documents, strict_retry=attempt == 1),
                    seed + attempt,
                )
            except Exception as exc:  # generation failure must not abort inference
                logger.warning("Setwise judge generation failed on attempt %d: %s", attempt + 1, exc)
                continue
            selected = self._parse_temporary_ids(response, set(temporary_ids))
            if selected is not None:
                return {temporary_to_docid[temp_id] for temp_id in selected}
            logger.warning("Setwise judge returned invalid setwise ids on attempt %d: %r", attempt + 1, response[:300])
        logger.warning("Setwise judge supplied no valid feedback after one retry.")
        return None


def build_setwise_judge(model_args: object) -> SetwiseJudge:
    """Instantiate a judge only in the TS-SetRank inference branch."""
    judge_type = getattr(model_args, "ts_judge_type", "generative")
    if judge_type == "mock":
        logger.warning("Using MockSetwiseJudge: this is a plumbing test, not a DL19 result.")
        return MockSetwiseJudge()
    if judge_type == "contextual":
        return ContextualSetwiseJudge(
            getattr(model_args, "model_name_or_path"), model_args
        )
    if judge_type != "generative":
        raise ValueError(f"Unsupported ts_judge_type: {judge_type}")
    return GenerativeSetwiseJudge(
        getattr(model_args, "setwise_model_name_or_path"),
        temperature=getattr(model_args, "ts_judge_temperature"),
        load_in_4bit=getattr(model_args, "ts_judge_load_in_4bit"),
        max_new_tokens=getattr(model_args, "ts_judge_max_new_tokens"),
        max_input_tokens=getattr(model_args, "ts_judge_max_input_tokens"),
        document_max_chars=getattr(model_args, "ts_judge_document_max_chars"),
        cache_dir=getattr(model_args, "cache_dir", None),
    )
