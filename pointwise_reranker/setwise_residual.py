"""Continuous multi-context residual reranking for a trained setwise head.

The SLL head is trained to compare passages *within a set*.  Its softmax
outputs therefore must not be read as absolute relevance labels.  This module
keeps the head output continuous: normalize logits inside each context, average
the per-document contextual corrections, then add a small residual only inside
the initial DeAR top-K range.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import torch

from setwise_head import load_setwise_head


EPS = 1e-8


def _zscore(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean()) / (values.std(unbiased=False) + EPS)


def build_context_groups(count: int, set_size: int, rounds: int) -> List[List[List[int]]]:
    """Return ``rounds`` partitions; every document appears exactly once/round.

    Round 1 is consecutive initial ranks.  Round 2 is a circular rank shift,
    so different documents meet at group boundaries.  Round 3 alternates the
    head and boundary halves before chunking, ensuring direct head/boundary
    comparisons while preserving one appearance per document.
    """
    if count < 1 or set_size < 2 or count % set_size:
        raise ValueError("context count must be a positive multiple of set_size.")
    if not 1 <= rounds <= 3:
        raise ValueError("setwise_residual_context_rounds must be in [1, 3].")

    def chunk(order: Sequence[int]) -> List[List[int]]:
        return [list(order[start : start + set_size]) for start in range(0, count, set_size)]

    initial = list(range(count))
    partitions = [chunk(initial)]
    if rounds >= 2:
        # A non-zero shift smaller than one group changes the memberships,
        # unlike simply permuting the existing consecutive groups.
        shift = max(1, set_size // 2)
        shifted = initial[shift:] + initial[:shift]
        partitions.append(chunk(shifted))
    if rounds >= 3:
        half = count // 2
        if count % 2:
            raise ValueError("The head/boundary cross grouping requires an even top-K.")
        interleaved = [item for pair in zip(initial[:half], initial[half:]) for item in pair]
        partitions.append(chunk(interleaved))

    for partition in partitions:
        flattened = [index for group in partition for index in group]
        if sorted(flattened) != initial:
            raise AssertionError("Every contextual round must cover each document exactly once.")
    return partitions


def rerank_with_contextual_residual(
    entries: Sequence[Dict[str, Any]],
    checkpoint_path: str,
    *,
    top_k: int,
    set_size: int,
    rounds: int,
    gamma: float,
    head=None,
) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
    """Apply the stated top-K-only continuous residual and return full order."""
    if top_k < 2 or top_k % set_size:
        raise ValueError("setwise_residual_top_k must be >=2 and divisible by set_size.")
    if not math.isfinite(gamma):
        raise ValueError("setwise_residual_gamma must be finite.")

    # Stable original-rank tie breaking is needed both for protected outsiders
    # and for equal adapted scores.
    ordered = [
        entry for _, entry in sorted(
            enumerate(entries), key=lambda pair: (-float(pair[1]["z"]), pair[0], str(pair[1]["docid"]))
        )
    ]
    count = min(top_k, len(ordered))
    if count < top_k:
        raise ValueError(
            f"Query has only {count} candidates, but contextual top-K is {top_k}."
        )
    top_entries = ordered[:count]
    groups_by_round = build_context_groups(count, set_size, rounds)
    hidden = torch.stack([entry["h"].detach().cpu().float() for entry in top_entries])
    # In normal inference the caller loads this once after releasing DeAR and
    # reuses it for all queries.  Keeping the fallback makes the function
    # convenient and self-contained for unit tests.
    if head is None:
        head = load_setwise_head(checkpoint_path, map_location="cpu").float().eval()
    else:
        head = head.cpu().float().eval()

    contextual_sum = torch.zeros(count, dtype=torch.float32)
    observations = torch.zeros(count, dtype=torch.int64)
    group_traces: List[Dict[str, Any]] = []
    with torch.no_grad():
        for round_index, partition in enumerate(groups_by_round, start=1):
            for group in partition:
                logits = head(hidden[group].unsqueeze(0)).squeeze(0).float()
                normalized = _zscore(logits)
                for local_index, global_index in enumerate(group):
                    contextual_sum[global_index] += normalized[local_index]
                    observations[global_index] += 1
                group_traces.append({
                    "round": round_index,
                    "initial_ranks": [index + 1 for index in group],
                    "docids": [str(top_entries[index]["docid"]) for index in group],
                    "raw_logits": [float(value) for value in logits.tolist()],
                    "normalized_logits": [float(value) for value in normalized.tolist()],
                })
    if not torch.all(observations.eq(rounds)):
        raise AssertionError("Each top-K document must receive exactly one score per round.")

    dear_scores = torch.tensor([float(entry["z"]) for entry in top_entries])
    dear_normalized = _zscore(dear_scores)
    contextual = contextual_sum / observations.float()
    adapted_scores = dear_normalized + float(gamma) * contextual
    adapted_indices = sorted(
        range(count), key=lambda index: (
            -float(adapted_scores[index]), index, str(top_entries[index]["docid"])
        )
    )
    # The caller serializes this order with strictly descending scores.  That
    # prevents numeric score scales from allowing top-K outsiders back inside.
    results = [(str(top_entries[index]["docid"]), float(adapted_scores[index])) for index in adapted_indices]
    results.extend((str(entry["docid"]), float(entry["z"])) for entry in ordered[count:])
    trace = {
        "candidate_topk": count,
        "set_size": set_size,
        "context_rounds": rounds,
        "gamma": float(gamma),
        "groups": group_traces,
        "documents": [
            {
                "docid": str(entry["docid"]),
                "initial_rank": index + 1,
                "dear_score": float(entry["z"]),
                "dear_zscore": float(dear_normalized[index]),
                "contextual_score": float(contextual[index]),
                "observations": int(observations[index]),
                "adapted_score": float(adapted_scores[index]),
            }
            for index, entry in enumerate(top_entries)
        ],
    }
    return results, trace
