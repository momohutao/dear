"""Query-local Thompson-sampling SetRank inference post-processor."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Dict, List, Sequence, Tuple

from setwise_judge import SetwiseJudge


EPS = 1e-8


def _zscore(values: Sequence[float]) -> List[float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    scale = math.sqrt(variance)
    return [(value - mean) / (scale + EPS) for value in values]


@dataclass
class _Posterior:
    docid: str
    document: str
    dear_score: float
    original_rank: int
    alpha: float
    beta: float
    selection_count: int = 0
    sample_count: int = 0
    success_count: int = 0
    failure_count: int = 0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


class TSSetRanker:
    """Fresh Beta posteriors for one query; no model parameter is updated."""

    def __init__(self, model_args: object, judge: SetwiseJudge) -> None:
        self.args = model_args
        self.judge = judge
        self._validate()

    def _validate(self) -> None:
        if self.args.ts_candidate_topk < 1 or self.args.ts_batch_size < 1:
            raise ValueError("ts_candidate_topk and ts_batch_size must be positive.")
        if self.args.ts_budget < 0 or self.args.ts_exploration_rounds < 0:
            raise ValueError("TS round counts must be non-negative.")
        if self.args.ts_prior_mode not in {"uniform", "dear"}:
            raise ValueError("ts_prior_mode must be uniform or dear.")
        if self.args.ts_final_mode not in {"posterior", "fusion"}:
            raise ValueError("ts_final_mode must be posterior or fusion.")
        if not 0.0 <= self.args.ts_fusion_weight <= 1.0:
            raise ValueError("ts_fusion_weight must be in [0, 1].")
        if self.args.ts_prior_temperature <= 0:
            raise ValueError("ts_prior_temperature must be positive.")

    @staticmethod
    def _initial_order(entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # enumerate preserves the incoming candidate order as the original
        # rank tie-breaker. docid remains the final deterministic tie-breaker.
        indexed = list(enumerate(entries))
        return [
            item for _, item in sorted(
                indexed,
                key=lambda pair: (-float(pair[1]["z"]), pair[0], str(pair[1]["docid"])),
            )
        ]

    def _build_posteriors(self, entries: Sequence[Dict[str, Any]]) -> List[_Posterior]:
        scores = [float(entry["z"]) for entry in entries]
        normalized = _zscore(scores)
        states: List[_Posterior] = []
        for original_rank, (entry, score_norm) in enumerate(zip(entries, normalized), start=1):
            if self.args.ts_prior_mode == "uniform":
                alpha, beta = 1.0, 1.0
            else:
                probability = 1.0 / (1.0 + math.exp(-score_norm / self.args.ts_prior_temperature))
                alpha = 1.0 + self.args.ts_prior_strength * probability
                beta = 1.0 + self.args.ts_prior_strength * (1.0 - probability)
            states.append(_Posterior(
                docid=str(entry["docid"]),
                document=str(entry["document"]),
                dear_score=float(entry["z"]),
                original_rank=original_rank,
                alpha=alpha,
                beta=beta,
            ))
        return states

    @staticmethod
    def _state_key(state: _Posterior) -> Tuple[float, int, str]:
        return (-state.dear_score, state.original_rank, state.docid)

    def _choose_batch(
        self,
        candidates: Sequence[_Posterior],
        round_index: int,
        rng: random.Random,
    ) -> List[_Posterior]:
        size = min(self.args.ts_batch_size, len(candidates))
        if round_index < self.args.ts_exploration_rounds:
            # Selection and successful posterior observation are separate:
            # failed judge calls must not repeatedly consume the same first
            # exploration slots, while ``sample_count`` remains an auditable
            # count of actual Beta observations.
            unseen = [state for state in candidates if state.selection_count == 0]
            selected = rng.sample(unseen, min(size, len(unseen)))
            if len(selected) < size:
                remaining = [state for state in candidates if state not in selected]
                selected.extend(rng.sample(remaining, size - len(selected)))
            return selected
        draws = [(rng.betavariate(state.alpha, state.beta), state) for state in candidates]
        draws.sort(key=lambda pair: (-pair[0], *self._state_key(pair[1])))
        return [state for _, state in draws[:size]]

    def rerank(
        self,
        query_id: str,
        query: str,
        entries: Sequence[Dict[str, Any]],
        query_index: int,
    ) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
        """Return a top-K-only rerank and auditable query-local posterior trace."""
        initial_entries = self._initial_order(entries)
        states = self._build_posteriors(initial_entries)
        candidate_count = min(self.args.ts_candidate_topk, len(states))
        candidates = states[:candidate_count]
        rng = random.Random(int(self.args.ts_seed) + int(query_index))
        feedback_failures = 0

        for round_index in range(self.args.ts_budget):
            batch = self._choose_batch(candidates, round_index, rng)
            for state in batch:
                state.selection_count += 1
            response = self.judge.judge(
                query,
                [state.document for state in batch],
                [state.docid for state in batch],
                int(self.args.ts_seed) + int(query_index) + round_index,
            )
            if response is None:
                feedback_failures += 1
                continue
            relevant_ids = set(response)
            for state in batch:
                state.sample_count += 1
                if state.docid in relevant_ids:
                    state.alpha += 1.0
                    state.success_count += 1
                else:
                    state.beta += 1.0
                    state.failure_count += 1

        if self.args.ts_final_mode == "posterior":
            final_scores = [state.mean for state in candidates]
        else:
            dear_norm = _zscore([state.dear_score for state in candidates])
            ts_logit = [math.log((state.mean + EPS) / (1.0 - state.mean + EPS)) for state in candidates]
            ts_norm = _zscore(ts_logit)
            final_scores = [
                (1.0 - self.args.ts_fusion_weight) * dear + self.args.ts_fusion_weight * posterior
                for dear, posterior in zip(dear_norm, ts_norm)
            ]
        for state, final_score in zip(candidates, final_scores):
            state.final_score = final_score
        for state in states[candidate_count:]:
            state.final_score = state.dear_score
        candidates.sort(key=lambda state: (-state.final_score, *self._state_key(state)))
        # Candidates outside K are never mixed back into the adapted range.
        results = [(state.docid, state.final_score) for state in candidates]
        results.extend((state.docid, state.dear_score) for state in states[candidate_count:])
        trace = {
            "query_id": str(query_id),
            "query_index": int(query_index),
            "seed": int(self.args.ts_seed) + int(query_index),
            "candidate_topk": candidate_count,
            "budget": int(self.args.ts_budget),
            "batch_size": int(self.args.ts_batch_size),
            "feedback_failures": feedback_failures,
            "documents": [
                {
                    "docid": state.docid,
                    "original_rank": state.original_rank,
                    "dear_score": state.dear_score,
                    "selection_count": state.selection_count,
                    "sample_count": state.sample_count,
                    "success_count": state.success_count,
                    "failure_count": state.failure_count,
                    "alpha": state.alpha,
                    "beta": state.beta,
                    "posterior_mean": state.mean,
                    "final_score": getattr(state, "final_score", None),
                }
                for state in states
            ],
        }
        return results, trace
