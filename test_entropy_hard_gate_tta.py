#!/usr/bin/env python
"""CPU-only invariants for holdout entropy hard-routing TTA."""
from types import SimpleNamespace
import tempfile

import torch

from pointwise_reranker.entropy_hard_gate_tta import (
    adapt_entropy_hard_gate_one_query,
    alternating_folds,
    normalized_entropy,
)
from pointwise_reranker.teacher_score_cache import append_teacher_score_cache, load_teacher_score_cache


def args(**updates):
    values = dict(
        tta_candidate_k=30, tta_gamma=0.1, tta_steps=4, tta_lr=1e-2,
        tta_teacher_temperature=2.0, tta_student_temperature=2.0,
        tta_entropy_temperature=2.0, tta_entropy_threshold=0.0,
        tta_anchor_weight=0.01, tta_u_norm_weight=1e-4,
        tta_force_adapted=False, tta_single_fold_gate=False,
    )
    values.update(updates)
    return SimpleNamespace(**values)


def entries(count=30):
    generator = torch.Generator().manual_seed(7)
    hidden = torch.randn(count, 16, generator=generator)
    # teacher deliberately disagrees with the initial order; no labels/qrels.
    return [
        {"docid": f"d{i}", "z": float(count - i), "teacher_z": float(i), "h": hidden[i]}
        for i in range(count)
    ]


def docids(result):
    return [docid for docid, _ in result]


def main():
    device = torch.device("cpu")
    fold_a, fold_v = alternating_folds(30, device)
    assert fold_a.tolist() == list(range(0, 30, 2))
    assert fold_v.tolist() == list(range(1, 30, 2))
    assert 0.0 <= float(normalized_entropy(torch.tensor([0.2] * 5))) <= 1.0

    base, base_trace = adapt_entropy_hard_gate_one_query(entries(), args(tta_gamma=0.0), device)
    assert base_trace["selected_branch"] == "base" and base_trace["skip_reason"] == "zero_gamma"
    assert docids(base) == [f"d{i}" for i in range(30)]

    zero_steps, trace = adapt_entropy_hard_gate_one_query(entries(), args(tta_steps=0), device)
    assert trace["selected_branch"] == "base" and trace["skip_reason"] == "zero_tta_steps"
    assert docids(zero_steps) == docids(base)

    forced, forced_trace = adapt_entropy_hard_gate_one_query(entries(), args(tta_force_adapted=True), device)
    assert forced_trace["selected_branch"] == "adapted"
    assert forced_trace["hard_branch_identity"]
    assert not forced_trace["student_grad_present"] and not forced_trace["teacher_grad_present"]
    assert len(forced) == 30 and set(docids(forced)) == set(docids(base))

    small, small_trace = adapt_entropy_hard_gate_one_query(entries(7), args(tta_candidate_k=30), device)
    assert len(small) == 7
    assert small_trace["candidate_count"] == 7

    with tempfile.TemporaryDirectory() as directory:
        path = f"{directory}/teacher.jsonl"
        append_teacher_score_cache(path, "/teacher", 30, [{"qid": "q", "docid": "d", "score": 1.5}])
        assert load_teacher_score_cache(path, "/teacher", 30)[("q", "d", "/teacher", 30)] == 1.5
        assert load_teacher_score_cache(path, "/teacher", 20) == {}
    print("entropy hard-gate CPU unit tests passed")


if __name__ == "__main__":
    main()
