"""Holdout-entropy-consistent, hard-routed query-level TTA.

The only trainable objects are ephemeral query vectors.  The final result is
always one complete branch: frozen DeAR+KK+SLL (``base``) or a separately
optimized Top-K residual branch (``adapted``).  This module intentionally
contains no qrels access and never blends two branch score vectors.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F


EPS = 1e-8


def zscore(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean()) / (values.std(unbiased=False) + EPS)


def alternating_folds(count: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ranks 1,3,... and 2,4,...; accepts a dynamically shortened Top-K."""
    indices = torch.arange(count, device=device)
    return indices[::2], indices[1::2]


def normalized_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    if probabilities.numel() < 2:
        return probabilities.new_tensor(float("nan"))
    entropy = -(probabilities * probabilities.clamp_min(EPS).log()).sum()
    return entropy / math.log(float(probabilities.numel()))


def _kl_teacher_to_student(
    teacher_hat: torch.Tensor,
    scores: torch.Tensor,
    index: torch.Tensor,
    teacher_temperature: float,
    student_temperature: float,
) -> torch.Tensor:
    teacher_prob = F.softmax(teacher_hat[index] / teacher_temperature, dim=-1).detach()
    student_log_prob = F.log_softmax(scores[index] / student_temperature, dim=-1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")


def _optimize_query_vector(
    hidden: torch.Tensor,
    base_scores: torch.Tensor,
    teacher_hat: torch.Tensor,
    loss_index: torch.Tensor,
    model_args: object,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Optimize a fresh FP32 vector using *only* ``loss_index`` documents."""
    u_q = torch.zeros(
        hidden.size(-1), device=hidden.device, dtype=torch.float32, requires_grad=True
    )
    result: Dict[str, Any] = {
        "initial_loss": None,
        "final_loss": None,
        "steps": 0,
        "valid": False,
        "reason": None,
        "u_grad_present": False,
    }
    if int(model_args.tta_steps) <= 0:
        result["reason"] = "zero_tta_steps"
        return u_q.detach(), result
    optimizer = torch.optim.AdamW([u_q], lr=float(model_args.tta_lr), weight_decay=0.0)

    def objective() -> Tuple[torch.Tensor, torch.Tensor]:
        bounded = torch.tanh(hidden @ u_q)
        adapted = base_scores + float(model_args.tta_gamma) * bounded
        kl = _kl_teacher_to_student(
            teacher_hat,
            adapted,
            loss_index,
            float(model_args.tta_teacher_temperature),
            float(model_args.tta_student_temperature),
        )
        loss = (
            kl
            + float(model_args.tta_anchor_weight) * bounded[loss_index].square().mean()
            + float(model_args.tta_u_norm_weight) * u_q.square().mean()
        )
        return loss, bounded

    with torch.no_grad():
        initial_loss, _ = objective()
    if not torch.isfinite(initial_loss):
        result["reason"] = "nonfinite_initial_loss"
        del optimizer
        return u_q.detach(), result
    result["initial_loss"] = float(initial_loss.item())
    best_loss = float(initial_loss.item())
    best_u = u_q.detach().clone()
    for step in range(int(model_args.tta_steps)):
        loss, _ = objective()
        if not torch.isfinite(loss):
            result["reason"] = "nonfinite_optimization_loss"
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if u_q.grad is None or not torch.isfinite(u_q.grad).all():
            result["reason"] = "nonfinite_or_missing_u_gradient"
            break
        result["u_grad_present"] = True
        gradient = u_q.grad.detach().clone()
        before_step_u = u_q.detach().clone()
        optimizer.step()
        result["steps"] = step + 1
        with torch.no_grad():
            candidate_loss, _ = objective()
        if not torch.isfinite(candidate_loss):
            result["reason"] = "nonfinite_post_step_loss"
            break
        # The temperature-smoothed KL is often O(1e-3); a sign-normalized
        # Adam first step at 1e-2 can overshoot the anchor optimum.  Enforce
        # conservative descent of the complete fold objective.  This only
        # changes the ephemeral u_q and never touches student/teacher weights.
        if float(candidate_loss.item()) > float(loss.detach().item()):
            adam_candidate = u_q.detach().clone()
            accepted = False
            factors = (0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625)
            with torch.no_grad():
                for factor in factors:
                    u_q.copy_(before_step_u + factor * (adam_candidate - before_step_u))
                    trial_loss, _ = objective()
                    if torch.isfinite(trial_loss) and float(trial_loss.item()) <= float(loss.detach().item()):
                        candidate_loss = trial_loss
                        accepted = True
                        break
                if not accepted:
                    norm = gradient.norm().clamp_min(EPS)
                    gradient_candidate = before_step_u - float(model_args.tta_lr) * gradient / norm
                    for factor in factors:
                        u_q.copy_(before_step_u + factor * (gradient_candidate - before_step_u))
                        trial_loss, _ = objective()
                        if torch.isfinite(trial_loss) and float(trial_loss.item()) <= float(loss.detach().item()):
                            candidate_loss = trial_loss
                            accepted = True
                            break
                if not accepted:
                    u_q.copy_(before_step_u)
                    candidate_loss = loss.detach()
        if float(candidate_loss.item()) < best_loss:
            best_loss = float(candidate_loss.item())
            best_u.copy_(u_q.detach())
    else:
        result["reason"] = None

    with torch.no_grad():
        u_q.copy_(best_u)
        final_loss, _ = objective()
    if torch.isfinite(final_loss):
        result["final_loss"] = float(final_loss.item())
        result["valid"] = result["reason"] is None
    else:
        result["reason"] = "nonfinite_final_loss"
    output = u_q.detach().clone()
    del optimizer, u_q
    return output, result


def _entropy_from_scores(scores: torch.Tensor, index: torch.Tensor, temperature: float) -> torch.Tensor:
    return normalized_entropy(F.softmax(scores[index] / temperature, dim=-1))


def adapt_entropy_hard_gate_one_query(
    entries: Sequence[Dict[str, Any]], model_args: object, device: torch.device
) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
    """Return an exact base/adapted branch choice for one query.

    Entries are sorted by original DeAR logit once.  Every transient tensor is
    local to this call, which provides the required episodic reset.
    """
    ordered = [
        item for _, item in sorted(
            enumerate(entries), key=lambda pair: (-float(pair[1]["z"]), pair[0], str(pair[1]["docid"]))
        )
    ]
    total_count = len(ordered)
    count = min(int(model_args.tta_candidate_k), total_count)
    diag: Dict[str, Any] = {
        "candidate_count": count,
        "base_entropy_on_a": None, "base_entropy_on_v": None,
        "adapt_entropy_on_a": None, "adapt_entropy_on_v": None,
        "delta_entropy_a": None, "delta_entropy_v": None,
        "fold_a_initial_loss": None, "fold_a_final_loss": None,
        "fold_v_initial_loss": None, "fold_v_final_loss": None,
        "fold_a_steps": 0, "fold_v_steps": 0,
        "selected_branch": "base", "skip_reason": None,
        "full_adaptation_steps": 0,
        "mean_abs_residual": 0.0, "max_abs_residual": 0.0,
        "top1_changed": False, "top10_order_changed": False,
        "top10_set_changed": False, "top10_replacement_count": 0,
        "student_grad_present": False, "teacher_grad_present": False,
        "hard_branch_identity": True,
    }

    # s0 is the original DeAR+KK+SLL branch on the whole candidate list.  The
    # optimizer works on z-hat, but for output we retain the original-logit
    # scale: z + sigma_z*r is a strictly positive affine transform of
    # z-hat+r.  Therefore it has exactly the requested adapted ordering while
    # a rejected base branch is byte-for-byte the ordinary DeAR score branch.
    all_z = torch.tensor([item["z"] for item in ordered], dtype=torch.float32, device=device).detach()
    base_all_scores = zscore(all_z).detach()
    output_scale = all_z.std(unbiased=False) + EPS

    def base_result(reason: str | None) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
        diag["skip_reason"] = reason
        return [
            (str(item["docid"]), float(all_z[index]))
            for index, item in enumerate(ordered)
        ], diag

    if count < 2:
        return base_result("fewer_than_two_candidates")
    top = ordered[:count]
    if any(item.get("teacher_z") is None or item.get("h") is None for item in top):
        return base_result("missing_teacher_score_or_hidden")
    z = all_z[:count]
    teacher = torch.tensor([item["teacher_z"] for item in top], dtype=torch.float32, device=device).detach()
    hidden = torch.stack([item["h"] for item in top]).to(device=device, dtype=torch.float32).detach()
    hidden = F.normalize(hidden, p=2, dim=-1).detach()
    if not torch.isfinite(z).all() or not torch.isfinite(teacher).all() or not torch.isfinite(hidden).all():
        return base_result("nonfinite_input")
    if teacher.std(unbiased=False).item() <= EPS:
        return base_result("near_zero_teacher_variance")
    if float(model_args.tta_gamma) == 0.0:
        return base_result("zero_gamma")
    if int(model_args.tta_steps) <= 0:
        return base_result("zero_tta_steps")
    if float(model_args.tta_entropy_temperature) <= 0:
        return base_result("nonpositive_entropy_temperature")
    a_idx, v_idx = alternating_folds(count, device)
    if a_idx.numel() < 2 or v_idx.numel() < 2:
        return base_result("fold_with_fewer_than_two_candidates")
    base_scores = base_all_scores[:count]
    teacher_hat = zscore(teacher).detach()
    entropy_temperature = float(model_args.tta_entropy_temperature)
    with torch.no_grad():
        base_entropy_a = _entropy_from_scores(base_scores, a_idx, entropy_temperature)
        base_entropy_v = _entropy_from_scores(base_scores, v_idx, entropy_temperature)
    if not torch.isfinite(base_entropy_a) or not torch.isfinite(base_entropy_v):
        return base_result("nonfinite_base_entropy")
    diag["base_entropy_on_a"] = float(base_entropy_a.item())
    diag["base_entropy_on_v"] = float(base_entropy_v.item())

    # Fold A is optimized on A only and evaluated only on the held-out V.
    u_a, fold_a = _optimize_query_vector(hidden, base_scores, teacher_hat, a_idx, model_args)
    diag["fold_a_initial_loss"], diag["fold_a_final_loss"] = fold_a["initial_loss"], fold_a["final_loss"]
    diag["fold_a_steps"] = fold_a["steps"]
    if not fold_a["valid"]:
        return base_result("fold_a_" + str(fold_a["reason"]))
    with torch.no_grad():
        scores_by_a = base_scores + float(model_args.tta_gamma) * torch.tanh(hidden @ u_a)
        adapt_entropy_v = _entropy_from_scores(scores_by_a, v_idx, entropy_temperature)
    if not torch.isfinite(adapt_entropy_v):
        return base_result("nonfinite_adapt_entropy_v")
    delta_a = base_entropy_v - adapt_entropy_v
    diag["adapt_entropy_on_v"] = float(adapt_entropy_v.item())
    diag["delta_entropy_a"] = float(delta_a.item())

    # This is intentionally a distinct zero initialization and optimizer.
    u_v, fold_v = _optimize_query_vector(hidden, base_scores, teacher_hat, v_idx, model_args)
    diag["fold_v_initial_loss"], diag["fold_v_final_loss"] = fold_v["initial_loss"], fold_v["final_loss"]
    diag["fold_v_steps"] = fold_v["steps"]
    if not fold_v["valid"]:
        return base_result("fold_v_" + str(fold_v["reason"]))
    with torch.no_grad():
        scores_by_v = base_scores + float(model_args.tta_gamma) * torch.tanh(hidden @ u_v)
        adapt_entropy_a = _entropy_from_scores(scores_by_v, a_idx, entropy_temperature)
    if not torch.isfinite(adapt_entropy_a):
        return base_result("nonfinite_adapt_entropy_a")
    delta_v = base_entropy_a - adapt_entropy_a
    diag["adapt_entropy_on_a"] = float(adapt_entropy_a.item())
    diag["delta_entropy_v"] = float(delta_v.item())

    threshold = float(model_args.tta_entropy_threshold)
    gate_a = bool(delta_a.item() > threshold)
    gate_v = bool(delta_v.item() > threshold)
    choose_adapted = bool(model_args.tta_force_adapted) or (
        gate_a if model_args.tta_single_fold_gate else (gate_a and gate_v)
    )
    if not choose_adapted:
        return base_result("holdout_entropy_gate_rejected")

    full_idx = torch.arange(count, device=device)
    u_full, full = _optimize_query_vector(hidden, base_scores, teacher_hat, full_idx, model_args)
    diag["full_adaptation_steps"] = full["steps"]
    if not full["valid"]:
        return base_result("full_" + str(full["reason"]))
    with torch.no_grad():
        residual = float(model_args.tta_gamma) * torch.tanh(hidden @ u_full)
        if not torch.isfinite(residual).all():
            return base_result("nonfinite_full_residual")
        adapted_top_scores = base_scores + residual
    base_order = list(range(count))
    adapted_order = sorted(range(count), key=lambda i: (-float(adapted_top_scores[i]), i, str(top[i]["docid"])))
    base_docids = [str(top[i]["docid"]) for i in base_order]
    adapted_docids = [str(top[i]["docid"]) for i in adapted_order]
    diag["selected_branch"] = "adapted"
    diag["skip_reason"] = None
    diag["mean_abs_residual"] = float(residual.abs().mean().item())
    diag["max_abs_residual"] = float(residual.abs().max().item())
    diag["top1_changed"] = adapted_docids[0] != base_docids[0]
    diag["top10_order_changed"] = adapted_docids[:10] != base_docids[:10]
    diag["top10_set_changed"] = set(adapted_docids[:10]) != set(base_docids[:10])
    diag["top10_replacement_count"] = len(set(adapted_docids[:10]) - set(base_docids[:10]))
    # No interpolation: the returned list is the complete adapted branch.
    full_adapted_scores = all_z.clone()
    full_adapted_scores[:count] = all_z[:count] + output_scale * residual
    # This is s1 exactly up to its shared positive affine score transform:
    # Top-K residuals plus unchanged s0 logits outside K.
    result = [
        (str(item["docid"]), float(full_adapted_scores[index]))
        for index, item in enumerate(ordered)
    ]
    del u_a, u_v, u_full
    return result, diag
