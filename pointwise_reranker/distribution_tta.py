"""RankLLaMA continuous-feedback query-level distribution TTA.

Only a fresh query-local vector receives gradients.  Student/teacher logits
and classifier-position states are detached before entering this module.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


EPS = 1e-8


def _zscore(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean()) / (values.std(unbiased=False) + EPS)


def build_overlapping_windows(count: int, window_size: int, stride: int) -> List[Tuple[int, int]]:
    """Deterministic rank windows; includes the tail if the stride misses it."""
    if count < 2:
        return []
    if window_size < 2 or stride < 1:
        raise ValueError("tta_window_size must be >=2 and tta_window_stride must be positive.")
    width = min(count, window_size)
    if width == count:
        return [(0, count)]
    starts = list(range(0, count - width + 1, stride))
    tail_start = count - width
    if starts[-1] != tail_start:
        starts.append(tail_start)
    return [(start, start + width) for start in starts]


def _kl_teacher_student(teacher_prob: torch.Tensor, adapted_scores: torch.Tensor, temperature: float) -> torch.Tensor:
    return F.kl_div(
        F.log_softmax(adapted_scores / temperature, dim=-1),
        teacher_prob,
        reduction="batchmean",
    )


def adapt_distribution_one_query(
    entries: Sequence[Dict[str, Any]],
    model_args: object,
    device: torch.device,
) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
    """Fit/discard one FP32 u_q and return a Top-K-constrained order."""
    ordered = [
        entry for _, entry in sorted(
            enumerate(entries),
            key=lambda pair: (-float(pair[1]["z"]), pair[0], str(pair[1]["docid"])),
        )
    ]
    total_count = len(ordered)
    configured_k = int(model_args.tta_candidate_k)
    count = min(configured_k, total_count)
    diagnostics: Dict[str, Any] = {
        "candidate_k": count,
        "configured_candidate_k": configured_k,
        "teacher_confidence": None,
        "tta_applied": False,
        "skip_reason": None,
        "initial_global_kl": None,
        "final_global_kl": None,
        "initial_total_loss": None,
        "final_total_loss": None,
        "actual_steps": 0,
        "accepted_steps": 0,
        "max_abs_residual": 0.0,
        "mean_abs_residual": 0.0,
        "top10_order_changed": False,
        "top10_set_changed": False,
        "teacher_dear_top10_overlap": None,
        "student_grad_present": False,
        "teacher_grad_present": False,
    }

    def original_result(reason: str) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
        diagnostics["skip_reason"] = reason
        return [(str(entry["docid"]), float(entry["z"])) for entry in ordered], diagnostics

    if count < 2:
        return original_result("fewer_than_two_candidates")
    if count < configured_k:
        # The requested dynamic shrinking is safe, but record it explicitly.
        diagnostics["candidate_k"] = count
    top = ordered[:count]
    if any(entry.get("teacher_z") is None for entry in top):
        return original_result("missing_teacher_score")

    z = torch.tensor([entry["z"] for entry in top], dtype=torch.float32, device=device).detach()
    teacher = torch.tensor([entry["teacher_z"] for entry in top], dtype=torch.float32, device=device).detach()
    if not torch.isfinite(z).all() or not torch.isfinite(teacher).all():
        return original_result("nonfinite_input_score")
    if teacher.std(unbiased=False).item() <= EPS:
        return original_result("near_zero_teacher_variance")
    hidden = torch.stack([entry["h"] for entry in top]).to(device=device, dtype=torch.float32).detach()
    hidden = F.normalize(hidden, p=2, dim=-1).detach()
    if not torch.isfinite(hidden).all():
        return original_result("nonfinite_hidden_state")

    teacher_temperature = float(model_args.tta_teacher_temperature)
    student_temperature = float(model_args.tta_student_temperature)
    if teacher_temperature <= 0 or student_temperature <= 0:
        raise ValueError("Distribution TTA temperatures must be positive.")
    z_hat = _zscore(z)
    t_hat = _zscore(teacher)
    teacher_prob = F.softmax(t_hat / teacher_temperature, dim=-1).detach()
    entropy = -(teacher_prob * teacher_prob.clamp_min(EPS).log()).sum()
    confidence = float((1.0 - entropy / math.log(float(count))).clamp(0.0, 1.0).item())
    gate = confidence if model_args.tta_use_confidence_gate else 1.0
    diagnostics["teacher_confidence"] = confidence
    if model_args.tta_use_confidence_gate and confidence < float(model_args.tta_min_teacher_confidence):
        return original_result("teacher_confidence_below_threshold")

    teacher_top10 = {str(top[index]["docid"]) for index in torch.argsort(teacher, descending=True)[:min(10, count)].tolist()}
    dear_top10 = {str(entry["docid"]) for entry in top[:min(10, count)]}
    diagnostics["teacher_dear_top10_overlap"] = len(teacher_top10 & dear_top10) / float(min(10, count))
    windows = build_overlapping_windows(count, int(model_args.tta_window_size), int(model_args.tta_window_stride))
    window_weights = torch.tensor(
        [1.0 / math.log2(float(start + 1) + 2.0) for start, _ in windows],
        dtype=torch.float32,
        device=device,
    )
    window_weights = window_weights / window_weights.sum().clamp_min(EPS)

    # This is intentionally the *only* trainable tensor in the method.
    u_q = torch.zeros(hidden.size(-1), device=device, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.AdamW([u_q], lr=float(model_args.tta_lr), weight_decay=0.0)

    def compute_loss() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bounded = torch.tanh(hidden @ u_q)
        adapted = z_hat + float(gate) * float(model_args.tta_gamma) * bounded
        global_kl = _kl_teacher_student(teacher_prob, adapted, student_temperature)
        local_losses = []
        for start, end in windows:
            local_teacher = F.softmax(t_hat[start:end] / teacher_temperature, dim=-1).detach()
            local_losses.append(_kl_teacher_student(local_teacher, adapted[start:end], student_temperature))
        local_kl = torch.stack(local_losses).mul(window_weights).sum() if local_losses else adapted.new_zeros(())
        total = (
            global_kl
            + float(model_args.tta_lambda_local) * local_kl
            + float(model_args.tta_anchor_weight) * bounded.square().mean()
            + float(model_args.tta_u_norm_weight) * u_q.square().mean()
        )
        return total, global_kl, bounded

    with torch.no_grad():
        initial_total, initial_kl, _ = compute_loss()
    diagnostics["initial_global_kl"] = float(initial_kl.item())
    diagnostics["initial_total_loss"] = float(initial_total.item())
    # Compare losses *after* an optimizer step.  Comparing the pre-step loss
    # (the old implementation did this) makes the first iteration look stale
    # by construction and can stop a nominal 10-step adaptation after only
    # three updates.  Retaining the best query-local vector also makes the
    # inference-time procedure conservative: it never returns a vector whose
    # optimized objective is worse than the zero-residual initialization.
    best_loss = float(initial_total.item())
    best_u = u_q.detach().clone()
    stale_steps = 0
    for step in range(int(model_args.tta_steps)):
        total, _, _ = compute_loss()
        if not torch.isfinite(total):
            return original_result("nonfinite_optimization_loss")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        if u_q.grad is None or not torch.isfinite(u_q.grad).all():
            return original_result("nonfinite_or_missing_u_gradient")
        gradient = u_q.grad.detach().clone()
        before_step_u = u_q.detach().clone()
        optimizer.step()
        diagnostics["actual_steps"] = step + 1
        with torch.no_grad():
            post_step_total, _, _ = compute_loss()
        if not torch.isfinite(post_step_total):
            return original_result("nonfinite_post_step_loss")
        # The KLs can be around 1e-3 after temperature smoothing, whereas an
        # Adam sign-normalized first update at lr=1e-2 is comparatively large.
        # Backtrack only the *query-local* vector, retaining the configured
        # learning rate as an upper bound while enforcing a conservative
        # descent step for the complete trust-region objective.
        if float(post_step_total.item()) > float(total.detach().item()):
            candidate_u = u_q.detach().clone()
            accepted = False
            with torch.no_grad():
                factors = (0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625)
                for factor in factors:
                    u_q.copy_(before_step_u + factor * (candidate_u - before_step_u))
                    trial_total, _, _ = compute_loss()
                    if torch.isfinite(trial_total) and float(trial_total.item()) <= float(total.detach().item()):
                        post_step_total = trial_total
                        accepted = True
                        break
                if not accepted:
                    # Adam's element-wise sign normalization can be poorly
                    # aligned with the small span of 30 normalized document
                    # vectors.  The true negative-gradient direction is a
                    # guaranteed local descent direction; use it as a
                    # conservative fallback rather than silently returning a
                    # practically zero adaptation.
                    grad_norm = gradient.norm().clamp_min(EPS)
                    gradient_candidate = before_step_u - float(model_args.tta_lr) * gradient / grad_norm
                    for factor in factors:
                        u_q.copy_(before_step_u + factor * (gradient_candidate - before_step_u))
                        trial_total, _, _ = compute_loss()
                        if torch.isfinite(trial_total) and float(trial_total.item()) <= float(total.detach().item()):
                            post_step_total = trial_total
                            accepted = True
                            break
                if not accepted:
                    u_q.copy_(before_step_u)
                    post_step_total = total.detach()
        current_loss = float(post_step_total.item())
        if current_loss < float(total.detach().item()):
            diagnostics["accepted_steps"] += 1
        improvement = best_loss - current_loss
        # Preserve any strictly better iterate.  ``tolerance`` decides only
        # whether progress is large enough to reset patience; it must not
        # discard a valid small improvement merely because the objective is a
        # small KL value (which is common with temperature 2.0).
        if improvement > 0.0:
            best_loss = current_loss
            best_u.copy_(u_q.detach())
        if improvement > float(model_args.tta_early_stop_tolerance):
            stale_steps = 0
        else:
            stale_steps += 1
        if (
            int(model_args.tta_early_stop_patience) > 0
            and stale_steps >= int(model_args.tta_early_stop_patience)
        ):
            break

    with torch.no_grad():
        u_q.copy_(best_u)
        final_total, final_kl, bounded = compute_loss()
        if not torch.isfinite(final_total) or not torch.isfinite(bounded).all():
            return original_result("nonfinite_final_residual")
        adapted = z_hat + float(gate) * float(model_args.tta_gamma) * bounded
    diagnostics["final_global_kl"] = float(final_kl.item())
    diagnostics["final_total_loss"] = float(final_total.item())
    actual_residual = float(gate) * float(model_args.tta_gamma) * bounded
    diagnostics["max_abs_residual"] = float(actual_residual.abs().max().item())
    diagnostics["mean_abs_residual"] = float(actual_residual.abs().mean().item())
    diagnostics["student_grad_present"] = False
    diagnostics["teacher_grad_present"] = False
    adapted_indices = sorted(range(count), key=lambda index: (-float(adapted[index]), index, str(top[index]["docid"])))
    adapted_docids = [str(top[index]["docid"]) for index in adapted_indices]
    diagnostics["top10_order_changed"] = adapted_docids[:10] != [str(entry["docid"]) for entry in top[:10]]
    diagnostics["top10_set_changed"] = set(adapted_docids[:10]) != dear_top10
    diagnostics["tta_applied"] = True
    results = [(str(top[index]["docid"]), float(adapted[index])) for index in adapted_indices]
    # Top-K-only serialization: outside documents cannot cross this boundary.
    results.extend((str(entry["docid"]), float(entry["z"])) for entry in ordered[count:])
    del optimizer, u_q
    return results, diagnostics
