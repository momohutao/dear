#reranker_inference.py
import logging
import json
import os
import sys
import gc
from contextlib import nullcontext

from tqdm import tqdm

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import (
    HfArgumentParser,
)


from arguments import ModelArguments, DataArguments, \
    ModelTrainingArguments as TrainingArguments

from data import HFRerankerInferenceDataset as HFRerankDataset, RerankerInferenceDataset, RerankerInferenceCollator
from modeling import DistillRerankerModel
from setwise_judge import build_setwise_judge
from setwise_head import load_setwise_head
from setwise_residual import rerank_with_contextual_residual
from distribution_tta import adapt_distribution_one_query
from entropy_hard_gate_tta import adapt_entropy_hard_gate_one_query
from teacher_score_cache import (
    append_teacher_score_cache,
    load_teacher_score_cache,
    teacher_identifier,
)
from ts_setrank import TSSetRanker
import time

start_time = time.time()
logger = logging.getLogger(__name__)


def _pad_teacher_rows(rows, pad_token_id):
    """Right-pad cached tokenizer tensors without changing token alignment."""
    keys = rows[0]["teacher_inputs"].keys()
    batch = {}
    for key in keys:
        tensors = [row["teacher_inputs"][key] for row in rows]
        maximum = max(tensor.numel() for tensor in tensors)
        pad_value = pad_token_id if key == "input_ids" else 0
        batch[key] = torch.stack([
            F.pad(tensor, (0, maximum - tensor.numel()), value=pad_value)
            for tensor in tensors
        ])
    return batch


def populate_distribution_teacher_scores(all_results, model_args, teacher_pad_token_id, device):
    """Attach cached RankLLaMA scores by qid/docid/checkpoint/candidate-K."""
    teacher_id = teacher_identifier(model_args.teacher_model_name_or_path)
    candidate_k = int(model_args.tta_candidate_k)
    cached = load_teacher_score_cache(
        model_args.tta_teacher_cache_path, teacher_id, candidate_k
    )
    selected, missing, cache_hits = [], [], 0
    seen_missing = set()
    for qid, entries in all_results.items():
        initial = sorted(entries, key=lambda entry: -float(entry["z"]))
        for entry in initial[:min(candidate_k, len(initial))]:
            key = (str(qid), str(entry["docid"]), teacher_id, candidate_k)
            if key in cached:
                entry["teacher_z"] = cached[key]
                cache_hits += 1
            elif key not in seen_missing:
                missing.append((str(qid), entry, key))
                seen_missing.add(key)
            else:
                # Duplicate docids deliberately reuse the one score but retain
                # their independent candidate occurrences/order.
                entry["teacher_z"] = None
            selected.append((str(qid), entry, key))

    cache_misses = len(missing)
    new_records = []
    if missing:
        teacher = AutoModelForSequenceClassification.from_pretrained(
            model_args.teacher_model_name_or_path,
            num_labels=1,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        ).to(device).eval()
        teacher.config.pad_token_id = int(teacher_pad_token_id)
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        score_by_key = {}
        batch_size = 8
        with torch.no_grad():
            for start in range(0, len(missing), batch_size):
                chunk = missing[start:start + batch_size]
                batch = _pad_teacher_rows([entry for _, entry, _ in chunk], teacher_pad_token_id)
                batch = {key: value.to(device) for key, value in batch.items()}
                logits = teacher(**batch, return_dict=True).logits.squeeze(-1).float().cpu().tolist()
                for (qid, entry, key), score in zip(chunk, logits):
                    score_by_key[key] = float(score)
                    entry["teacher_z"] = float(score)
                    new_records.append({"qid": qid, "docid": entry["docid"], "score": float(score)})
        if any(parameter.grad is not None for parameter in teacher.parameters()):
            raise RuntimeError("Frozen RankLLaMA unexpectedly accumulated gradients.")
        del teacher
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for qid, entry, key in selected:
            if entry.get("teacher_z") is None:
                entry["teacher_z"] = score_by_key[key]
    for _, entries in all_results.items():
        for entry in entries:
            entry.pop("teacher_inputs", None)
    written = append_teacher_score_cache(
        model_args.tta_teacher_cache_path, teacher_id, candidate_k, new_records
    )
    return {
        "teacher_id": teacher_id,
        "candidate_k": candidate_k,
        "cache_path": model_args.tta_teacher_cache_path,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_records_written": written,
    }


def adapt_one_query(entries, model_args, device):
    """Fit a fresh low-rank residual scorer for exactly one query.

    DeAR representations and logits in ``entries`` are detached.  Therefore
    gradients below can only update the temporary A and B matrices.
    """
    entries = sorted(entries, key=lambda item: item["z"], reverse=True)
    z = torch.tensor([item["z"] for item in entries], device=device)
    h = torch.stack([item["h"] for item in entries]).to(device).float()

    eps = 1e-8
    z_hat = (z - z.mean()) / (z.std(unbiased=False) + eps)
    candidate_count = min(model_args.tta_top_k, len(entries))
    required_count = model_args.tta_n_pos + model_args.tta_n_neg
    if candidate_count < required_count:
        logger.warning(
            "Skipping TTA for a query with only %d candidates; need at least %d.",
            candidate_count,
            required_count,
        )
        return [(item["docid"], z_hat[index].item()) for index, item in enumerate(entries)]

    initial_order = torch.argsort(z, descending=True)
    selected = initial_order[:candidate_count]
    pos_idx = selected[:model_args.tta_n_pos]
    neg_idx = selected[-model_args.tta_n_neg:]
    correction_pairs = None
    if model_args.tta_correction_mode:
        if not model_args.tta_use_teacher:
            raise ValueError("TTA correction mode requires --tta_use_teacher true.")
        teacher_z = torch.tensor(
            [item["teacher_z"] for item in entries], device=device
        )
        correction_top_k = min(model_args.tta_correction_top_k, len(entries))
        if correction_top_k < 2:
            return [
                (item["docid"], z_hat[index].item())
                for index, item in enumerate(entries)
            ]

        # Teacher top-5 acts as a high-confidence anchor set; DeAR ranks 6--30
        # are the boundary candidates that may need correction.
        teacher_order = torch.argsort(teacher_z, descending=True)
        teacher_positive = teacher_order[:model_args.tta_teacher_top_k]
        dear_boundary = initial_order[5:correction_top_k]
        correction_candidates = torch.unique(
            torch.cat((teacher_positive, dear_boundary)), sorted=True
        )
        cz = z[correction_candidates]
        ct = teacher_z[correction_candidates]
        left, right = torch.triu_indices(
            correction_candidates.numel(),
            correction_candidates.numel(),
            offset=1,
            device=device,
        )
        dear_direction = torch.sign(cz[left] - cz[right])
        teacher_direction = torch.sign(ct[left] - ct[right])
        teacher_gap = (ct[left] - ct[right]).abs()
        disagreement = (
            dear_direction.ne(teacher_direction)
            & teacher_direction.ne(0)
            & teacher_gap.gt(model_args.tta_teacher_delta)
        )
        if not disagreement.any():
            logger.info("Skipping TTA: no confident teacher/DeAR correction pairs.")
            return [
                (item["docid"], z_hat[index].item())
                for index, item in enumerate(entries)
            ]
        correction_pairs = (
            correction_candidates[left[disagreement]],
            correction_candidates[right[disagreement]],
            teacher_direction[disagreement],
            torch.clamp(
                teacher_gap[disagreement]
                / model_args.tta_teacher_confidence_scale,
                max=1.0,
            ),
        )
    elif model_args.tta_use_teacher:
        teacher_z = torch.tensor(
            [item["teacher_z"] for item in entries], device=device
        )
        # Teacher ranks only the same top-K candidate set used for pseudo labels.
        teacher_order = selected[
            torch.argsort(teacher_z[selected], descending=True)
        ]
        teacher_top = teacher_order[:min(10, candidate_count)]
        teacher_bottom = teacher_order[-model_args.tta_n_neg:]
        pos_idx = pos_idx[torch.isin(pos_idx, teacher_top)]
        neg_idx = neg_idx[torch.isin(neg_idx, teacher_bottom)]
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            logger.info(
                "Skipping TTA: no teacher-agreed pseudo-positive or pseudo-negative."
            )
            return [
                (item["docid"], z_hat[index].item())
                for index, item in enumerate(entries)
            ]
    if correction_pairs is None:
        temperature = model_args.tta_temperature
        w_pos = F.softmax(z_hat[pos_idx] / temperature, dim=0)
        w_neg = F.softmax(-z_hat[neg_idx] / temperature, dim=0)

    # The residual scale is assigned from the *initial* DeAR rank.  This
    # preserves the most important head documents while still permitting
    # query-level adaptation below the protected boundary.
    per_doc_gamma = torch.full_like(z_hat, model_args.tta_gamma)
    if model_args.tta_protect_top_k > 0:
        protected = initial_order[:model_args.tta_protect_top_k]
        per_doc_gamma[protected] = 0.0
    if model_args.tta_top10_gamma >= 0:
        top10_start = model_args.tta_protect_top_k
        top10 = initial_order[top10_start:10]
        per_doc_gamma[top10] = model_args.tta_top10_gamma

    # A direct query-level direction.  u=0 makes the initial residual exactly
    # zero, but d(h^T u)/du=h, so it receives a non-zero first-step gradient.
    u = torch.zeros(h.size(-1), device=device, requires_grad=True)
    optimizer = torch.optim.AdamW(
        [u], lr=model_args.tta_learning_rate, weight_decay=0.0
    )

    for _ in range(model_args.tta_steps):
        raw_residual = h @ u
        # A bounded residual cannot overwhelm the normalized DeAR score.
        residual = per_doc_gamma * torch.tanh(raw_residual)
        adapted_scores = z_hat + residual
        if correction_pairs is not None:
            left_idx, right_idx, targets, pair_weights = correction_pairs
            margins = adapted_scores[left_idx] - adapted_scores[right_idx]
            pair_loss = (
                pair_weights * F.softplus(-targets * margins)
            ).sum() / pair_weights.sum().clamp_min(1e-8)
        else:
            margins = (
                adapted_scores[pos_idx].unsqueeze(1)
                - adapted_scores[neg_idx].unsqueeze(0)
            )
            pair_weights = w_pos.unsqueeze(1) * w_neg.unsqueeze(0)
            pair_loss = (pair_weights * F.softplus(-margins)).sum()
        anchor_loss = residual.square().mean()
        loss = pair_loss + model_args.tta_anchor_weight * anchor_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        raw_residual = h @ u
        residual = per_doc_gamma * torch.tanh(raw_residual)
        adapted_scores = z_hat + residual
    return [
        (item["docid"], adapted_scores[index].item())
        for index, item in enumerate(entries)
    ]


def _normalized_rank_disagreement(scores_a, scores_b):
    """Mean absolute rank displacement in [0, 1] for one candidate list."""
    count = scores_a.numel()
    if count < 2:
        return scores_a.new_zeros(())
    rank_a = torch.empty(count, device=scores_a.device, dtype=torch.float)
    rank_b = torch.empty_like(rank_a)
    rank_a[torch.argsort(scores_a, descending=True)] = torch.arange(
        count, device=scores_a.device, dtype=torch.float
    )
    rank_b[torch.argsort(scores_b, descending=True)] = torch.arange(
        count, device=scores_a.device, dtype=torch.float
    )
    return (rank_a - rank_b).abs().mean() / float(count - 1)


def adapt_one_query_listwise_reward(entries, model_args, device):
    """TOUR-style continuous-reward adaptation for one query.

    The initial DeAR top-M set is the *only* set that may be reordered.  The
    RankLLaMA reward becomes a soft teacher distribution, while the original
    DeAR distribution is an explicit KL trust region.  No DeAR or teacher
    parameter receives a gradient: only the fresh vector ``u`` is optimized.

    The returned sequence is already the final rank order.  This matters
    because scores outside top-M must never be able to enter the adapted set.
    """
    if not model_args.tta_use_teacher:
        raise ValueError(
            "Continuous list-reward TTA requires --tta_use_teacher true."
        )
    if model_args.tta_teacher_temperature <= 0:
        raise ValueError("tta_teacher_temperature must be positive.")
    if model_args.tta_student_temperature <= 0:
        raise ValueError("tta_student_temperature must be positive.")
    if model_args.tta_listwise_top_k < 2:
        raise ValueError("tta_listwise_top_k must be at least 2.")

    # Keep the complete initial ranking so candidates outside M retain their
    # exact DeAR order.  We adapt only the first M entries below.
    initial_entries = sorted(entries, key=lambda item: item["z"], reverse=True)
    count = min(model_args.tta_listwise_top_k, len(initial_entries))
    if count < 2:
        return [(item["docid"], float(item["z"])) for item in initial_entries]

    top_entries = initial_entries[:count]
    z = torch.tensor([item["z"] for item in top_entries], device=device)
    teacher_z = torch.tensor(
        [item["teacher_z"] for item in top_entries], device=device
    )
    h = torch.stack([item["h"] for item in top_entries]).to(device).float()
    # classifier_embeddings are L2-normalized when cached.  Re-normalize in
    # fp32 after CPU fp16 storage to preserve the stated h-hat invariant.
    h = F.normalize(h, p=2, dim=-1)

    eps = 1e-8
    z_hat = (z - z.mean()) / (z.std(unbiased=False) + eps)
    teacher_hat = (teacher_z - teacher_z.mean()) / (
        teacher_z.std(unbiased=False) + eps
    )
    p_teacher = F.softmax(
        teacher_hat / model_args.tta_teacher_temperature, dim=0
    )
    p_initial = F.softmax(
        z_hat / model_args.tta_student_temperature, dim=0
    )

    normalized_entropy = -(
        p_teacher * p_teacher.clamp_min(eps).log()
    ).sum() / torch.log(torch.tensor(float(count), device=device))
    disagreement = _normalized_rank_disagreement(z_hat, teacher_hat)
    if normalized_entropy.item() > model_args.tta_teacher_entropy_max:
        logger.info(
            "Skipping list-reward TTA: teacher entropy %.4f exceeds %.4f.",
            normalized_entropy.item(), model_args.tta_teacher_entropy_max,
        )
        return [
            (item["docid"], float(item["z"]))
            for item in initial_entries
        ]
    if disagreement.item() < model_args.tta_min_rank_disagreement:
        logger.info(
            "Skipping list-reward TTA: rank disagreement %.4f is below %.4f.",
            disagreement.item(), model_args.tta_min_rank_disagreement,
        )
        return [
            (item["docid"], float(item["z"]))
            for item in initial_entries
        ]

    # Optional top-rank discount is applied only to the teacher target.  The
    # trust-region term must retain the original DeAR distribution p0 from the
    # stated objective; discounting p0 would weaken protection of its initial
    # top ranks.  Renormalizing p_teacher keeps the teacher term a valid KL.
    if model_args.tta_position_discount:
        initial_rank = torch.arange(1, count + 1, device=device, dtype=torch.float)
        discount = 1.0 / torch.log2(initial_rank + 1.0)
        p_teacher = p_teacher * discount
        p_teacher = p_teacher / p_teacher.sum().clamp_min(eps)

    u = torch.zeros(h.size(-1), device=device, requires_grad=True)
    optimizer = torch.optim.AdamW(
        [u], lr=model_args.tta_listwise_learning_rate, weight_decay=0.0
    )
    for _ in range(model_args.tta_listwise_steps):
        adapted_scores = z_hat + h @ u
        log_p_student = F.log_softmax(
            adapted_scores / model_args.tta_student_temperature, dim=0
        )
        list_loss = F.kl_div(log_p_student, p_teacher, reduction="sum")
        trust_loss = F.kl_div(log_p_student, p_initial, reduction="sum")
        l2_loss = u.square().sum()
        loss = (
            list_loss
            + model_args.tta_trust_weight * trust_loss
            + model_args.tta_u_l2_weight * l2_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        adapted_scores = z_hat + h @ u
        adapted_order = torch.argsort(adapted_scores, descending=True)

    logger.info(
        "Applied list-reward TTA: M=%d entropy=%.4f disagreement=%.4f "
        "list_KL=%.4f trust_KL=%.4f ||u||=%.4f.",
        count,
        normalized_entropy.item(),
        disagreement.item(),
        list_loss.item(),
        trust_loss.item(),
        u.norm().item(),
    )

    # Return the adapted top-M order followed by untouched candidates.  The
    # caller deliberately does not sort this result again in listwise mode.
    ordered_results = [
        (
            top_entries[index]["docid"],
            adapted_scores[index].item(),
        )
        for index in adapted_order.tolist()
    ]
    ordered_results.extend(
        (item["docid"], float(item["z"]))
        for item in initial_entries[count:]
    )
    return ordered_results

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if training_args.local_rank > 0 or training_args.n_gpu > 1:
        raise NotImplementedError('Multi-GPU encoding is not supported.')
    enabled_postprocessors = sum((
        bool(model_args.tta_enabled),
        bool(model_args.use_distribution_tta),
        bool(model_args.use_entropy_hard_gate_tta),
        bool(model_args.use_ts_setrank),
        bool(model_args.setwise_residual_enabled),
    ))
    if enabled_postprocessors > 1:
        raise ValueError(
            "Legacy TTA, distribution TTA, entropy-hard-gate TTA, TS-SetRank, and the continuous setwise residual are "
            "independent post-processors; enable only one."
        )
    if model_args.setwise_residual_enabled:
        if model_args.setwise_residual_top_k % model_args.set_size:
            raise ValueError("setwise_residual_top_k must be divisible by set_size.")
        if not 1 <= model_args.setwise_residual_context_rounds <= 3:
            raise ValueError("setwise_residual_context_rounds must be in [1, 3].")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        (
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.model_name_or_path
        ),
        cache_dir=model_args.cache_dir,
    )

    if (
        tokenizer.name_or_path.lower().startswith("qwen")
        and tokenizer.bos_token_id is None
    ):
        tokenizer.bos_token = "<|endoftext|>"
        tokenizer.bos_token_id = (
            tokenizer.convert_tokens_to_ids(
                "<|endoftext|>"
            )
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    # 必须在加载模型前先获得教师 tokenizer 的 PAD ID
    tokenizer_teacher = AutoTokenizer.from_pretrained(
        model_args.teacher_model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    if tokenizer_teacher.pad_token is None:
        tokenizer_teacher.pad_token = (
            tokenizer_teacher.eos_token
        )

    tokenizer_teacher.padding_side = "right"

    if (
        tokenizer_teacher.name_or_path
        .lower()
        .startswith("qwen")
        and tokenizer_teacher.bos_token_id is None
    ):
        tokenizer_teacher.bos_token = "<|endoftext|>"
        tokenizer_teacher.bos_token_id = (
            tokenizer_teacher.convert_tokens_to_ids(
                "<|endoftext|>"
            )
        )

    # The teacher can either filter legacy TTA pseudo-labels or be evaluated alone.
    # Distribution TTA scores only its original top-K later, after DeAR has
    # produced the initial ranking and the student has been released.
    # It is never added to the student score.
    use_teacher = model_args.tta_use_teacher or model_args.teacher_only

    model = DistillRerankerModel.load(
        student_model_path=model_args.model_name_or_path,
        teacher_model_path=(
            model_args.teacher_model_name_or_path
        ),
        student_pad_token_id=tokenizer.pad_token_id,
        teacher_pad_token_id=(
            tokenizer_teacher.pad_token_id
        ),
        use_teacher=use_teacher,
        cache_dir=model_args.cache_dir,
        torch_dtype=(
            torch.float16
            if training_args.fp16
            else torch.float32
        ),
        loss_type=model_args.loss_type,
        model_args=model_args,
    )
    if model_args.use_distribution_tta or model_args.use_entropy_hard_gate_tta:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None

    if (
        model.student.config.pad_token_id
        != tokenizer.pad_token_id
    ):
        raise ValueError(
            "Student model/tokenizer pad_token_id "
            "mismatch during inference: "
            f"model={model.student.config.pad_token_id}, "
            f"tokenizer={tokenizer.pad_token_id}"
        )

    if (
        use_teacher
        and model.teacher.config.pad_token_id
        != tokenizer_teacher.pad_token_id
    ):
        raise ValueError(
            "Teacher model/tokenizer pad_token_id mismatch during inference: "
            f"model={model.teacher.config.pad_token_id}, "
            f"tokenizer={tokenizer_teacher.pad_token_id}"
        )

    logger.info(
        "Inference PAD alignment: student=%s",
        tokenizer.pad_token_id,
    )
    rerank_dataset = HFRerankDataset(tokenizer_student=tokenizer, tokenizer_teacher=tokenizer_teacher ,data_args=data_args, cache_dir=data_args.data_cache_dir or model_args.cache_dir)

    rerank_dataset = RerankerInferenceDataset(
            rerank_dataset.process(data_args.encode_num_shard, data_args.encode_shard_index),
            tokenizer, tokenizer_teacher, max_q_len=data_args.q_max_len, max_p_len=data_args.p_max_len
        )

    rerank_loader = DataLoader(
        rerank_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=RerankerInferenceCollator(
            tokenizer, tokenizer_teacher
            #max_length=data_args.q_max_len+data_args.p_max_len,
            #padding='max_length'
        ),
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )
    model = model.to(training_args.device)
    model.eval()
    # The setwise judge is intentionally not loaded yet.  DeAR must first
    # score every candidate and place only CPU-resident raw scores/text in
    # ``all_results``; then its GPU weights are released before the judge is
    # loaded.  A disabled TS branch still has no judge or extra work.
    ts_reranker = None
    contextual_residual_head = None
    all_results = {}
    start_time = time.time()  # Start timing inference
    for (
        batch_query_ids,
        batch_text_ids,
        batch,
        teacher,
        student_region_ids,
        batch_raw_queries,
        batch_raw_documents,
    ) in tqdm(rerank_loader):
        with torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
            with torch.no_grad():
                for k, v in batch.items():
                    batch[k] = v.to(training_args.device)
                if use_teacher:
                    for k, v in teacher.items():
                        teacher[k] = v.to(training_args.device)
                student_region_ids = student_region_ids.to(training_args.device)
                model_output = model(
                    batch,
                    teacher if use_teacher else None,
                    student_region_ids=student_region_ids,
                )
                scores = model_output.scores.cpu().detach().numpy()
                teacher_scores = None
                if use_teacher:
                    teacher_scores = model_output.teacher_scores.cpu().detach().numpy()
                if model_args.teacher_only:
                    scores = teacher_scores
                for i in range(len(scores)):
                    qid = batch_query_ids[i]
                    docid = batch_text_ids[i]
                    if qid not in all_results:
                        all_results[qid] = []
                    if model_args.tta_enabled:
                        # fp16 CPU storage keeps the complete run practical;
                        # vectors are restored to fp32 before optimization.
                        all_results[qid].append({
                            "docid": docid,
                            "z": float(scores[i][0]),
                            "h": model_output.classifier_embeddings[i].detach().cpu().half(),
                            "teacher_z": float(teacher_scores[i][0]) if use_teacher else None,
                        })
                    elif model_args.use_distribution_tta or model_args.use_entropy_hard_gate_tta:
                        if model_output.setwise_embeddings is None:
                            raise RuntimeError(
                                "Distribution TTA requires classifier-position hidden states."
                            )
                        all_results[qid].append({
                            "docid": docid,
                            "z": float(scores[i][0]),
                            "h": model_output.setwise_embeddings[i].detach().cpu().float(),
                            # Preserve tokenizer-produced teacher inputs on CPU;
                            # RankLLaMA is run only after DeAR Top-K is known.
                            "teacher_inputs": {
                                key: value[i].detach().cpu().clone()
                                for key, value in teacher.items()
                            },
                        })
                    elif model_args.use_ts_setrank:
                        entry = {
                            "docid": docid,
                            "z": float(scores[i][0]),
                            "query": batch_raw_queries[i],
                            "document": batch_raw_documents[i],
                        }
                        if model_args.ts_judge_type == "contextual":
                            if model_output.setwise_embeddings is None:
                                raise RuntimeError(
                                    "Contextual TS judge requires cached DeAR classifier embeddings."
                                )
                            entry["h"] = (
                                model_output.setwise_embeddings[i].detach().cpu().half()
                            )
                        all_results[qid].append(entry)
                    elif model_args.setwise_residual_enabled:
                        if model_output.setwise_embeddings is None:
                            raise RuntimeError(
                                "Continuous setwise residual requires classifier-position embeddings."
                            )
                        all_results[qid].append({
                            "docid": docid,
                            "z": float(scores[i][0]),
                            # The SLL head was trained on this raw classifier
                            # representation, so do not L2-normalize it here.
                            "h": model_output.setwise_embeddings[i].detach().cpu().half(),
                        })
                    else:
                        all_results[qid].append((docid, float(scores[i][0])))
    end_time = time.time()  # End timing inference
    print(f"Total question processing time (inference only): {end_time - start_time:.2f} seconds")

    if (
        model_args.use_ts_setrank
        or model_args.setwise_residual_enabled
        or model_args.use_distribution_tta
        or model_args.use_entropy_hard_gate_tta
    ):
        # all_results now contains plain Python floats/strings only.  Release
        # the entire pointwise model before allocating the independent
        # instruction judge; TS posterior/fusion operations themselves are
        # CPU-only in ts_setrank.py.
        # The last batch/output can still hold small CUDA tensors; dropping
        # them makes the phase boundary explicit and measurable.
        if "model_output" in locals():
            del model_output
        if "batch" in locals():
            del batch
        if "teacher" in locals():
            del teacher
        if "student_region_ids" in locals():
            del student_region_ids
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if model_args.use_distribution_tta or model_args.use_entropy_hard_gate_tta:
            logger.info("Released frozen DeAR+KK+SLL pointwise model; loading RankLLaMA only for original Top-%d.", model_args.tta_candidate_k)
            distribution_teacher_cache_stats = populate_distribution_teacher_scores(
                all_results,
                model_args,
                tokenizer_teacher.pad_token_id,
                training_args.device,
            )
        elif model_args.use_ts_setrank:
            logger.info(
                "Released frozen DeAR pointwise model; loading independent setwise judge: %s",
                model_args.setwise_model_name_or_path,
            )
            ts_reranker = TSSetRanker(model_args, build_setwise_judge(model_args))
        else:
            logger.info(
                "Released frozen DeAR pointwise model; running CPU continuous setwise residual."
            )
            contextual_residual_head = load_setwise_head(
                model_args.model_name_or_path, map_location="cpu"
            ).float().eval()

    trace_records = []
    with open(data_args.encoded_save_path, 'w') as f:
        for query_index, qid in enumerate(all_results):
            entropy_selected_branch = None
            if model_args.tta_enabled:
                if model_args.tta_listwise_reward_mode:
                    results = adapt_one_query_listwise_reward(
                        all_results[qid], model_args, training_args.device
                    )
                else:
                    results = adapt_one_query(
                        all_results[qid], model_args, training_args.device
                    )
            elif model_args.use_distribution_tta:
                results, trace = adapt_distribution_one_query(
                    all_results[qid], model_args, training_args.device
                )
                trace["qid"] = str(qid)
                trace_records.append(trace)
            elif model_args.use_entropy_hard_gate_tta:
                results, trace = adapt_entropy_hard_gate_one_query(
                    all_results[qid], model_args, training_args.device
                )
                trace["qid"] = str(qid)
                trace_records.append(trace)
                entropy_selected_branch = trace["selected_branch"]
            elif model_args.use_ts_setrank:
                entries = all_results[qid]
                query_texts = {entry["query"] for entry in entries}
                if len(query_texts) != 1:
                    raise ValueError(f"TS-SetRank query text mismatch for query_id={qid}.")
                if model_args.ts_judge_type == "contextual":
                    ts_reranker.judge.set_query_embeddings({
                        entry["docid"]: entry["h"] for entry in entries
                    })
                results, trace = ts_reranker.rerank(
                    qid,
                    entries[0]["query"],
                    entries,
                    query_index,
                )
                trace_records.append(trace)
            elif model_args.setwise_residual_enabled:
                results, trace = rerank_with_contextual_residual(
                    all_results[qid],
                    model_args.model_name_or_path,
                    top_k=model_args.setwise_residual_top_k,
                    set_size=model_args.set_size,
                    rounds=model_args.setwise_residual_context_rounds,
                    gamma=model_args.setwise_residual_gamma,
                    head=contextual_residual_head,
                )
                trace["query_id"] = str(qid)
                trace_records.append(trace)
            else:
                results = all_results[qid]
            # List-reward mode has already constructed the final order: its
            # top-M-only constraint would be violated by a global score sort.
            if model_args.use_entropy_hard_gate_tta:
                # Exact hard routing: this branch emits genuine s0 or genuine
                # s1 scores on their shared z-hat scale.  Do not substitute
                # synthetic monotonic TREC ranks, which would alter tie
                # behavior and create a false apparent TTA gain.
                results = sorted(results, key=lambda item: item[1], reverse=True)
            elif (
                model_args.use_ts_setrank or model_args.setwise_residual_enabled
                or model_args.use_distribution_tta or model_args.use_entropy_hard_gate_tta
            ):
                # TSSetRanker already returns the constrained final order:
                # adapted top-K first, then untouched DeAR outsiders.
                pass
            elif not (
                model_args.tta_enabled
                and model_args.tta_listwise_reward_mode
            ):
                results = sorted(results, key=lambda x: x[1], reverse=True)
            else:
                # TREC evaluators rank by the score column, not by physical
                # line order or the rank field emitted by our converter.  The
                # listwise routine has already built the only valid order
                # (adapted top-M followed by untouched candidates), so emit a
                # strictly decreasing serialization score to make evaluation
                # obey that order exactly.  Keeping heterogeneous raw/adapted
                # logits here would accidentally let rank M+1 enter top-M.
                results = [
                    (docid, float(len(results) - rank))
                    for rank, (docid, _) in enumerate(results)
                ]
            if (
                model_args.use_ts_setrank or model_args.setwise_residual_enabled
                or model_args.use_distribution_tta
            ):
                # TS-SetRank constructed a constrained top-K order.  TREC
                # evaluates the numeric score, not file order, so serialize a
                # strict descending score to prevent outsiders re-entering K.
                results = [
                    (docid, float(len(results) - rank))
                    for rank, (docid, _) in enumerate(results)
                ]
            for docid, score in results:
                f.write(f'{qid}\t{docid}\t{score}\n')
    should_write_trace = (
        model_args.use_ts_setrank
        or model_args.setwise_residual_enabled
        or (model_args.use_distribution_tta and model_args.tta_log_diagnostics)
        or (model_args.use_entropy_hard_gate_tta and model_args.tta_log_diagnostics)
    )
    if should_write_trace:
        trace_path = (
            model_args.ts_trace_path
            if model_args.use_ts_setrank
            else (
                model_args.setwise_residual_trace_path
                if model_args.setwise_residual_enabled
                else None
            )
        ) or (
            data_args.encoded_save_path
            + (
                ".ts_setrank_trace.jsonl"
                if model_args.use_ts_setrank
                else (
                    ".setwise_residual_trace.jsonl"
                    if model_args.setwise_residual_enabled
                    else (
                        ".entropy_hard_gate_tta_diagnostics.jsonl"
                        if model_args.use_entropy_hard_gate_tta
                        else ".distribution_tta_diagnostics.jsonl"
                    )
                )
            )
        )
        with open(trace_path, "w", encoding="utf-8") as trace_file:
            for record in trace_records:
                trace_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Wrote contextual reranking trace to %s", trace_path)
        if model_args.use_distribution_tta:
            summary_path = trace_path.replace(".jsonl", ".summary.json")
            applied = [record for record in trace_records if record["tta_applied"]]
            finite = lambda key: [record[key] for record in applied if record[key] is not None]
            summary = {
                "queries": len(trace_records),
                "adapted_queries": len(applied),
                "skipped_queries": len(trace_records) - len(applied),
                "average_teacher_confidence": sum(finite("teacher_confidence")) / max(1, len(finite("teacher_confidence"))),
                "average_initial_global_kl": sum(finite("initial_global_kl")) / max(1, len(finite("initial_global_kl"))),
                "average_final_global_kl": sum(finite("final_global_kl")) / max(1, len(finite("final_global_kl"))),
                "top10_order_changed_queries": sum(record["top10_order_changed"] for record in applied),
                "top10_set_changed_queries": sum(record["top10_set_changed"] for record in applied),
                "average_max_abs_residual": sum(finite("max_abs_residual")) / max(1, len(finite("max_abs_residual"))),
                "nonfinite_fallbacks": sum("nonfinite" in str(record["skip_reason"]) for record in trace_records),
                "teacher_cache": distribution_teacher_cache_stats,
            }
            with open(summary_path, "w", encoding="utf-8") as summary_file:
                json.dump(summary, summary_file, ensure_ascii=False, indent=2, sort_keys=True)
            logger.info("Wrote distribution-TTA summary to %s", summary_path)
        elif model_args.use_entropy_hard_gate_tta:
            summary_path = trace_path.replace(".jsonl", ".summary.json")
            applied = [record for record in trace_records if record["selected_branch"] == "adapted"]
            finite = lambda key: [record[key] for record in trace_records if record.get(key) is not None]
            summary = {
                "queries": len(trace_records),
                "base_branch_queries": len(trace_records) - len(applied),
                "adapted_branch_queries": len(applied),
                "numeric_fallbacks": sum("nonfinite" in str(record["skip_reason"]) for record in trace_records),
                "average_delta_entropy_a": sum(finite("delta_entropy_a")) / max(1, len(finite("delta_entropy_a"))),
                "average_delta_entropy_v": sum(finite("delta_entropy_v")) / max(1, len(finite("delta_entropy_v"))),
                "average_mean_abs_residual": sum(finite("mean_abs_residual")) / max(1, len(finite("mean_abs_residual"))),
                "top1_changed_queries": sum(record["top1_changed"] for record in applied),
                "top10_order_changed_queries": sum(record["top10_order_changed"] for record in applied),
                "top10_set_changed_queries": sum(record["top10_set_changed"] for record in applied),
                "teacher_cache": distribution_teacher_cache_stats,
            }
            with open(summary_path, "w", encoding="utf-8") as summary_file:
                json.dump(summary, summary_file, ensure_ascii=False, indent=2, sort_keys=True)
            logger.info("Wrote entropy-hard-gate TTA summary to %s", summary_path)

if __name__ == "__main__":
    main()
