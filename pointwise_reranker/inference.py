#reranker_inference.py
import logging
import os
import re
import sys
from contextlib import nullcontext
from dataclasses import dataclass, field

from tqdm import tqdm

import torch

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import (
    HfArgumentParser,
)


from arguments import ModelArguments, DataArguments, \
    ModelTrainingArguments as TrainingArguments

from data import HFRerankerInferenceDataset as HFRerankDataset, RerankerInferenceDataset, RerankerInferenceCollator
from modeling import DistillRerankerModel
import time

start_time = time.time()
logger = logging.getLogger(__name__)


@dataclass
class TTAArguments:
    """Inference-only test-time adaptation options."""

    tta: bool = field(default=False)
    tta_lr: float = field(default=1e-5)
    tta_temperature: float = field(default=1.0)
    tta_max_delta: float = field(default=0.01)
    # Validation aid; -1 means process the complete input.
    tta_num_queries: int = field(default=-1)
    # Optional compatibility switch for reproducing the checkpoint's
    # historical PAD/EOS score-position evaluation. Disabled by default.
    tta_use_tokenizer_pad: bool = field(default=False)


def _setup_tta(model):
    """Freeze the model and expose only the final block's RMSNorm scales."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    candidates = []
    for name, parameter in model.student.named_parameters():
        match = re.search(
            r"layers\.(\d+)\.(input_layernorm|post_attention_layernorm)\.weight$",
            name,
        )
        if match:
            candidates.append((int(match.group(1)), name, parameter))

    if not candidates:
        raise RuntimeError(
            "TTA could not find input_layernorm/post_attention_layernorm weights."
        )

    final_layer = max(layer for layer, _, _ in candidates)
    tta_parameters = [
        (name, parameter)
        for layer, name, parameter in candidates
        if layer == final_layer
    ]
    for _, parameter in tta_parameters:
        parameter.requires_grad_(True)

    original = {
        name: parameter.detach().clone()
        for name, parameter in tta_parameters
    }
    logger.info(
        "TTA enabled: final layer=%d, trainable RMSNorm weights=%s",
        final_layer,
        [name for name, _ in tta_parameters],
    )
    return tta_parameters, original


def _restore_tta_parameters(tta_parameters, original):
    with torch.no_grad():
        for name, parameter in tta_parameters:
            parameter.copy_(original[name])


def _max_tta_difference(tta_parameters, original):
    if not tta_parameters:
        return 0.0
    return max(
        (parameter.detach() - original[name]).abs().max().item()
        for name, parameter in tta_parameters
    )


def _tta_score_query(
    model,
    batch,
    tta_parameters,
    original,
    tta_lr,
    tta_temperature,
    tta_max_delta,
    fp16,
):
    """One entropy-minimizing update and one scoring pass for one query."""
    _restore_tta_parameters(tta_parameters, original)
    restored_difference = _max_tta_difference(tta_parameters, original)

    def score_with_current_parameters():
        amp_context = torch.cuda.amp.autocast() if fp16 else nullcontext()
        with amp_context:
            return model(batch, None).scores.reshape(-1)

    # The pre-update pass is deliberately inference-only and uses all candidates.
    with torch.no_grad():
        initial_scores = score_with_current_parameters()

    # This is the only gradient update for the query. No labels, qrels, or teacher
    # scores participate in the objective.
    with torch.enable_grad():
        scores_for_adaptation = score_with_current_parameters()
        probabilities = torch.softmax(
            scores_for_adaptation.float() / tta_temperature,
            dim=-1,
        )
        entropy = -(
            probabilities * torch.log(probabilities + 1e-12)
        ).sum()
        # Keep Adam moments in FP32: FP16 squared gradients can underflow,
        # producing updates far larger than the requested learning rate.
        master_parameters = [
            parameter.detach().float().clone().requires_grad_(True)
            for _, parameter in tta_parameters
        ]
        optimizer = torch.optim.Adam(
            master_parameters,
            lr=tta_lr,
            eps=1e-4,
        )
        optimizer.zero_grad(set_to_none=True)
        for _, parameter in tta_parameters:
            parameter.grad = None
        entropy.backward()
        gradient_norm = torch.sqrt(
            sum(
                parameter.grad.float().pow(2).sum()
                for _, parameter in tta_parameters
                if parameter.grad is not None
            )
        ) if any(parameter.grad is not None for _, parameter in tta_parameters) else torch.tensor(0.0, device=entropy.device)
        for master, (_, parameter) in zip(master_parameters, tta_parameters):
            if parameter.grad is not None:
                master.grad = parameter.grad.detach().float().clone()
        optimizer.step()

        with torch.no_grad():
            for master, (name, parameter) in zip(master_parameters, tta_parameters):
                master.clamp_(
                    min=original[name].float() - tta_max_delta,
                    max=original[name].float() + tta_max_delta,
                )
                parameter.copy_(master)

    changed_difference = _max_tta_difference(tta_parameters, original)
    with torch.no_grad():
        adapted_scores = score_with_current_parameters()
        adapted_probabilities = torch.softmax(
            adapted_scores.float() / tta_temperature,
            dim=-1,
        )
        adapted_entropy = -(
            adapted_probabilities
            * torch.log(adapted_probabilities + 1e-12)
        ).sum()

    accepted = adapted_entropy < entropy

    # Do not leak adaptation from this query into the next one.
    _restore_tta_parameters(tta_parameters, original)
    final_difference = _max_tta_difference(tta_parameters, original)
    return (
        initial_scores.detach(),
        adapted_scores.detach(),
        entropy.detach().float().item(),
        adapted_entropy.detach().float().item(),
        gradient_norm.detach().float().item(),
        bool(accepted.item()),
        restored_difference,
        changed_difference,
        final_difference,
    )

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, TTAArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, tta_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, tta_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if training_args.local_rank > 0 or training_args.n_gpu > 1:
        raise NotImplementedError('Multi-GPU encoding is not supported.')

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )
    if tokenizer.name_or_path.lower().startswith("qwen") and tokenizer.bos_token_id is None:
        tokenizer.bos_token = "<|endoftext|>"#标记序列的开始
        tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token#标记序列的结束
    tokenizer.padding_side = 'right'
    use_teacher =  False 
    model = DistillRerankerModel.load(
        student_model_path=model_args.model_name_or_path,
        teacher_model_path =model_args.teacher_model_name_or_path,
        use_teacher= use_teacher,
        cache_dir=model_args.cache_dir,
        torch_dtype=torch.float16 if training_args.fp16 else torch.float32,
        loss_type = model_args.loss_type
    )
    tokenizer_teacher = AutoTokenizer.from_pretrained(
        model_args.teacher_model_name_or_path,
        cache_dir=model_args.cache_dir
    )
    if tokenizer_teacher.pad_token_id is None:
        tokenizer_teacher.pad_token_id = tokenizer_teacher.eos_token_id

    if tokenizer_teacher.name_or_path.lower().startswith("qwen") and tokenizer_teacher.bos_token_id is None:
        tokenizer_teacher.bos_token = "<|endoftext|>"
        #tokenizer_teacher.pad_token_id = tokenizer_teacher.unk_token_id
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
    if tta_args.tta and tta_args.tta_use_tokenizer_pad:
        model.student.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    all_results = {}
    start_time = time.time()  # Start timing inference

    if tta_args.tta:
        tta_parameters, original_tta_parameters = _setup_tta(model)
        pending_query_id = None
        pending_batches = []
        debug_query_count = 0
        processed_query_count = 0
        stop_after_debug_queries = False

        def process_pending_query(query_id, query_batches):
            nonlocal debug_query_count, processed_query_count
            if query_id is None:
                return
            query_batch = {
                key: torch.cat([part[key] for part in query_batches], dim=0)
                for key in query_batches[0]
                if key != "_docids"
            }
            (
                initial_scores,
                scores,
                entropy,
                adapted_entropy,
                gradient_norm,
                accepted,
                restored_difference,
                changed_difference,
                final_difference,
            ) = _tta_score_query(
                model,
                query_batch,
                tta_parameters,
                original_tta_parameters,
                tta_args.tta_lr,
                tta_args.tta_temperature,
                tta_args.tta_max_delta,
                training_args.fp16,
            )
            if debug_query_count < 3:
                logger.info(
                    "TTA query=%s candidates=%d entropy=%.8f "
                    "adapted_entropy=%.8f grad_norm=%.3e accepted=%s "
                    "restore_before=%.3e update_max=%.3e restore_after=%.3e "
                    "score_changed=%.3e",
                    query_id,
                    scores.numel(),
                    entropy,
                    adapted_entropy,
                    gradient_norm,
                    accepted,
                    restored_difference,
                    changed_difference,
                    final_difference,
                    (scores.float() - initial_scores.float()).abs().max().item(),
                )
                debug_query_count += 1
            score_values = scores.cpu().numpy()
            docids = [
                docid for part in query_batches for docid in part["_docids"]
            ]
            for docid, score in zip(docids, score_values):
                all_results.setdefault(query_id, []).append((docid, score))
            processed_query_count += 1

        # Keep only student tensors for TTA. The existing collator/padding and
        # model score position are untouched.
        for batch_query_ids, batch_text_ids, batch, _teacher in tqdm(rerank_loader):
            device_batch = {
                key: value.to(training_args.device)
                for key, value in batch.items()
            }
            start = 0
            while start < len(batch_query_ids):
                query_id = batch_query_ids[start]
                end = start + 1
                while end < len(batch_query_ids) and batch_query_ids[end] == query_id:
                    end += 1
                query_part = {
                    key: value[start:end]
                    for key, value in device_batch.items()
                }
                query_part["_docids"] = batch_text_ids[start:end]
                if pending_query_id is None:
                    pending_query_id = query_id
                if query_id != pending_query_id:
                    process_pending_query(pending_query_id, pending_batches)
                    if (
                        tta_args.tta_num_queries > 0
                        and processed_query_count >= tta_args.tta_num_queries
                    ):
                        stop_after_debug_queries = True
                        break
                    pending_batches = []
                    pending_query_id = query_id
                pending_batches.append(query_part)
                start = end
            if stop_after_debug_queries:
                break
        if not stop_after_debug_queries:
            process_pending_query(pending_query_id, pending_batches)
    else:
        # Keep the original inference path unchanged when TTA is disabled.
        for (batch_query_ids, batch_text_ids, batch, teacher) in tqdm(rerank_loader):
            with torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
                with torch.no_grad():
                    for k, v in batch.items():
                        batch[k] = v.to(training_args.device)
                    for k, v in teacher.items():
                        teacher[k] = v.to(training_args.device)
                    model_output = model(batch, None)
                    scores = model_output.scores.cpu().detach().numpy()
                    if use_teacher:
                        teacher_scores = model_output.teacher_scores.cpu().detach().numpy()
                        scores = scores + teacher_scores
                    else:
                        scores = scores #+ teacher_scores
                    for i in range(len(scores)):
                        qid = batch_query_ids[i]
                        docid = batch_text_ids[i]
                        score = scores[i][0]
                        if qid not in all_results:
                            all_results[qid] = []
                        all_results[qid].append((docid, score))
    end_time = time.time()  # End timing inference
    print(f"Total question processing time (inference only): {end_time - start_time:.2f} seconds")

    with open(data_args.encoded_save_path, 'w') as f:
        for qid in all_results:
            results = sorted(all_results[qid], key=lambda x: x[1], reverse=True)
            for docid, score in results:
                f.write(f'{qid}\t{docid}\t{score}\n')

if __name__ == "__main__":
    main()
