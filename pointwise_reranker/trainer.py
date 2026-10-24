# trainer.py
import os
import json
import logging
import math
import re
import subprocess
import sys
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, SequentialSampler
from transformers.trainer import Trainer
from transformers import TrainerCallback
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from peft import get_peft_model_state_dict
from eval_text import format_eval_document

logger = logging.getLogger(__name__)

class RerankerTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_amp = (
            self.args.bf16
            or self.args.fp16
        )

        self._component_sums = {}
        self._component_count = 0

    def _save(
        self,
        output_dir: Optional[str] = None,
        state_dict=None,
    ):
        output_dir = (
            output_dir
            or self.args.output_dir
        )

        # ZeRO-3不能只在rank 0中进行参数gather，
        # 因此这里跳过Trainer默认保存。
        if is_deepspeed_zero3_enabled():
            logger.info(
                "Skip Trainer._save under ZeRO-3. "
                "PEFT adapters are saved by the "
                "periodic callback or after training."
            )
            return

        self.model.save(output_dir)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        **kwargs,
    ):
        outputs = model(
            pair_student=inputs["student"],
            pair_teacher=inputs["teacher"],
            student_region_ids=inputs.get(
                "student_region_ids"
            ),
            teacher_region_ids=inputs.get(
                "teacher_region_ids"
            ),
        )

        component_values = {
            "dear_loss": outputs.dear_loss,
            "loss_margin_mse": (
                outputs.loss_margin_mse
            ),
            "weighted_margin_mse": (
                outputs.weighted_margin_mse
            ),
            "loss_rat": outputs.loss_rat,
            "weighted_rat": outputs.weighted_rat,
            "loss_rvr": outputs.loss_rvr,
            "weighted_rvr": outputs.weighted_rvr,
            "loss_down": outputs.loss_down,
            "weighted_down": outputs.weighted_down,
            "kd_loss": outputs.loss_logit,
            "total": outputs.loss,
        }

        for name, value in component_values.items():
            if value is None:
                continue

            # Keep detached metrics on the GPU until a logging step. Calling
            # .item() for every metric and microbatch synchronizes the GPU.
            scalar = value.detach().float()
            if name in self._component_sums:
                self._component_sums[name].add_(scalar)
            else:
                self._component_sums[name] = scalar.clone()

        self._component_count += 1

        if return_outputs:
            return outputs.loss, outputs

        return outputs.loss

    def log(self, logs, *args, **kwargs):
        if self._component_count > 0:
            for name, total_value in (
                self._component_sums.items()
            ):
                logs[name] = (
                    total_value.item() / self._component_count
                )

            self._component_sums = {}
            self._component_count = 0

        return super().log(
            logs,
            *args,
            **kwargs,
        )

def save_zero3_peft_adapter(
    model,
    output_dir: str,
):
    import os
    import torch
    import torch.distributed as dist
    import deepspeed

    os.makedirs(output_dir, exist_ok=True)

    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    is_rank0 = rank == 0

    # 确保四个 rank 都到达保存位置。
    if distributed:
        dist.barrier()

    gathered_state_dict = {}

    # 关键：这里必须由所有 rank 一起执行，不可套 if rank == 0。
    for name, param in model.student.named_parameters():
        if (
            "lora_" not in name
            and "modules_to_save" not in name
        ):
            continue

        with deepspeed.zero.GatheredParameters(
            [param],
            modifier_rank=0,
        ):
            if is_rank0:
                gathered_state_dict[name] = (
                    param.detach()
                    .cpu()
                    .clone()
                )

    # gather 全部结束后，只有 rank 0 写磁盘。
    if is_rank0:
        model.student.save_pretrained(
            output_dir,
            state_dict=gathered_state_dict,
            safe_serialization=False,
        )
        logger.info(
            "Saved complete ZeRO-3 PEFT adapter to %s",
            output_dir,
        )

    # rank 0 写完文件后，其余 rank 才允许退出。
    if distributed:
        dist.barrier()
class PeriodicAdapterSaveCallback(TrainerCallback):
    def __init__(self, save_steps: int):
        self.save_steps = int(save_steps)
        self.last_saved_step = -1

    def on_step_end(
        self,
        args,
        state,
        control,
        model=None,
        **kwargs,
    ):
        step = int(state.global_step)

        if self.save_steps <= 0:
            return control

        if step <= 0:
            return control

        if step % self.save_steps != 0:
            return control

        if step == self.last_saved_step:
            return control

        checkpoint_dir = os.path.join(
            args.output_dir,
            f"adapter-step-{step}",
        )

        # ZeRO-3 gather必须由所有rank共同进入。
        if is_deepspeed_zero3_enabled():
            save_zero3_peft_adapter(
                model,
                checkpoint_dir,
            )
        elif state.is_world_process_zero:
            os.makedirs(
                checkpoint_dir,
                exist_ok=True,
            )
            model.student.save_pretrained(
                checkpoint_dir,
                safe_serialization=False,
            )

        self.last_saved_step = step

        if state.is_world_process_zero:
            logger.info(
                "Saved inference adapter at step %d to %s",
                step,
                checkpoint_dir,
            )

        return control


class _DLEvalDataset(Dataset):
    """Small, read-only DL candidate dataset with DeAR's exact pair packing."""

    def __init__(self, path, tokenizer, q_max_len, p_max_len):
        self.rows = []
        with open(path, "r", encoding="utf-8") as reader:
            for line_number, line in enumerate(reader, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    self.rows.append(
                        (
                            str(row["query_id"]),
                            str(row["docid"]),
                            str(row["query"]),
                            str(row.get("title", "")),
                            str(row["text"]),
                        )
                    )
                except (KeyError, TypeError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"Malformed early-stop evaluation row at {path}:{line_number}"
                    ) from exc
        if not self.rows:
            raise ValueError(f"Empty early-stop evaluation file: {path}")
        self.tokenizer = tokenizer
        self.q_max_len = q_max_len
        self.p_max_len = p_max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        query_id, docid, query, title, text = self.rows[index]
        query_ids = self.tokenizer.encode(
            "query: " + query,
            add_special_tokens=False,
            max_length=self.q_max_len,
            truncation=True,
        )
        passage_ids = self.tokenizer.encode(
            format_eval_document(title, text),
            add_special_tokens=False,
            max_length=self.p_max_len - 3,
            truncation=True,
        )
        encoded = self.tokenizer.prepare_for_model(
            [self.tokenizer.bos_token_id] + query_ids,
            [self.tokenizer.bos_token_id] + passage_ids,
            add_special_tokens=False,
            truncation="only_second",
            max_length=self.q_max_len + self.p_max_len,
            padding=False,
            return_token_type_ids=False,
        )
        return query_id, docid, encoded


class _DLEvalCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        query_ids = [item[0] for item in features]
        docids = [item[1] for item in features]
        batch = self.tokenizer.pad(
            [item[2] for item in features],
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return query_ids, docids, batch


class DLAverageEarlyStoppingCallback(TrainerCallback):
    """Select checkpoints by DL19/DL20 or BEIR-8 mean nDCG@10."""

    def __init__(
        self,
        tokenizer,
        eval_steps,
        patience,
        min_delta,
        batch_size,
        q_max_len,
        p_max_len,
        dl19_path,
        dl20_path,
        dl19_qrels,
        dl20_qrels,
        metric="dl",
        beir_data_dir=None,
        beir_qrels_dir=None,
    ):
        self.eval_steps = int(eval_steps)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.batch_size = int(batch_size)
        if self.eval_steps <= 0:
            raise ValueError("early-stop eval_steps must be positive")
        if self.patience <= 0:
            raise ValueError("early-stop patience must be positive")
        if self.batch_size <= 0:
            raise ValueError("early-stop eval batch size must be positive")
        if tokenizer.bos_token_id is None:
            raise ValueError("Early-stop tokenizer must define bos_token_id")
        self.tokenizer = tokenizer
        self.max_length = int(q_max_len) + int(p_max_len)
        self.metric = metric
        self.q_max_len, self.p_max_len = q_max_len, p_max_len
        self.dataset_paths = {"dl19": dl19_path, "dl20": dl20_path}
        self.qrels = {"dl19": dl19_qrels, "dl20": dl20_qrels}
        if metric == "beir8":
            names = {
                "covid": "trec-covid", "nfc": "nfcorpus",
                "touche": "webis-touche2020", "dbpedia": "dbpedia-entity",
                "scifact": "scifact", "signal": "signal1m",
                "news": "trec-news", "robust04": "robust04",
            }
            self.dataset_paths = {name: os.path.join(beir_data_dir, name + ".jsonl") for name in names}
            self.qrels = {name: os.path.join(beir_qrels_dir, f"qrels.beir-v1.0.0-{slug}.test.txt")
                          for name, slug in names.items()}
        elif metric != "dl":
            raise ValueError("early_stop_metric must be 'dl' or 'beir8'")
        for path in [*self.dataset_paths.values(), *self.qrels.values()]:
            if not os.path.isfile(path) or os.path.getsize(path) == 0:
                raise ValueError(f"Missing or empty evaluation file: {path}")
        self.best_score = -math.inf
        self.bad_evaluations = 0
        self.last_evaluated_step = -1
        self.history = []

    @staticmethod
    def _distributed_info():
        distributed = dist.is_available() and dist.is_initialized()
        return (
            distributed,
            dist.get_rank() if distributed else 0,
            dist.get_world_size() if distributed else 1,
        )

    def _score_dataset(self, model, dataset, args):
        distributed, rank, world_size = self._distributed_info()
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
            if distributed
            else SequentialSampler(dataset)
        )
        loader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            collate_fn=_DLEvalCollator(self.tokenizer, self.max_length),
            num_workers=0,
            pin_memory=True,
        )
        local_results = []
        model.eval()
        with torch.no_grad():
            for query_ids, docids, batch in loader:
                batch = {key: value.to(args.device) for key, value in batch.items()}
                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=bool(args.fp16),
                ):
                    # Bypass the training wrapper's train_n_passages grouping.
                    # This is the same student classifier used by inference.
                    scores = model.student(**batch, return_dict=True).logits.reshape(-1)
                local_results.extend(
                    (qid, docid, float(score))
                    for qid, docid, score in zip(
                        query_ids,
                        docids,
                        scores.detach().float().cpu().tolist(),
                    )
                )
        if distributed:
            gathered = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, local_results)
            if rank == 0:
                return [item for shard in gathered for item in shard]
            return None
        return local_results

    @staticmethod
    def _qid_key(query_id):
        return (0, int(query_id)) if query_id.isdigit() else (1, query_id)

    def _evaluate_with_trec(self, name, results, qrels_path, step_dir):
        # DistributedSampler pads its tail. A dictionary removes those duplicates.
        unique = {(query_id, docid): score for query_id, docid, score in results}
        grouped = {}
        for (query_id, docid), score in unique.items():
            grouped.setdefault(query_id, []).append((docid, score))
        run_path = os.path.join(step_dir, f"ranked_{name}.trec")
        with open(run_path, "w", encoding="utf-8") as writer:
            for query_id in sorted(grouped, key=self._qid_key):
                ranked = sorted(grouped[query_id], key=lambda item: item[1], reverse=True)
                for rank, (docid, score) in enumerate(ranked, 1):
                    writer.write(f"{query_id} Q0 {docid} {rank} {score:.10g} earlystop\n")
        command = [
            sys.executable,
            "-m",
            "pyserini.eval.trec_eval",
            "-c",
            "-m",
            "ndcg_cut.10",
            qrels_path,
            run_path,
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=True)
        eval_log = completed.stdout + completed.stderr
        with open(os.path.join(step_dir, f"{name}_evaluation.log"), "w", encoding="utf-8") as writer:
            writer.write(eval_log)
        match = re.search(r"ndcg_cut_10\s+all\s+([0-9.]+)", eval_log)
        if not match:
            raise RuntimeError(f"Could not parse {name} nDCG@10 from trec_eval output")
        return float(match.group(1))

    @staticmethod
    def _update_best_link(output_dir, checkpoint_dir):
        target = os.path.join(output_dir, "best-adapter")
        temporary = target + f".tmp.{os.getpid()}"
        if os.path.lexists(temporary):
            os.unlink(temporary)
        os.symlink(os.path.relpath(checkpoint_dir, output_dir), temporary)
        if os.path.isdir(target) and not os.path.islink(target):
            raise RuntimeError(f"Refusing to replace real directory: {target}")
        os.replace(temporary, target)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = int(state.global_step)
        if step <= 0 or step % self.eval_steps != 0 or step == self.last_evaluated_step:
            return control
        if model is None:
            raise RuntimeError("Early-stop callback did not receive the model")
        self.last_evaluated_step = step
        distributed, rank, _ = self._distributed_info()

        checkpoint_dir = os.path.join(args.output_dir, f"adapter-step-{step}")
        if is_deepspeed_zero3_enabled():
            save_zero3_peft_adapter(model, checkpoint_dir)
        elif rank == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.student.save_pretrained(checkpoint_dir, safe_serialization=False)
        if distributed:
            dist.barrier()

        all_results = {}
        for name, path in self.dataset_paths.items():
            dataset = _DLEvalDataset(path, self.tokenizer, self.q_max_len, self.p_max_len)
            all_results[name] = self._score_dataset(model, dataset, args)
            del dataset

        payload = torch.zeros(6, dtype=torch.float64, device=args.device)
        if rank == 0:
            step_dir = os.path.join(args.output_dir, "early-stop-eval", f"step-{step}")
            os.makedirs(step_dir, exist_ok=True)
            scores = {name: self._evaluate_with_trec(name, results, self.qrels[name], step_dir)
                      for name, results in all_results.items()}
            mean_score = sum(scores.values()) / len(scores)
            if self.metric == "beir8":
                with open(os.path.join(step_dir, "beir8_summary.tsv"), "w", encoding="utf-8") as writer:
                    writer.write("dataset\tndcg@10\tndcg@10_x100\n")
                    for name, value in scores.items():
                        writer.write(f"{name}\t{value:.4f}\t{value * 100:.2f}\n")
            improved = mean_score > self.best_score + self.min_delta
            if improved:
                self.best_score = mean_score
                self.bad_evaluations = 0
                self._update_best_link(args.output_dir, checkpoint_dir)
            else:
                self.bad_evaluations += 1
            should_stop = self.bad_evaluations >= self.patience
            record = {
                "step": step,
                "metric": self.metric,
                **{f"{name}_ndcg_cut_10": value for name, value in scores.items()},
                "mean_ndcg_cut_10": mean_score,
                "improved": improved,
                "best_mean_ndcg_cut_10": self.best_score,
                "bad_evaluations": self.bad_evaluations,
                "patience": self.patience,
                "best_checkpoint": os.path.realpath(os.path.join(args.output_dir, "best-adapter")),
            }
            self.history.append(record)
            with open(os.path.join(args.output_dir, "early_stop_state.json"), "w", encoding="utf-8") as writer:
                json.dump({"history": self.history, "latest": record}, writer, indent=2)
                writer.write("\n")
            logger.info(
                "Early-stop eval step=%d metric=%s scores=%s mean=%.4f "
                "best=%.4f bad=%d/%d improved=%s",
                step, self.metric, scores, mean_score, self.best_score,
                self.bad_evaluations, self.patience, improved,
            )
            payload = torch.tensor(
                [0, 0, mean_score, self.best_score, self.bad_evaluations, float(should_stop)],
                dtype=torch.float64,
                device=args.device,
            )
        if distributed:
            dist.broadcast(payload, src=0)
        self.best_score = float(payload[3].item())
        self.bad_evaluations = int(payload[4].item())
        state.best_metric = self.best_score
        state.best_model_checkpoint = os.path.realpath(os.path.join(args.output_dir, "best-adapter"))
        if bool(payload[5].item()):
            control.should_training_stop = True
        model.train()
        return control
