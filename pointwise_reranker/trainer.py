# trainer.py
import os
import logging
import math
from collections import Counter
from typing import Optional

import torch
from transformers.trainer import Trainer
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from peft import get_peft_model_state_dict
from setwise_head import SETWISE_HEAD_CONFIG, SETWISE_HEAD_WEIGHTS

logger = logging.getLogger(__name__)

class RerankerTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_amp = self.args.bf16 or self.args.fp16
        # RG-DAL must not influence the early adaptation of the pointwise
        # ranker.  Keep its first `ratio` of optimizer steps at exactly zero,
        # then linearly ramp it over another interval of the same length.
        ratio = float(getattr(self.model, "dal_warmup_ratio", 0.0))
        max_steps = int(self.args.max_steps)
        self.dal_hold_steps = max(0, math.ceil(max_steps * ratio)) if max_steps > 0 else 0
        self.dal_ramp_steps = self.dal_hold_steps
        self._reset_pn_lambda_log_totals()
        self._repro_first_batch_logged = False
        self._repro_first_qids = []
        self._repro_first_10_qids_logged = False
        self._hard_injected_count = 0
        self._hard_query_count = 0
        self._hard_reasons = Counter()
        self._ucd_loss_sum = 0.0
        self._ucd_batches = 0

    def _reset_pn_lambda_log_totals(self):
        self._pn_lambda_loss_sum = 0.0
        self._pn_lambda_delta_sum = 0.0
        self._pn_lambda_batches = 0

    def log(self, logs):
        if self._hard_query_count:
            for reason in ('missing_candidate', 'probability_missed', 'already_present', 'replaced'):
                logs['hard_' + reason + '_count'] = self._hard_reasons[reason]
            self._hard_reasons.clear()
            logs['hard_injected_queries'] = self._hard_injected_count
            logs['hard_injection_rate'] = self._hard_injected_count / self._hard_query_count
            self._hard_injected_count = self._hard_query_count = 0
        if bool(getattr(self.model, "use_pn_lambda", False)):
            count = self._pn_lambda_batches
            logs["pn_lambda_loss"] = self._pn_lambda_loss_sum / count if count else 0.0
            logs["pn_lambda_mean_delta_ndcg"] = (
                self._pn_lambda_delta_sum / count if count else 0.0
            )
            self._reset_pn_lambda_log_totals()
        if bool(getattr(self.model, "use_ucd", False)):
            logs["ucd_loss"] = self._ucd_loss_sum / self._ucd_batches if self._ucd_batches else 0.0
            logs["ucd_weighted_loss"] = float(getattr(self.model, "lambda_ucd", 0.0)) * logs["ucd_loss"]
            self._ucd_loss_sum = 0.0
            self._ucd_batches = 0
        return super().log(logs)

    def _dal_scale(self) -> float:
        if not bool(getattr(self.model, "use_dal", False)):
            return 0.0
        step = int(self.state.global_step)
        if step < self.dal_hold_steps:
            return 0.0
        if self.dal_ramp_steps == 0:
            return 1.0
        return min(1.0, (step - self.dal_hold_steps + 1) / self.dal_ramp_steps)

    def create_optimizer(self):
        """Use a small continued-training LR for DeAR and a larger head LR."""
        if self.optimizer is not None:
            return self.optimizer
        head_prefix = "setwise_head."
        backbone_params, head_params = [], []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            (head_params if name.startswith(head_prefix) else backbone_params).append(parameter)
        groups = [{
            "params": backbone_params,
            "lr": self.args.learning_rate,
            "weight_decay": self.args.weight_decay,
        }]
        if head_params:
            head_lr = self.model.setwise_head_learning_rate
            if head_lr is None:
                raise ValueError("Missing setwise_head_learning_rate for contextual SLL.")
            groups.append({
                "params": head_params,
                "lr": head_lr,
                "weight_decay": self.args.weight_decay,
            })
        logger.info(
            "Optimizer parameter groups: DeAR/KK=%d params at lr=%g; "
            "setwise_head=%d params at lr=%s",
            len(backbone_params),
            self.args.learning_rate,
            len(head_params),
            self.model.setwise_head_learning_rate if head_params else "n/a",
        )
        self.optimizer = torch.optim.AdamW(
            groups,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )
        logger.info(
            "REPRO optimizer=AdamW betas=(%s,%s) eps=%s "
            "weight_decay=%s initial_state_entries=%d",
            self.args.adam_beta1,
            self.args.adam_beta2,
            self.args.adam_epsilon,
            self.args.weight_decay,
            len(self.optimizer.state),
        )
        return self.optimizer

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir or self.args.output_dir

        # ZeRO-3 下不能在 Trainer._save() 中 gather：
        # Trainer._save() 通常仅 rank 0 会执行，会导致 NCCL 死锁。
        if is_deepspeed_zero3_enabled():
            logger.info(
                "Skip Trainer._save under ZeRO-3. "
                "The PEFT adapter will be saved after training on all ranks."
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
        repro_qids = inputs.pop("_repro_qids", [])
        repro_positive_docids = inputs.pop("_repro_positive_docids", [])
        repro_negative_docids = inputs.pop("_repro_negative_docids", [])
        self._hard_reasons.update(inputs.pop('_hard_reason', []))
        hard_injected = inputs.pop('_hard_injected', [])
        self._hard_injected_count += sum(hard_injected)
        self._hard_query_count += len(hard_injected)
        if not self._repro_first_batch_logged:
            logger.info("REPRO first_batch_qids=%s", repro_qids)
            logger.info("REPRO first_batch_positive_docids=%s", repro_positive_docids)
            logger.info("REPRO first_batch_negative_docids=%s", repro_negative_docids)
            self._repro_first_batch_logged = True
        if not self._repro_first_10_qids_logged:
            remaining = 10 - len(self._repro_first_qids)
            self._repro_first_qids.extend(repro_qids[:remaining])
            if len(self._repro_first_qids) == 10:
                logger.info("REPRO first_10_sampled_qids=%s", self._repro_first_qids)
                self._repro_first_10_qids_logged = True
        outputs = model(
            pair_student=inputs["student"],
            pair_teacher=inputs["teacher"],
            student_region_ids=inputs.get(
                "student_region_ids"
            ),
            teacher_region_ids=inputs.get(
                "teacher_region_ids"
            ),
            dal_scale=self._dal_scale(),
        )

        if model.training and bool(getattr(model, "use_pn_lambda", False)):
            self._pn_lambda_batches += 1
            self._pn_lambda_loss_sum += float(
                outputs.loss_pn_lambda.detach().float().item()
            )
            self._pn_lambda_delta_sum += float(
                outputs.pn_lambda_mean_delta_ndcg.detach().float().item()
            )
        if model.training and bool(getattr(model, "use_ucd", False)):
            self._ucd_loss_sum += float(outputs.loss_ucd.detach().float().item())
            self._ucd_batches += 1

        if return_outputs:
            return outputs.loss, outputs

        return outputs.loss
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

    # The contextual head is outside PEFT, so gather and save it explicitly.
    # Without this step a resumed model would silently get a random setwise
    # head at inference/training time.
    if getattr(model, "setwise_head", None) is not None:
        setwise_state = {}
        for name, param in model.setwise_head.named_parameters():
            with deepspeed.zero.GatheredParameters([param], modifier_rank=0):
                if is_rank0:
                    setwise_state[name] = param.detach().cpu().clone()
        if is_rank0:
            torch.save(setwise_state, os.path.join(output_dir, SETWISE_HEAD_WEIGHTS))
            import json
            with open(os.path.join(output_dir, SETWISE_HEAD_CONFIG), "w", encoding="utf-8") as file:
                json.dump(model.setwise_head.export_config(), file, indent=2, sort_keys=True)
            logger.info("Saved contextual setwise head to %s", output_dir)

    # rank 0 写完文件后，其余 rank 才允许退出。
    if distributed:
        dist.barrier()
