# modeling.py
from dataclasses import dataclass
from typing import Dict, Optional

import logging
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from transformers import AutoModelForSequenceClassification, PreTrainedModel
from transformers.utils import ModelOutput

from peft import PeftConfig, PeftModel, get_peft_model, LoraConfig, TaskType

from arguments import ModelArguments, ModelTrainingArguments as TrainingArguments
from rank_loss import RankLoss
from region_relation import (
    RegionRelationDistiller,
)
logger = logging.getLogger(__name__)

# LlamaForSequenceClassification按config.pad_token_id寻找"最后一个非pad token"打分。
# 设为词表中不存在的id，使打分位置固定为序列最后一个位置（padding到max_length后的最后一位），
# 与原DeAR一致，并避免序列中出现id 0（Llama-3中为"!"）时打分位置提前。
SCORE_POSITION_PAD_TOKEN_ID = -1


def use_last_position_scoring(model):
    model.config.pad_token_id = SCORE_POSITION_PAD_TOKEN_ID

@dataclass
class RerankerOutput(ModelOutput):
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None
    teacher_scores: Optional[Tensor] = None

    dear_loss: Optional[Tensor] = None
    loss_rank: Optional[Tensor] = None
    loss_logit: Optional[Tensor] = None

    loss_margin_mse: Optional[Tensor] = None
    weighted_margin_mse: Optional[Tensor] = None

    loss_rat: Optional[Tensor] = None
    weighted_rat: Optional[Tensor] = None

    loss_rvr: Optional[Tensor] = None
    weighted_rvr: Optional[Tensor] = None

    loss_down: Optional[Tensor] = None
    weighted_down: Optional[Tensor] = None

class DistillRerankerModel(nn.Module):
    def __init__(
        self,
        student_model: PeftModel,
        teacher_model: Optional[PreTrainedModel],
        train_batch_size: Optional[int] = None,
        train_n_passages: Optional[int] = None,
        temperature: float = 2.0,
        alpha: float = 0.5,
        loss_type: str = "softmax_ce",
        use_teacher: bool = True,
        model_args: Optional[ModelArguments] = None,
    ):
        super().__init__()
        self.student = student_model
        self.teacher = teacher_model
        self.config = student_model.config
        self.train_batch_size = train_batch_size
        self.train_n_passages = train_n_passages
        self.temperature = temperature
        self.alpha = alpha
        self.loss_type = loss_type
        self.use_teacher = use_teacher

        self.lambda_margin_mse = (
            model_args.lambda_margin_mse
            if model_args is not None
            else 0.0
        )

        self.use_relation_distill = bool(
            model_args is not None
            and model_args.use_relation_distill
        )

        self.lambda_rat = (
            model_args.lambda_rat
            if model_args is not None
            else 0.0
        )

        self.lambda_rvr = (
            model_args.lambda_rvr
            if model_args is not None
            else 0.0
        )

        self.lambda_down = (
            model_args.lambda_down
            if model_args is not None
            else 0.0
        )

        self.region_distiller = None

        if self.use_relation_distill:
            #参数里的层加入到teacher_layers数组
            teacher_layers = [
                int(x.strip())
                for x in (
                    model_args.teacher_relation_layers
                    .split(",")
                )
                if x.strip()
            ]

            self.region_distiller = (
                RegionRelationDistiller(
                    student=self.student,
                    teacher=self.teacher,
                    student_layer=(
                        model_args.student_relation_layer
                    ),
                    teacher_layers=teacher_layers,
                    relation_heads=(
                        model_args.relation_heads
                    ),
                    temperature=(
                        model_args.relation_temperature
                    ),
                    use_rope=(
                        model_args.relation_use_rope
                    ),
                    eps=model_args.relation_eps,
                )
            )

        if self.teacher is not None and self.use_teacher:
            for p in self.teacher.parameters():
                p.requires_grad = False

        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        print(
            f"[DistillRerankerModel] loss_type={loss_type}, "
            f"T={temperature}, alpha={alpha}, "
            f"lambda_margin_mse={self.lambda_margin_mse}"
        )

    def forward(
    self,
    pair_student: Dict[str, Tensor],
    pair_teacher: Optional[Dict[str, Tensor]] = None,
    student_region_ids: Optional[Tensor] = None,
    teacher_region_ids: Optional[Tensor] = None,
):
        if self.region_distiller is not None:
            self.region_distiller.clear()
        # student logits: (B*N, 1)
        student_logits = self.student(**pair_student, return_dict=True).logits  # (batch*passages, 1)

        if self.train_batch_size:
            with torch.no_grad():
                teacher_logits = None
                if self.use_teacher and pair_teacher is not None:
                    teacher_logits = self.teacher(**pair_teacher, return_dict=True).logits  # (B*N, 1)

            # 按每个query的passage数分组，最后一个不满的batch也能正确分组
            n_passages = self.train_n_passages
            if n_passages is None or student_logits.size(0) % n_passages != 0:
                raise ValueError(
                    f"Cannot group {student_logits.size(0)} logits "
                    f"by train_n_passages={n_passages}."
                )
            grouped_student = student_logits.view(-1, n_passages)  # (B, N)
            grouped_teacher = teacher_logits.view(-1, n_passages) if teacher_logits is not None else None
            batch_size = grouped_student.size(0)

            # main supervised ranking loss
            if self.loss_type == "ranknet_teacher":
                relevance = grouped_teacher.detach()
                main_loss = RankLoss.rank_net(grouped_student, relevance)
            elif self.loss_type == "ranknet_binary":
                relevance = torch.zeros_like(grouped_student)
                relevance[:, 0] = 1
                main_loss = RankLoss.rank_net(grouped_student, relevance)
            elif self.loss_type == "softmax_ce":
                target = torch.zeros(batch_size, dtype=torch.long, device=grouped_student.device)
                main_loss = self.cross_entropy(grouped_student, target)
            elif self.loss_type == "listnet_teacher":
                relevance = grouped_teacher.detach()
                main_loss = RankLoss.list_net(grouped_student, relevance)
            elif self.loss_type == "lambda_loss_teacher":
                relevance = grouped_teacher.detach()
                main_loss = RankLoss.lambda_loss(grouped_student, relevance, k=5, weighing_scheme="ndcgLoss2_scheme")
            elif self.loss_type == "listnet_binary":
                relevance = torch.zeros_like(grouped_student); relevance[:, 0] = 1
                main_loss = RankLoss.list_net(grouped_student, relevance)
            elif self.loss_type == "lambda_loss_binary":
                relevance = torch.zeros_like(grouped_student); relevance[:, 0] = 1
                main_loss = RankLoss.lambda_loss(grouped_student, relevance, k=5, weighing_scheme="ndcgLoss2_scheme")
            elif self.loss_type == "ranknet_binary_softmax_ce":
                relevance = torch.zeros_like(grouped_student); relevance[:, 0] = 1
                loss_rank = RankLoss.rank_net(grouped_student, relevance)
                target = torch.zeros(batch_size, dtype=torch.long, device=grouped_student.device)
                loss_ce = self.cross_entropy(grouped_student, target)
                beta = 0.5
                main_loss = beta * loss_rank + (1 - beta) * loss_ce
            else:
                raise ValueError(f"Unsupported loss type: {self.loss_type}")

            # KD loss (requires teacher)
            if grouped_teacher is not None:
                s = F.log_softmax(grouped_student / self.temperature, dim=1)
                t = F.softmax(grouped_teacher / self.temperature, dim=1)
                kl = F.kl_div(s, t, reduction="batchmean") * (self.temperature ** 2)
                total = (1 - self.alpha) * main_loss + self.alpha * kl
            else:
                total = main_loss
            dear_loss = total
            # Margin-MSE：
            # 第0列为正例，后面的列为该query对应的负例
            loss_margin_mse = torch.zeros(
                (),
                device=student_logits.device,
                dtype=torch.float32,
            )

            if self.lambda_margin_mse != 0.0:
                if grouped_teacher is None:
                    raise ValueError(
                        "Margin-MSE requires teacher logits."
                    )

                if grouped_student.size(1) < 2:
                    raise ValueError(
                        "Margin-MSE requires train_n_passages >= 2."
                    )

                # 学生：正例分数 - 每个负例分数
                student_margin = (
                    grouped_student[:, :1]
                    - grouped_student[:, 1:]
                ).float()

                # 教师：正例分数 - 每个负例分数
                teacher_margin = (
                    grouped_teacher[:, :1]
                    - grouped_teacher[:, 1:]
                ).detach().float()

                loss_margin_mse = F.smooth_l1_loss(
                    student_margin,
                    teacher_margin,
                    reduction="mean",
                )
            loss_rat = torch.zeros(
                (),
                device=student_logits.device,
                dtype=torch.float32,
            )

            loss_rvr = torch.zeros(
                (),
                device=student_logits.device,
                dtype=torch.float32,
            )
            loss_down = torch.zeros(
                (),
                device=student_logits.device,
                dtype=torch.float32,
            )

            if self.use_relation_distill:
                if student_region_ids is None:
                    raise ValueError(
                        "Missing student_region_ids."
                    )

                if teacher_region_ids is None:
                    raise ValueError(
                        "Missing teacher_region_ids."
                    )

                loss_rat, loss_rvr, loss_down = (
                    self.region_distiller.compute_losses(
                        student_region_ids=student_region_ids,
                        teacher_region_ids=teacher_region_ids,
                        student_attention_mask=(
                            pair_student["attention_mask"]
                        ),
                        teacher_attention_mask=(
                            pair_teacher["attention_mask"]
                        ),
                        compute_rvr=(self.lambda_rvr != 0.0),
                    )
                )

            total = (
                dear_loss
                + self.lambda_margin_mse * loss_margin_mse
                + self.lambda_rat * loss_rat
                + self.lambda_rvr * loss_rvr
                + self.lambda_down * loss_down
            )
            if (
                self.training
                and not getattr(self, "_printed_total_debug", False)
            ):
                print(
                    "[TotalLossDebug] "
                    f"dear_loss={dear_loss.detach().float().item():.10f}, "
                    f"loss_margin_mse={loss_margin_mse.detach().float().item():.10f}, "
                    f"weighted_margin_mse="
                    f"{(self.lambda_margin_mse * loss_margin_mse).detach().float().item():.10f}, "
                    f"lambda_margin_mse={self.lambda_margin_mse}, "
                    f"loss_rat={loss_rat.detach().float().item():.10f}, "
                    f"weighted_rat="
                    f"{(self.lambda_rat * loss_rat).detach().float().item():.10f}, "
                    f"lambda_rat={self.lambda_rat}, "
                    f"loss_down={loss_down.detach().float().item():.10f}, "
                    f"weighted_down="
                    f"{(self.lambda_down * loss_down).detach().float().item():.10f}, "
                    f"lambda_down={self.lambda_down}, "
                    f"total={total.detach().float().item():.10f}"
                )

                self._printed_total_debug = True
            return RerankerOutput(
                loss=total,
                scores=student_logits,
                teacher_scores=teacher_logits,

                dear_loss=dear_loss.detach(),
                loss_rank=main_loss.detach(),
                loss_logit=kl.detach(),

                loss_margin_mse=loss_margin_mse.detach(),
                weighted_margin_mse=(
                    self.lambda_margin_mse
                    * loss_margin_mse
                ).detach(),

                loss_rat=loss_rat.detach(),
                weighted_rat=(
                    self.lambda_rat
                    * loss_rat
                ).detach(),

                loss_rvr=loss_rvr.detach(),
                weighted_rvr=(
                    self.lambda_rvr
                    * loss_rvr
                ).detach(),

                loss_down=loss_down.detach(),
                weighted_down=(
                    self.lambda_down * loss_down
                ).detach(),
            )

        # inference mode
        tlogits = None
        if self.use_teacher and pair_teacher is not None:
            with torch.no_grad():
                tlogits = self.teacher(**pair_teacher, return_dict=True).logits
        return RerankerOutput(loss=None, scores=student_logits, teacher_scores=tlogits)

    def gradient_checkpointing_enable(self):
        # match llama base structures
        if hasattr(self.student, "base_model") and hasattr(self.student.base_model, "model"):
            self.student.base_model.model.gradient_checkpointing_enable()

    @classmethod
    def load(
        cls,
        student_model_path,
        teacher_model_path=None,
        temperature=2.0,
        alpha=0.3,
        use_teacher=False,
        loss_type="softmax_ce",
        **hf_kwargs,
    ):
        config = PeftConfig.from_pretrained(student_model_path)
        student_base = AutoModelForSequenceClassification.from_pretrained(
            config.base_model_name_or_path, num_labels=1, **hf_kwargs
        )
        use_last_position_scoring(student_base)
        student = PeftModel.from_pretrained(student_base, student_model_path)
        student = student.merge_and_unload()

        teacher = None
        if use_teacher and teacher_model_path:
            teacher = AutoModelForSequenceClassification.from_pretrained(
                teacher_model_path, num_labels=1, **hf_kwargs
            )
            use_last_position_scoring(teacher)

        return cls(
            student_model=student,
            teacher_model=teacher,
            temperature=temperature,
            alpha=alpha,
            use_teacher=use_teacher,
            loss_type=loss_type
        )

    @classmethod
    def build(cls, model_args: ModelArguments, train_args: TrainingArguments, train_n_passages: int,
              temperature=2.0, alpha=0.5, **hf_kwargs):
        student_base = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path, num_labels=1, **hf_kwargs
        )
        if train_args.gradient_checkpointing:
            student_base.enable_input_require_grads()
        use_last_position_scoring(student_base)

        peft_cfg = LoraConfig(
            base_model_name_or_path=model_args.model_name_or_path,
            task_type=TaskType.SEQ_CLS,
            r=32,
            lora_alpha=64,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
            inference_mode=False,
        )
        student = get_peft_model(student_base, peft_cfg)

        teacher = AutoModelForSequenceClassification.from_pretrained(
            model_args.teacher_model_name_or_path, num_labels=1, **hf_kwargs
        )
        use_last_position_scoring(teacher)

        model = cls(
            student_model=student,
            teacher_model=teacher,
            train_batch_size=(
                train_args.per_device_train_batch_size
            ),
            train_n_passages=train_n_passages,
            temperature=temperature,
            alpha=alpha,
            loss_type=model_args.loss_type,
            use_teacher=True,
            model_args=model_args,
        )
        # deepspeed compat
        for m in model.modules():
            if not hasattr(m, "ds_grads_remaining"):
                m.ds_grads_remaining = 0
        return model

    def save(self, output_dir: str):
        self.student.save_pretrained(output_dir)
