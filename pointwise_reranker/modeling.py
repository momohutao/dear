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
from setwise_head import (
    SETWISE_HEAD_CONFIG,
    SETWISE_HEAD_WEIGHTS,
    SetwiseContextualHead,
    load_setwise_head,
    save_setwise_head,
    setwise_head_exists,
)
logger = logging.getLogger(__name__)

@dataclass
class RerankerOutput(ModelOutput):
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None
    teacher_scores: Optional[Tensor] = None
    query_embeddings: Optional[Tensor] = None
    document_embeddings: Optional[Tensor] = None
    classifier_embeddings: Optional[Tensor] = None
    setwise_embeddings: Optional[Tensor] = None

    dear_loss: Optional[Tensor] = None
    loss_rank: Optional[Tensor] = None
    loss_logit: Optional[Tensor] = None

    loss_dal_pn: Optional[Tensor] = None
    loss_dal_pn_gap: Optional[Tensor] = None
    loss_dal_nn: Optional[Tensor] = None
    dal_nn_active_pairs: Optional[Tensor] = None
    dal_pn_pair_weight_sum: Optional[Tensor] = None
    dal_scale: Optional[Tensor] = None
    weighted_dal: Optional[Tensor] = None

    loss_pn_lambda: Optional[Tensor] = None
    weighted_pn_lambda: Optional[Tensor] = None
    pn_lambda_mean_delta_ndcg: Optional[Tensor] = None
    loss_ucd: Optional[Tensor] = None
    weighted_ucd: Optional[Tensor] = None

    loss_rat: Optional[Tensor] = None
    weighted_rat: Optional[Tensor] = None

    loss_rvr: Optional[Tensor] = None
    weighted_rvr: Optional[Tensor] = None
    # Compatibility field for historical setwise-head inference.
    setwise_logits: Optional[Tensor] = None


def compute_pn_lambda_loss(
    student: Tensor,
    temperature: float,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """Return label-directed, nDCG swap-weighted PN loss.

    Column zero is the single labelled positive and every later column is a
    sampled negative.  With one binary-relevance positive, IDCG is one, so
    the absolute nDCG change from swapping a PN pair is simply the absolute
    difference between their rank discounts.  Ranks and weights are detached;
    gradients flow only through the pairwise softplus loss.
    """
    if student.ndim != 2:
        raise ValueError("PN-lambda student scores must have shape [B, passages].")
    if student.size(1) < 2:
        raise ValueError("PN-lambda requires at least one positive and one negative.")
    if temperature <= 0:
        raise ValueError("pn_lambda_temperature must be positive.")
    if eps <= 0:
        raise ValueError("PN-lambda eps must be positive.")

    scores = student.float()
    with torch.no_grad():
        order = torch.argsort(scores.detach(), dim=1, descending=True)
        ranks = torch.empty_like(order)
        rank_values = torch.arange(
            1,
            scores.size(1) + 1,
            device=scores.device,
            dtype=order.dtype,
        ).unsqueeze(0).expand_as(order)
        ranks.scatter_(1, order, rank_values)
        discounts = 1.0 / torch.log2(ranks.to(scores.dtype) + 1.0)
        delta_ndcg = (discounts[:, :1] - discounts[:, 1:]).abs()
        weight_sum = delta_ndcg.sum(dim=1, keepdim=True)
        normalized_weights = delta_ndcg / weight_sum.clamp_min(float(eps))

    current_margin = scores[:, :1] - scores[:, 1:]
    pair_loss = F.softplus(-current_margin / float(temperature))
    loss = (normalized_weights * pair_loss).sum(dim=1).mean()
    mean_delta_ndcg = delta_ndcg.mean()
    return loss, mean_delta_ndcg


def compute_ucd_loss(
    grouped_student: Tensor,
    grouped_teacher: Tensor,
    mu: float,
    num_low: int,
    num_high: int,
) -> Tensor:
    """Dynamic under/over-estimation correction with teacher ordering."""
    if mu <= 0 or num_low < 0 or num_high < 0:
        raise ValueError("UCD requires mu > 0 and nonnegative sample counts.")
    if num_low == 0 or grouped_student.numel() == 0:
        return grouped_student.new_zeros(())

    losses = []
    for student_row, teacher_row in zip(grouped_student, grouped_teacher):
        with torch.no_grad():
            student_order = torch.argsort(student_row, descending=True, stable=True)
            teacher_order = torch.argsort(teacher_row, descending=True, stable=True)
            student_rank = torch.empty_like(student_order)
            teacher_rank = torch.empty_like(teacher_order)
            student_rank[student_order] = torch.arange(student_order.numel(), device=student_row.device)
            teacher_rank[teacher_order] = torch.arange(teacher_order.numel(), device=student_row.device)
            low_idx = torch.nonzero(student_rank > teacher_rank, as_tuple=False).flatten()
            high_idx = torch.nonzero(student_rank < teacher_rank, as_tuple=False).flatten()
            low_weights = torch.tanh((mu * (student_rank[low_idx] - teacher_rank[low_idx]).float()).clamp_min(0))
            high_weights = torch.tanh((mu * (teacher_rank[high_idx] - student_rank[high_idx]).float()).clamp_min(0))
            if low_idx.numel() == 0 or low_weights.sum() <= 0:
                continue
            low_take = min(num_low, low_idx.numel())
            low_sel = low_idx[torch.multinomial(low_weights, low_take, replacement=False)]
            low_sel = low_sel[torch.argsort(teacher_rank[low_sel], stable=True)]
            if high_idx.numel() and high_weights.sum() > 0 and num_high > 0:
                high_take = min(num_high, high_idx.numel())
                high_sel = high_idx[torch.multinomial(high_weights, high_take, replacement=False)]
            else:
                high_sel = high_idx.new_empty((0,))

        position_losses = []
        for k, low_doc in enumerate(low_sel):
            allowed_high = high_sel[teacher_rank[high_sel] > teacher_rank[low_doc]]
            competitors = student_row[low_sel[k:]]
            if allowed_high.numel():
                competitors = torch.cat((competitors, student_row[allowed_high]), dim=0)
            position_losses.append(-(student_row[low_doc] - torch.logsumexp(competitors, dim=0)))
        if position_losses:
            losses.append(torch.stack(position_losses).mean())
    return torch.stack(losses).mean() if losses else grouped_student.new_zeros(())

class DistillRerankerModel(nn.Module):
    def __init__(
        self,
        student_model: PeftModel,
        teacher_model: Optional[PreTrainedModel],
        train_batch_size: Optional[int] = None,
        temperature: float = 2.0,
        alpha: float = 0.5,
        loss_type: str = "softmax_ce",
        use_teacher: bool = True,
        model_args: Optional[ModelArguments] = None,
        setwise_head: Optional[SetwiseContextualHead] = None,
    ):
        super().__init__()
        self.student = student_model
        self.teacher = teacher_model
        self.config = student_model.config
        self.train_batch_size = train_batch_size
        self.temperature = temperature
        self.alpha = alpha
        self.loss_type = loss_type
        self.use_teacher = use_teacher
        self.setwise_head = setwise_head
        self.use_setwise_head = setwise_head is not None
        self.setwise_head_learning_rate = (
            model_args.setwise_head_learning_rate if model_args is not None else None
        )

        self.use_dal = bool(model_args is not None and model_args.use_dal)
        self.use_pn_gap = bool(model_args is not None and model_args.use_pn_gap)
        self.lambda_dal = model_args.lambda_dal if model_args is not None else 0.0
        self.dal_pn_weight = model_args.dal_pn_weight if model_args is not None else 1.0
        self.dal_nn_weight = model_args.dal_nn_weight if model_args is not None else 0.25
        self.dal_nn_margin_threshold = (
            model_args.dal_nn_margin_threshold if model_args is not None else 0.5
        )
        self.dal_max_teacher_negatives = (
            model_args.dal_max_teacher_negatives if model_args is not None else 2
        )
        self.dal_pn_teacher_weight = (
            model_args.dal_pn_teacher_weight if model_args is not None else 0.1
        )
        self.dal_pn_teacher_temperature = (
            model_args.dal_pn_teacher_temperature if model_args is not None else 0.25
        )
        self.dal_gap_temperature = (
            model_args.dal_gap_temperature if model_args is not None else 0.25
        )
        self.dal_teacher_confidence_scale = (
            model_args.dal_teacher_confidence_scale if model_args is not None else 0.5
        )
        self.dal_teacher_margin_cap = (
            model_args.dal_teacher_margin_cap if model_args is not None else 2.0
        )
        self.dal_warmup_ratio = model_args.dal_warmup_ratio if model_args is not None else 0.0
         
        self.use_pn_lambda = bool(model_args is not None and model_args.use_pn_lambda)
        self.lambda_pn_lambda = (
            model_args.lambda_pn_lambda if model_args is not None else 0.0
        )
        self.pn_lambda_temperature = (
            model_args.pn_lambda_temperature if model_args is not None else 0.25
        )
        self.use_ucd = bool(model_args is not None and model_args.use_ucd)
        self.lambda_ucd = model_args.lambda_ucd if model_args is not None else 0.0
        self.ucd_mu = model_args.ucd_mu if model_args is not None else 0.5
        self.ucd_num_low = model_args.ucd_num_low if model_args is not None else 2
        self.ucd_num_high = model_args.ucd_num_high if model_args is not None else 2
        if self.lambda_ucd < 0 or self.ucd_mu <= 0:
            raise ValueError("UCD lambda must be nonnegative and ucd_mu positive.")

        if self.lambda_pn_lambda < 0:
            raise ValueError("lambda_pn_lambda must be nonnegative.")
        if self.pn_lambda_temperature <= 0:
            raise ValueError("pn_lambda_temperature must be positive.")

        if model_args is not None and model_args.lambda_sll != 0.0:
            raise ValueError("Setwise SLL training was removed; use --lambda_sll 0.")
        if train_batch_size and self.use_setwise_head:
            raise ValueError("Setwise head training was removed; use --use_setwise_head false.")
        if model_args is not None and model_args.lambda_margin_sll != 0.0:
            raise ValueError("Margin SLL was removed; use --lambda_margin_sll 0.")
      
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
            f"use_setwise_head={self.use_setwise_head}, "
            f"use_dal={self.use_dal}, "
            f"lambda_dal={self.lambda_dal}, "
         
            f"use_pn_lambda={self.use_pn_lambda}, "
            f"lambda_pn_lambda={self.lambda_pn_lambda}"
        )

    def _rg_dal_losses(
        self,
        student: Tensor,
        teacher: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return PN-gap loss, pair-weight sum and zero NN.

        Column 0 is the labelled MS MARCO positive; all later columns are
        sampled negatives. Teacher and student margins use the same negative
        column. Teacher confidence and student gap gates are detached pair
        weights; only positive teacher margins contribute.
        """
        if student.size(1) < 2:
            raise ValueError("RG-DAL requires train_n_passages >= 2.")
        if self.dal_pn_teacher_temperature <= 0:
            raise ValueError("dal_pn_teacher_temperature must be positive.")
        if student.ndim != 2 or teacher.shape != student.shape:
            raise ValueError("Student and teacher must have matching [B, passages] scores.")
        for name in (
            "dal_gap_temperature",
            "dal_teacher_confidence_scale",
            "dal_teacher_margin_cap",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.dal_pn_teacher_weight < 0:
            raise ValueError("dal_pn_teacher_weight must be nonnegative.")

        student = student.float()
        teacher = teacher.detach().float()
        teacher_std = teacher.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
        teacher_norm = (teacher - teacher.mean(dim=1, keepdim=True)) / teacher_std
        student_std = student.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
        student_norm = (student - student.mean(dim=1, keepdim=True)) / student_std
        student_pos_margin = student_norm[:, :1] - student_norm[:, 1:]

        teacher_pos_margin = (teacher_norm[:, :1] - teacher_norm[:, 1:]).detach()
        teacher_confidence = (
            teacher_pos_margin.clamp_min(0.0) / self.dal_teacher_confidence_scale
        ).clamp_max(1.0)
        # The gate uses the original teacher margin. Only the loss target is
        # capped. Detach both numerator and denominator weights.
        margin_gap = teacher_pos_margin - student_pos_margin.detach()
        student_gap_gate = torch.sigmoid(margin_gap / self.dal_gap_temperature).detach()
        pair_weight = (teacher_confidence * student_gap_gate).detach()
        teacher_target = teacher_pos_margin.clamp_max(self.dal_teacher_margin_cap)
        pair_loss = self.dal_pn_teacher_temperature * F.softplus(
            (teacher_target - student_pos_margin) / self.dal_pn_teacher_temperature
        )
        pair_weight_sum = pair_weight.sum()
        loss_pn_gap = (pair_weight * pair_loss).sum() / (pair_weight_sum + 1e-8)
        loss_pn = self.dal_pn_teacher_weight * loss_pn_gap

        # This recipe intentionally has no L_NN computation or contribution.
        loss_nn = student.new_zeros(())
        return (
            loss_pn,
            loss_pn_gap,
            pair_weight_sum,
            loss_nn,
        )

    @staticmethod
    def _print_score_position_debug(
        name: str,
        model: nn.Module,
        pair: Dict[str, Tensor],
        region_ids: Tensor,
    ) -> None:
        """比较Llama分类头实际取分位置与S区域位置。"""

        input_ids = pair["input_ids"]
        attention_mask = pair["attention_mask"]
        pad_token_id = model.config.pad_token_id
        seq_len = input_ids.size(-1)

        # 完全复现 Transformers 4.44.2 中
        # LlamaForSequenceClassification 的位置选择逻辑
        if pad_token_id is None:
            classifier_position = torch.full(
                (input_ids.size(0),),
                seq_len - 1,
                device=input_ids.device,
                dtype=torch.long,
            )
        else:
            classifier_position = (
                input_ids.eq(pad_token_id)
                .int()
                .argmax(dim=-1)
                - 1
            ) % seq_len

        positions = torch.arange(
            seq_len,
            device=input_ids.device,
        ).unsqueeze(0).expand_as(input_ids)

        # 根据attention_mask计算最后有效token
        last_valid_position = positions.masked_fill(
            attention_mask.eq(0),
            -1,
        ).max(dim=-1).values

        # 你当前data.py的区域编号：
        # Q=0、D=1、S=2、PAD=-1
        s_position = positions.masked_fill(
            region_ids.ne(2),
            -1,
        ).max(dim=-1).values

        # 查看实际用于padding的token ID
        padding_values = input_ids.masked_select(
            attention_mask.eq(0)
        ).unique()

        print(
            f"\n===== Score Position Debug: {name} ====="
        )
        print(
            "model.config.pad_token_id:",
            pad_token_id,
        )
        print(
            "input_shape:",
            tuple(input_ids.shape),
        )
        print(
            "observed_padding_token_ids:",
            padding_values.detach().cpu().tolist(),
        )
        print(
            "last_valid_position:",
            last_valid_position.detach().cpu().tolist(),
        )
        print(
            "classifier_position:",
            classifier_position.detach().cpu().tolist(),
        )
        print(
            "s_position:",
            s_position.detach().cpu().tolist(),
        )
        print(
            "classifier_equals_last_valid:",
            classifier_position.eq(
                last_valid_position
            ).detach().cpu().tolist(),
        )
        print(
            "classifier_equals_s:",
            classifier_position.eq(
                s_position
            ).detach().cpu().tolist(),
        )
        print(
            "=========================================\n",
            flush=True,
        )

    def forward(
    self,
    pair_student: Dict[str, Tensor],
    pair_teacher: Optional[Dict[str, Tensor]] = None,
    student_region_ids: Optional[Tensor] = None,
    teacher_region_ids: Optional[Tensor] = None,
    dal_scale: float = 1.0,
):
        if self.region_distiller is not None:
            self.region_distiller.clear()
        # Hidden states are used only for inference representation extraction.
        need_region_embeddings = (
            not self.training and student_region_ids is not None
        )
        need_hidden = need_region_embeddings or (not self.training and self.use_setwise_head)
        student_outputs = self.student(
            **pair_student,
            return_dict=True,
            output_hidden_states=need_hidden,
        )
        student_logits = student_outputs.logits

        query_embeddings = None
        document_embeddings = None
        classifier_embeddings = None
        classifier_hidden = None
        if need_region_embeddings:
            hidden = student_outputs.hidden_states[-1].float()
            q_mask = student_region_ids.eq(0).unsqueeze(-1)
            d_mask = student_region_ids.eq(1).unsqueeze(-1)
            q_count = q_mask.sum(dim=1).clamp_min(1)
            d_count = d_mask.sum(dim=1).clamp_min(1)
            query_embeddings = F.normalize(
                (hidden * q_mask).sum(dim=1) / q_count,
                p=2,
                dim=-1,
            )
            document_embeddings = F.normalize(
                (hidden * d_mask).sum(dim=1) / d_count,
                p=2,
                dim=-1,
            )
        if need_hidden:
            hidden = student_outputs.hidden_states[-1]
            input_ids = pair_student["input_ids"]
            seq_len = input_ids.size(-1)
            pad_token_id = self.student.config.pad_token_id
            if pad_token_id is None:
                classifier_position = torch.full(
                    (input_ids.size(0),), seq_len - 1,
                    device=input_ids.device, dtype=torch.long,
                )
            else:
                classifier_position = (
                    input_ids.eq(pad_token_id).int().argmax(dim=-1) - 1
                ) % seq_len
            if student_region_ids is not None:
                score_counts = student_region_ids.eq(2).sum(dim=-1)
                if not torch.all(score_counts.eq(1)):
                    raise RuntimeError("Each example must contain one score-position region id.")
                region_classifier_position = student_region_ids.eq(2).long().argmax(dim=-1)
                if not torch.equal(classifier_position, region_classifier_position):
                    raise RuntimeError(
                        "Classifier-position alignment failed: region_ids does not match "
                        "the sequence-classification score token."
                    )
            classifier_hidden = hidden[
                torch.arange(hidden.size(0), device=hidden.device),
                classifier_position,
            ]
            if need_region_embeddings:
                classifier_embeddings = F.normalize(
                    classifier_hidden.float(), p=2, dim=-1
                )

        if self.train_batch_size:
            with torch.no_grad():
                teacher_logits = None
                if (
                    self.use_teacher
                    and pair_teacher is not None
                ):
                    teacher_logits = self.teacher(
                        **pair_teacher,
                        return_dict=True,
                    ).logits
                

            # 仅在第一个batch、rank 0打印一次
            if (
                self.training
                and student_region_ids is not None
                and not getattr(
                    self,
                    "_printed_score_position_debug",
                    False,
                )
            ):
                distributed = (
                    torch.distributed.is_available()
                    and torch.distributed.is_initialized()
                )
                is_rank_zero = (
                    not distributed
                    or torch.distributed.get_rank() == 0
                )

                if is_rank_zero:
                    self._print_score_position_debug(
                        name="student",
                        model=self.student,
                        pair=pair_student,
                        region_ids=student_region_ids,
                    )

                    if (
                        teacher_logits is not None
                        and pair_teacher is not None
                        and teacher_region_ids is not None
                    ):
                        self._print_score_position_debug(
                            name="teacher",
                            model=self.teacher,
                            pair=pair_teacher,
                            region_ids=teacher_region_ids,
                        )

                self._printed_score_position_debug = True

            grouped_student = student_logits.view(
                self.train_batch_size,
                -1,
            )
            grouped_teacher = teacher_logits.view(self.train_batch_size, -1) if teacher_logits is not None else None
        

            # main supervised ranking loss
            if self.loss_type == "ranknet_teacher":
                relevance = grouped_teacher.detach()
                main_loss = RankLoss.rank_net(grouped_student, relevance)
            elif self.loss_type == "ranknet_binary":
                relevance = torch.zeros_like(grouped_student)
                relevance[:, 0] = 1
                main_loss = RankLoss.rank_net(grouped_student, relevance)
            elif self.loss_type == "softmax_ce":
                target = torch.zeros(self.train_batch_size, dtype=torch.long, device=grouped_student.device)
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
                target = torch.zeros(self.train_batch_size, dtype=torch.long, device=grouped_student.device)
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
            loss_dal_pn = torch.zeros((), device=student_logits.device, dtype=torch.float32)
            loss_dal_pn_gap = torch.zeros((), device=student_logits.device, dtype=torch.float32)
            loss_dal_nn = torch.zeros((), device=student_logits.device, dtype=torch.float32)
            dal_nn_active_pairs = torch.zeros((), device=student_logits.device, dtype=torch.long)
            dal_pn_pair_weight_sum = torch.zeros((), device=student_logits.device, dtype=torch.float32)
            if self.use_dal and self.use_pn_gap:
                if grouped_teacher is None:
                    raise ValueError("RG-DAL requires frozen teacher logits.")
                (
                    loss_dal_pn,
                    loss_dal_pn_gap,
                    dal_pn_pair_weight_sum,
                    loss_dal_nn,
                ) = self._rg_dal_losses(grouped_student, grouped_teacher)
            
            loss_pn_lambda = torch.zeros(
                (), device=student_logits.device, dtype=torch.float32
            )
            pn_lambda_mean_delta_ndcg = torch.zeros(
                (), device=student_logits.device, dtype=torch.float32
            )
            if self.use_pn_lambda:
                loss_pn_lambda, pn_lambda_mean_delta_ndcg = compute_pn_lambda_loss(
                    grouped_student,
                    self.pn_lambda_temperature,
                )
            loss_ucd = torch.zeros((), device=student_logits.device, dtype=torch.float32)
            if self.use_ucd:
                if grouped_teacher is None:
                    raise ValueError("UCD requires frozen teacher logits.")
                loss_ucd = compute_ucd_loss(
                    grouped_student,
                    grouped_teacher.detach(),
                    self.ucd_mu,
                    self.ucd_num_low,
                    self.ucd_num_high,
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

            if self.use_relation_distill:
                if student_region_ids is None:
                    raise ValueError(
                        "Missing student_region_ids."
                    )

                if teacher_region_ids is None:
                    raise ValueError(
                        "Missing teacher_region_ids."
                    )

                loss_rat, loss_rvr = (
                    self.region_distiller.compute_losses(
                        student_region_ids=student_region_ids,
                        teacher_region_ids=teacher_region_ids,
                        student_attention_mask=(
                            pair_student["attention_mask"]
                        ),
                        teacher_attention_mask=(
                            pair_teacher["attention_mask"]
                        ),
                    )
                )

            total = (
                dear_loss
                + self.lambda_rat * loss_rat
                + self.lambda_rvr * loss_rvr
                + float(dal_scale) * self.lambda_dal * (
                    self.dal_pn_weight * loss_dal_pn
                    + self.dal_nn_weight * loss_dal_nn
                )
                + self.lambda_pn_lambda * loss_pn_lambda
                + self.lambda_ucd * loss_ucd
            )
            if (
                self.training
                and not getattr(self, "_printed_total_debug", False)
            ):
                print(
                    "[TotalLossDebug] "
                    f"dear_loss={dear_loss.detach().float().item():.10f}, "
                    f"loss_rat={loss_rat.detach().float().item():.10f}, "
                    f"weighted_rat="
                    f"{(self.lambda_rat * loss_rat).detach().float().item():.10f}, "
                    f"lambda_rat={self.lambda_rat}, "
                    f"loss_dal_pn={loss_dal_pn.detach().float().item():.10f}, "
                    f"loss_dal_pn_gap={loss_dal_pn_gap.detach().float().item():.10f}, "
                    f"dal_pn_pair_weight_sum={dal_pn_pair_weight_sum.detach().item():.6f}, "
                    f"loss_dal_nn={loss_dal_nn.detach().float().item():.10f}, "
                    f"dal_nn_active_pairs={dal_nn_active_pairs.detach().item()}, "
                    f"dal_scale={float(dal_scale):.6f}, "
                    f"weighted_dal={(float(dal_scale) * self.lambda_dal * (self.dal_pn_weight * loss_dal_pn + self.dal_nn_weight * loss_dal_nn)).detach().float().item():.10f}, "
                    f"loss_pn_lambda={loss_pn_lambda.detach().float().item():.10f}, "
                    f"weighted_pn_lambda={(self.lambda_pn_lambda * loss_pn_lambda).detach().float().item():.10f}, "
                    f"pn_lambda_mean_delta_ndcg={pn_lambda_mean_delta_ndcg.detach().float().item():.6f}, "
                    f"total={total.detach().float().item():.10f}"
                )

                self._printed_total_debug = True
            return RerankerOutput(
                loss=total,
                scores=student_logits,
                teacher_scores=teacher_logits,

                dear_loss=dear_loss.detach(),
                loss_rank=main_loss.detach(),
                loss_logit=kl.detach() if grouped_teacher is not None else None,

                loss_dal_pn=loss_dal_pn.detach(),
                loss_dal_pn_gap=loss_dal_pn_gap.detach(),
                loss_dal_nn=loss_dal_nn.detach(),
                dal_nn_active_pairs=dal_nn_active_pairs.detach(),
                dal_pn_pair_weight_sum=dal_pn_pair_weight_sum.detach(),
                dal_scale=torch.as_tensor(dal_scale, device=student_logits.device).detach(),
                weighted_dal=(
                    float(dal_scale) * self.lambda_dal *
                    (self.dal_pn_weight * loss_dal_pn + self.dal_nn_weight * loss_dal_nn)
                ).detach(),


                loss_pn_lambda=loss_pn_lambda.detach(),
                weighted_pn_lambda=(self.lambda_pn_lambda * loss_pn_lambda).detach(),
                pn_lambda_mean_delta_ndcg=pn_lambda_mean_delta_ndcg.detach(),

                loss_ucd=loss_ucd.detach(),
                weighted_ucd=(self.lambda_ucd * loss_ucd).detach(),

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

            )

        # inference mode
        tlogits = None
        if self.use_teacher and pair_teacher is not None:
            with torch.no_grad():
                tlogits = self.teacher(**pair_teacher, return_dict=True).logits
        return RerankerOutput(
            loss=None,
            scores=student_logits,
            teacher_scores=tlogits,
            query_embeddings=query_embeddings,
            document_embeddings=document_embeddings,
            classifier_embeddings=classifier_embeddings,
            setwise_embeddings=(
                classifier_hidden.float() if classifier_hidden is not None else None
            ),
        )

    def gradient_checkpointing_enable(self):
        # match llama base structures
        if hasattr(self.student, "base_model") and hasattr(self.student.base_model, "model"):
            self.student.base_model.model.gradient_checkpointing_enable()

    @classmethod
    def load(
            cls,
            student_model_path,
            teacher_model_path=None,
            student_pad_token_id=None,
            teacher_pad_token_id=None,
            temperature=2.0,
            alpha=0.3,
            use_teacher=False,
            loss_type="softmax_ce",
            model_args: Optional[ModelArguments] = None,
            **hf_kwargs,
        ):
        config = PeftConfig.from_pretrained(student_model_path)
        student_base = AutoModelForSequenceClassification.from_pretrained(
            config.base_model_name_or_path, num_labels=1, **hf_kwargs
        )
        if student_pad_token_id is None:
            raise ValueError(
                "student_pad_token_id must be provided "
                "by the student tokenizer."
            )

        student_base.config.pad_token_id = int(
            student_pad_token_id
        )
        student = PeftModel.from_pretrained(student_base, student_model_path)
        student = student.merge_and_unload()

        teacher = None
        if use_teacher and teacher_model_path:
            if teacher_pad_token_id is None:
                raise ValueError(
                    "teacher_pad_token_id must be provided "
                    "when use_teacher=True."
                )

            teacher = (
                AutoModelForSequenceClassification
                .from_pretrained(
                    teacher_model_path,
                    num_labels=1,
                    **hf_kwargs,
                )
            )

            teacher.config.pad_token_id = int(
                teacher_pad_token_id
            )

        setwise_head = None
        # A continuous residual run needs exactly the same saved head and
        # classifier-position representation as SLL/TS inference, even
        # though it does not enable the legacy binary-feedback TS branch.
        use_contextual_head = bool(
            model_args is not None
            and (
                model_args.use_setwise_head
                or model_args.setwise_residual_enabled
            )
        )
        if use_contextual_head:
            setwise_head = load_setwise_head(student_model_path)
        return cls(
            student_model=student,
            teacher_model=teacher,
            
            temperature=temperature,
            alpha=alpha,
            use_teacher=use_teacher,
            loss_type=loss_type,
            model_args=model_args,
            setwise_head=setwise_head,
        )

    @classmethod
    def build(
        cls,
        model_args: ModelArguments,
        train_args: TrainingArguments,
        student_pad_token_id: int,
        teacher_pad_token_id: int,
        temperature=2.0,
        alpha=0.5,
        **hf_kwargs,
    ):
        # Continuing from a DeAR adapter checkpoint must preserve its trained
        # LoRA and pointwise classification head.  For a base path we retain
        # the original new-LoRA initialization path.
        try:
            resume_peft_config = PeftConfig.from_pretrained(
                model_args.model_name_or_path
            )
        except (ValueError, OSError):
            resume_peft_config = None
        base_path = (
            resume_peft_config.base_model_name_or_path
            if resume_peft_config is not None
            else model_args.model_name_or_path
        )
        student_base = AutoModelForSequenceClassification.from_pretrained(
            base_path,
            num_labels=1,
            **hf_kwargs,
        )

        if train_args.gradient_checkpointing:
            student_base.enable_input_require_grads()

        if student_pad_token_id is None:
            raise ValueError(
                "student_pad_token_id must be provided "
                "by the student tokenizer."
            )

        student_base.config.pad_token_id = int(
            student_pad_token_id
        )

        if resume_peft_config is not None:
            student = PeftModel.from_pretrained(
                student_base,
                model_args.model_name_or_path,
                is_trainable=True,
            )
        else:
            peft_cfg = LoraConfig(
                base_model_name_or_path=model_args.model_name_or_path,
                task_type=TaskType.SEQ_CLS,
                r=32,
                lora_alpha=64,
                lora_dropout=0.1,
                target_modules=[
                    "q_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj",
                ],
                inference_mode=False,
            )
            student = get_peft_model(student_base, peft_cfg)

        teacher = (
            AutoModelForSequenceClassification
            .from_pretrained(
                model_args.teacher_model_name_or_path,
                num_labels=1,
                **hf_kwargs,
            )
        )

        if teacher_pad_token_id is None:
            raise ValueError(
                "teacher_pad_token_id must be provided "
                "by the teacher tokenizer."
            )

        teacher.config.pad_token_id = int(
            teacher_pad_token_id
        )

     
        setwise_head = None
        if model_args.use_setwise_head:
            if setwise_head_exists(model_args.model_name_or_path):
                setwise_head = load_setwise_head(model_args.model_name_or_path)
            else:
                setwise_head = SetwiseContextualHead(
                    input_dim=student.config.hidden_size,
                    hidden_dim=model_args.set_hidden_dim,
                    heads=model_args.set_heads,
                    layers=model_args.set_layers,
                )
        model = cls(
            student_model=student,
            teacher_model=teacher,
            train_batch_size=(
                train_args.per_device_train_batch_size
            ),
            temperature=temperature,
            alpha=alpha,
            loss_type=model_args.loss_type,
            use_teacher=True,
            model_args=model_args,
            setwise_head=setwise_head,
        )

        # DeepSpeed compatibility
        for module in model.modules():
            if not hasattr(
                module,
                "ds_grads_remaining",
            ):
                module.ds_grads_remaining = 0

        return model

    def save(self, output_dir: str):
        self.student.save_pretrained(output_dir)
        if self.setwise_head is not None:
            save_setwise_head(self.setwise_head, output_dir)
