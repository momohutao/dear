import math
from typing import Dict, List, Tuple

import torch
from torch import Tensor

from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)


def _resolve_llama_layers(model):
    """
    同时兼容：
    1. LlamaForSequenceClassification
    2. PEFT包装后的LlamaForSequenceClassification
    """

    if hasattr(model, "get_base_model"):
        base_model = model.get_base_model()
    else:
        base_model = model

    # LlamaForSequenceClassification.model.layers
    if (
        hasattr(base_model, "model")
        and hasattr(base_model.model, "layers")
    ):
        return base_model.model.layers

    # 处理可能存在的额外model包装
    if (
        hasattr(base_model, "model")
        and hasattr(base_model.model, "model")
        and hasattr(base_model.model.model, "layers")
    ):
        return base_model.model.model.layers

    raise ValueError(
        "Cannot find LLaMA transformer layers. "
        f"Model type: {type(model)}"
    )


class QKVHookStore:
    """
    捕获指定Transformer层中q_proj/k_proj/v_proj的输出。

    捕获到的Q/K尚未应用RoPE，之后在关系模块中显式应用。
    """

    def __init__(
        self,
        model,
        layer_indices: List[int],
        name: str,
    ):
        self.name = name
        self.layers = _resolve_llama_layers(model)
        self.layer_indices = layer_indices

        self.cache: Dict[int, Dict[str, Tensor]] = {}
        self.attention_modules = {}
        self.handles = []

        for layer_idx in layer_indices:
            if layer_idx < 0 or layer_idx >= len(self.layers):
                raise ValueError(
                    f"{name} layer index {layer_idx} is invalid. "
                    f"The model has {len(self.layers)} layers."
                )

            attention = self.layers[layer_idx].self_attn
            self.attention_modules[layer_idx] = attention

            for projection_name in [
                "q_proj",
                "k_proj",
                "v_proj",
            ]:
                projection = getattr(
                    attention,
                    projection_name,
                )

                handle = projection.register_forward_hook(
                    self._make_hook(
                        layer_idx,
                        projection_name,
                    )
                )

                self.handles.append(handle)

    def _make_hook(
        self,
        layer_idx: int,
        projection_name: str,
    ):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                output = output[0]

            if layer_idx not in self.cache:
                self.cache[layer_idx] = {}

            self.cache[layer_idx][projection_name] = output

        return hook

    def clear(self):
        self.cache.clear()

    def remove(self):
        for handle in self.handles:
            handle.remove()

        self.handles.clear()


def _split_original_heads(
    projected_states: Tensor,
    number_of_heads: int,
) -> Tensor:
    """
    输入：
        [B, L, number_of_heads * head_dim]

    输出：
        [B, number_of_heads, L, head_dim]
    """

    batch_size, sequence_length, hidden_size = (
        projected_states.shape
    )

    if hidden_size % number_of_heads != 0:
        raise ValueError(
            f"Projection dimension {hidden_size} "
            f"is not divisible by {number_of_heads} heads."
        )

    head_dim = hidden_size // number_of_heads

    return (
        projected_states
        .view(
            batch_size,
            sequence_length,
            number_of_heads,
            head_dim,
        )
        .transpose(1, 2)
        .contiguous()
    )


def _repeat_kv(
    hidden_states: Tensor,
    repeat_number: int,
) -> Tensor:
    """
    将GQA的KV heads展开到与Query heads相同。

    输入：
        [B, H_kv, L, D]

    输出：
        [B, H_kv * repeat_number, L, D]
    """

    if repeat_number == 1:
        return hidden_states

    batch_size, number_of_kv_heads, seq_len, head_dim = (
        hidden_states.shape
    )

    hidden_states = hidden_states[:, :, None, :, :]

    hidden_states = hidden_states.expand(
        batch_size,
        number_of_kv_heads,
        repeat_number,
        seq_len,
        head_dim,
    )

    return hidden_states.reshape(
        batch_size,
        number_of_kv_heads * repeat_number,
        seq_len,
        head_dim,
    )


def _reshape_relation_heads(
    hidden_states: Tensor,
    relation_heads: int,
) -> Tensor:
    """
    MiniLMv2 Relation Heads重构。

    输入：
        [B, original_heads, L, head_dim]

    先合并原始heads：
        [B, L, original_heads * head_dim]

    再重构为统一Relation Heads：
        [B, relation_heads, L, relation_dim]
    """

    batch_size, original_heads, seq_len, head_dim = (
        hidden_states.shape
    )

    merged_dimension = original_heads * head_dim

    if merged_dimension % relation_heads != 0:
        raise ValueError(
            f"Merged dimension {merged_dimension} "
            f"is not divisible by "
            f"{relation_heads} relation heads."
        )

    relation_dim = merged_dimension // relation_heads

    hidden_states = (
        hidden_states
        .transpose(1, 2)
        .contiguous()
        .view(
            batch_size,
            seq_len,
            merged_dimension,
        )
    )

    hidden_states = hidden_states.view(
        batch_size,
        seq_len,
        relation_heads,
        relation_dim,
    )

    return (
        hidden_states
        .permute(0, 2, 1, 3)
        .contiguous()
    )


def _build_position_ids(
    attention_mask: Tensor,
) -> Tensor:
    """
    根据attention_mask生成有效token的位置编号。

    右侧padding时：
        attention_mask = [1,1,1,0,0]
        position_ids   = [0,1,2,0,0]
    """

    position_ids = attention_mask.long().cumsum(-1) - 1

    position_ids.masked_fill_(
        attention_mask == 0,
        0,
    )

    return position_ids


def _apply_llama_rope(
    attention_module,
    query_states: Tensor,
    key_states: Tensor,
    position_ids: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    对Q/K应用LLaMA的RoPE。

    兼容不同transformers版本的rotary_emb接口。
    """

    if hasattr(attention_module, "rotary_emb"):
        rotary_embedding = attention_module.rotary_emb

    else:
        config = attention_module.config

        try:
            rotary_embedding = LlamaRotaryEmbedding(
                config=config,
            )
        except TypeError:
            head_dim = query_states.shape[-1]

            rotary_embedding = LlamaRotaryEmbedding(
                head_dim,
                max_position_embeddings=(
                    config.max_position_embeddings
                ),
                base=config.rope_theta,
            )

        rotary_embedding = rotary_embedding.to(
            query_states.device
        )

    try:
        # 新版transformers接口
        cos, sin = rotary_embedding(
            key_states,
            position_ids,
        )

    except TypeError:
        # 较旧transformers接口
        cos, sin = rotary_embedding(
            key_states,
            seq_len=key_states.shape[-2],
        )

    if cos.dim() == 2:
        # 旧版apply_rotary_pos_emb需要position_ids
        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            position_ids,
        )

    else:
        # 新版cos/sin已经按照position_ids计算
        try:
            query_states, key_states = (
                apply_rotary_pos_emb(
                    query_states,
                    key_states,
                    cos,
                    sin,
                    position_ids=None,
                    unsqueeze_dim=1,
                )
            )
        except TypeError:
            query_states, key_states = (
                apply_rotary_pos_emb(
                    query_states,
                    key_states,
                    cos,
                    sin,
                )
            )

    return query_states, key_states


def _compute_token_relation(
    source_states: Tensor,
    target_states: Tensor,
    attention_mask: Tensor,
    temperature: float,
    use_causal_mask: bool,
) -> Tensor:
    """
    计算token-token关系概率。

    source_states:
        [B, R, L, D]

    target_states:
        [B, R, L, D]

    输出：
        [B, R, L, L]
    """

    relation_dim = source_states.shape[-1]
    sequence_length = source_states.shape[-2]

    # 关系损失在FP32下计算，避免V100 FP16下出现NaN
    source_states = source_states.float()
    target_states = target_states.float()

    relation_logits = torch.matmul(
        source_states,
        target_states.transpose(-1, -2),
    )

    relation_logits = relation_logits / (
        math.sqrt(float(relation_dim))
        * temperature
    )

    # 屏蔽目标padding位置
    key_valid_mask = attention_mask.bool()[
        :,
        None,
        None,
        :,
    ]

    allowed_mask = key_valid_mask

    if use_causal_mask:
        causal_mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=relation_logits.device,
            )
        )

        causal_mask = causal_mask[
            None,
            None,
            :,
            :,
        ]

        allowed_mask = (
            allowed_mask
            & causal_mask
        )

    relation_logits = relation_logits.masked_fill(
        ~allowed_mask,
        torch.finfo(relation_logits.dtype).min,
    )

    relation_probabilities = torch.softmax(
        relation_logits,
        dim=-1,
    )

    # padded source token不参与后续区域聚合
    source_valid_mask = attention_mask.bool()[
        :,
        None,
        :,
        None,
    ]

    relation_probabilities = (
        relation_probabilities
        * source_valid_mask
    )

    return relation_probabilities


def _aggregate_regions(
    token_relations: Tensor,
    region_ids: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    把token-token关系聚合成Q/D/S的3×3区域关系。

    token_relations:
        [B, R, L, L]

    region_ids:
        [B, L]
        Q=0, D=1, S=2, Padding=-1

    返回：
        region_relations: [B, R, 3, 3]
        region_valid:     [B, 3]
    """

    region_ids = region_ids.to(
        token_relations.device
    )

    batch_size, relation_heads, _, _ = (
        token_relations.shape
    )

    region_relations = torch.zeros(
        batch_size,
        relation_heads,
        3,
        3,
        dtype=token_relations.dtype,
        device=token_relations.device,
    )

    region_valid = torch.zeros(
        batch_size,
        3,
        dtype=torch.bool,
        device=token_relations.device,
    )

    for source_region in range(3):
        source_mask = (
            region_ids == source_region
        ).float()

        source_count = source_mask.sum(
            dim=-1
        )

        region_valid[:, source_region] = (
            source_count > 0
        )

        # 对源区域token取平均
        source_average = (
            token_relations
            * source_mask[
                :,
                None,
                :,
                None,
            ]
        ).sum(dim=2)

        source_average = source_average / (
            source_count[
                :,
                None,
                None,
            ].clamp_min(1.0)
        )

        # 对每个目标区域的概率质量求和
        for target_region in range(3):
            target_mask = (
                region_ids == target_region
            ).float()

            region_relations[
                :,
                :,
                source_region,
                target_region,
            ] = (
                source_average
                * target_mask[:, None, :]
            ).sum(dim=-1)

    return region_relations, region_valid


def _reverse_region_kl(
    student_probabilities: Tensor,
    teacher_probabilities: Tensor,
    student_region_valid: Tensor,
    teacher_region_valid: Tensor,
    temperature: float,
    eps: float,
) -> Tensor:
    """
    计算Idea定义的：
        KL(student || teacher)

    注意：这只用于新增RAT/RVR。
    原DeAR Logit KD仍然保持仓库原实现。
    """

    # Q/D/S必须在师生两侧都存在
    valid_samples = (
        student_region_valid
        & teacher_region_valid
    ).all(dim=-1)

    if not valid_samples.any():
        # 保持计算图且返回0
        return (
            student_probabilities.sum()
            * 0.0
        )

    student_probabilities = (
        student_probabilities[valid_samples]
    )

    teacher_probabilities = (
        teacher_probabilities[valid_samples]
        .detach()
    )
  
    student_probabilities = (
        student_probabilities.clamp_min(eps)
    )

    teacher_probabilities = (
        teacher_probabilities.clamp_min(eps)
    )

    # clamp后重新归一化
    student_probabilities = (
        student_probabilities
        / student_probabilities.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(eps)
    )

    teacher_probabilities = (
        teacher_probabilities
        / teacher_probabilities.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(eps)
    )

    kl = (
        student_probabilities
        * (
            student_probabilities.log()
            - teacher_probabilities.log()
        )
    ).sum(dim=-1)

    # kl: [batch, relation_heads, source_regions]

    source_weights = torch.tensor(
       [0.0, 0.25, 1.0],
        dtype=kl.dtype,
        device=kl.device,
    )

    weighted_kl = (
        kl
        * source_weights[None, None, :]
    ).sum(dim=-1)

    weighted_kl = (
        weighted_kl
        / source_weights.sum().clamp_min(eps)
    )

    return (
        weighted_kl.mean()
        * (temperature ** 2)
    )


class RegionRelationDistiller:
    """
    DeAR-RR的RAT/RVR关系蒸馏模块。

    该类没有可训练参数，只负责：
    1. 捕获Q/K/V；
    2. 计算区域关系；
    3. 返回RAT/RVR损失。
    """

    def __init__(
        self,
        student,
        teacher,
        student_layer: int,
        teacher_layers: List[int],
        relation_heads: int = 8,
        temperature: float = 2.0,
        use_rope: bool = True,
        eps: float = 1e-8,
    ):
        if len(teacher_layers) == 0:
            raise ValueError(
                "teacher_layers cannot be empty."
            )

        self.student_layer = student_layer
        self.teacher_layers = teacher_layers
        self.relation_heads = relation_heads
        self.temperature = temperature
        self.use_rope = use_rope
        self.eps = eps

        self.student_hooks = QKVHookStore(
            model=student,
            layer_indices=[student_layer],
            name="student",
        )

        self.teacher_hooks = QKVHookStore(
            model=teacher,
            layer_indices=teacher_layers,
            name="teacher",
        )

    def clear(self):
        self.student_hooks.clear()
        self.teacher_hooks.clear()

    def _layer_distributions(
        self,
        hook_store: QKVHookStore,
        layer_idx: int,
        region_ids: Tensor,
        attention_mask: Tensor,
    ):
        if layer_idx not in hook_store.cache:
            raise RuntimeError(
                f"No Q/K/V cache found for "
                f"{hook_store.name} layer {layer_idx}. "
                "Make sure model forward was executed "
                "after hooks were registered."
            )

        cached = hook_store.cache[layer_idx]

        for projection_name in [
            "q_proj",
            "k_proj",
            "v_proj",
        ]:
            if projection_name not in cached:
                raise RuntimeError(
                    f"Missing {projection_name} output "
                    f"at {hook_store.name} layer {layer_idx}."
                )

        query_projected = cached["q_proj"]
        key_projected = cached["k_proj"]
        value_projected = cached["v_proj"]

        attention_module = (
            hook_store.attention_modules[layer_idx]
        )

        config = attention_module.config

        number_of_query_heads = (
            config.num_attention_heads
        )

        number_of_kv_heads = getattr(
            config,
            "num_key_value_heads",
            number_of_query_heads,
        )

        if (
            number_of_query_heads
            % number_of_kv_heads
            != 0
        ):
            raise ValueError(
                "Query heads must be divisible by "
                "KV heads for GQA expansion."
            )

        query_states = _split_original_heads(
            query_projected,
            number_of_query_heads,
        )

        key_states = _split_original_heads(
            key_projected,
            number_of_kv_heads,
        )

        value_states = _split_original_heads(
            value_projected,
            number_of_kv_heads,
        )

        if self.use_rope:
            position_ids = _build_position_ids(
                attention_mask
            )

            query_states, key_states = (
                _apply_llama_rope(
                    attention_module,
                    query_states,
                    key_states,
                    position_ids,
                )
            )

        # RAT：K按照真实GQA规则展开
        repeat_number = (
            number_of_query_heads
            // number_of_kv_heads
        )

        key_states_for_rat = _repeat_kv(
            key_states,
            repeat_number,
        )

        query_relation_heads = (
            _reshape_relation_heads(
                query_states,
                self.relation_heads,
            )
        )

        key_relation_heads = (
            _reshape_relation_heads(
                key_states_for_rat,
                self.relation_heads,
            )
        )
        # RVR：使用未展开的真实V heads
        value_relation_heads = (
            _reshape_relation_heads(
                value_states,
                self.relation_heads,
            )
        )

        rat_token_relations = (
            _compute_token_relation(
                source_states=key_relation_heads,
                target_states=key_relation_heads,
                attention_mask=attention_mask,
                temperature=self.temperature,
                use_causal_mask=False,
            )
        )

        rvr_token_relations = (
            _compute_token_relation(
                source_states=value_relation_heads,
                target_states=value_relation_heads,
                attention_mask=attention_mask,
                temperature=self.temperature,
                use_causal_mask=False,
            )
        )

        rat_regions, region_valid = (
            _aggregate_regions(
                rat_token_relations,
                region_ids,
            )
        )

        rvr_regions, _ = _aggregate_regions(
            rvr_token_relations,
            region_ids,
        )

        return (
            rat_regions,
            rvr_regions,
            region_valid,
        )

    def compute_losses(
        self,
        student_region_ids: Tensor,
        teacher_region_ids: Tensor,
        student_attention_mask: Tensor,
        teacher_attention_mask: Tensor,
    ):
        """
        必须在学生和教师都完成forward之后调用。
        """

        (
            student_rat,
            student_rvr,
            student_valid,
        ) = self._layer_distributions(
            hook_store=self.student_hooks,
            layer_idx=self.student_layer,
            region_ids=student_region_ids,
            attention_mask=student_attention_mask,
        )

        teacher_rat_layers = []
        teacher_rvr_layers = []
        teacher_valid = None

        for teacher_layer in self.teacher_layers:
            (
                teacher_rat,
                teacher_rvr,
                current_teacher_valid,
            ) = self._layer_distributions(
                hook_store=self.teacher_hooks,
                layer_idx=teacher_layer,
                region_ids=teacher_region_ids,
                attention_mask=teacher_attention_mask,
            )

            teacher_rat_layers.append(
                teacher_rat.detach()
            )

            teacher_rvr_layers.append(
                teacher_rvr.detach()
            )

            if teacher_valid is None:
                teacher_valid = (
                    current_teacher_valid
                )

        # 单层时等于该层，多层时进行等权融合
        teacher_rat = torch.stack(
            teacher_rat_layers,
            dim=0,
        ).mean(dim=0)

        teacher_rvr = torch.stack(
            teacher_rvr_layers,
            dim=0,
        ).mean(dim=0)

        loss_rat = _reverse_region_kl(
            student_probabilities=student_rat,
            teacher_probabilities=teacher_rat,
            student_region_valid=student_valid,
            teacher_region_valid=teacher_valid,
            temperature=self.temperature,
            eps=self.eps,
        )

        loss_rvr = _reverse_region_kl(
            student_probabilities=student_rvr,
            teacher_probabilities=teacher_rvr,
            student_region_valid=student_valid,
            teacher_region_valid=teacher_valid,
            temperature=self.temperature,
            eps=self.eps,
        )

        return loss_rat, loss_rvr