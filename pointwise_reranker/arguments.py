# arguments.py
import os
from dataclasses import dataclass, field
from typing import Optional, List
from transformers import TrainingArguments

@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "HF model id or path (student)."})
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    teacher_model_name_or_path: str = field(default=None, metadata={"help": "Teacher model path/id."})
    temperature: float = field(default=2.0)
    alpha: float = field(default=0.3)
    # Legacy margin Smooth-L1 arguments retained for compatibility only.
    # Training requires lambda_margin_sll=0; margin_sll_beta is unused.
    lambda_margin_sll: float = field(default=0.0)
    margin_sll_beta: float = field(default=1.0)
    # Legacy setwise arguments retained for historical checkpoint inference.
    # Setwise SLL training is removed; lambda_sll must remain zero.
    use_setwise_head: bool = field(default=False)
    lambda_sll: float = field(default=0.0)
    set_size: int = field(default=5)
    set_heads: int = field(default=4)
    set_layers: int = field(default=2)
    set_hidden_dim: int = field(default=256)
    set_teacher_temperature: float = field(default=2.0)
    set_student_temperature: float = field(default=2.0)
    # Used only for the trained contextual head's TS feedback at inference.
    setwise_feedback_mode: str = field(default="topk")  # topk | threshold
    setwise_feedback_top_k: int = field(default=2)
    setwise_feedback_temperature: float = field(default=2.0)
    setwise_head_learning_rate: float = field(default=1e-4)
    # Reliability-gated dual-channel distillation (RG-DAL).
    # Both setwise SLL and margin Smooth-L1 training have been removed.
    use_dal: bool = field(default=False)
    lambda_dal: float = field(default=0.01)
    use_pn_gap: bool = field(default=True)
    dal_pn_weight: float = field(default=1.0)
    dal_nn_weight: float = field(default=0.0)
    # The NN Smooth-L1 teacher target is active only for a normalized teacher
    # margin strictly larger than this value.
    dal_nn_margin_threshold: float = field(default=0.5)
    dal_max_teacher_negatives: int = field(default=2)
    # Soft, detached pair weights for the PN teacher-student margin gap.
    dal_pn_teacher_weight: float = field(default=0.1)
    dal_pn_teacher_temperature: float = field(default=0.25)
    dal_gap_temperature: float = field(default=0.25)
    dal_teacher_confidence_scale: float = field(default=0.5)
    dal_teacher_margin_cap: float = field(default=2.0)
    # The first ratio of optimizer steps uses only DeAR+KK.  DAL then ramps
    # linearly during an equally long interval (0.2 => full DAL at 40%).
    dal_warmup_ratio: float = field(default=0.2)
     
    # Label-directed PN loss weighted by the absolute nDCG change caused by
    # swapping the labelled positive with each sampled negative.  The
    # per-query weights are normalized inside the loss.
    use_pn_lambda: bool = field(default=False)
    lambda_pn_lambda: float = field(default=0.05)
    pn_lambda_temperature: float = field(default=0.25)
    # Continuous contextual residual inference.  This is deliberately
    # separate from TS-SetRank: it never turns a listwise distribution into
    # binary relevance feedback or updates a Beta posterior.
    setwise_residual_enabled: bool = field(default=False)
    setwise_residual_top_k: int = field(default=30)
    setwise_residual_context_rounds: int = field(default=3)
    setwise_residual_gamma: float = field(default=0.1)
    # Defaults to <encoded_save_path>.setwise_residual_trace.jsonl.
    setwise_residual_trace_path: Optional[str] = field(default=None)
    # modeling
    untie_encoder: bool = field(default=False)
    loss_type: str = field(default="softmax_ce")  # softmax_ce | ranknet_teacher | ranknet_binary | listnet_* | lambda_*
    # projection / pooler (kept for compatibility)
    add_pooler: bool = field(default=False)
    projection_in_dim: int = field(default=768)
    projection_out_dim: int = field(default=768)
    normalize: bool = field(default=False)
    # dtype hint
    dtype: Optional[str] = field(default="float32")
        # DeAR-RR：区域关系蒸馏参数
    use_relation_distill: bool = field(
        default=False
    )

    lambda_rat: float = field(
        default=0.05
    )

    lambda_rvr: float = field(
        default=0.05
    )

    relation_temperature: float = field(
        default=2.0
    )

    relation_heads: int = field(
        default=8
    )

    # LLaMA-3.2-3B共有28层，最后层索引为27
    student_relation_layer: int = field(
        default=27
    )

    # LLaMA2-13B共有40层，最后层索引为39
    teacher_relation_layers: str = field(
        default="39"
    )

    relation_use_rope: bool = field(
        default=True
    )

    relation_eps: float = field(
        default=1e-8
    )

    # Query-level test-time low-rank adaptation (TTA).
    # These parameters are used only by inference.py; no DeAR weights are updated.
    tta_enabled: bool = field(default=False)
    tta_rank: int = field(default=8)
    tta_gamma: float = field(default=1.0)
    tta_steps: int = field(default=20)
    tta_learning_rate: float = field(default=1e-2)
    tta_top_k: int = field(default=100)
    tta_n_pos: int = field(default=5)
    tta_n_neg: int = field(default=20)
    tta_temperature: float = field(default=0.5)
    tta_anchor_weight: float = field(default=0.0)
    tta_use_teacher: bool = field(default=False)
    tta_protect_top_k: int = field(default=0)
    tta_top10_gamma: float = field(default=-1.0)
    tta_correction_mode: bool = field(default=False)
    tta_correction_top_k: int = field(default=30)
    tta_teacher_top_k: int = field(default=5)
    tta_teacher_delta: float = field(default=0.1)
    tta_teacher_confidence_scale: float = field(default=1.0)
    # TOUR-style continuous list-reward TTA.  The teacher supplies a soft
    # distribution over a query's initial DeAR top-M candidates; only a fresh
    # query-local vector u is optimized at inference time.
    tta_listwise_reward_mode: bool = field(default=False)
    tta_listwise_top_k: int = field(default=30)
    # Kept separate from legacy pairwise-TTA parameters: a 3072-dim direct
    # vector has a much larger aggregate Adam update than the old bounded
    # residual and therefore needs conservative defaults.
    tta_listwise_steps: int = field(default=5)
    tta_listwise_learning_rate: float = field(default=1e-3)
    tta_teacher_temperature: float = field(default=0.5)
    tta_student_temperature: float = field(default=1.0)
    tta_trust_weight: float = field(default=1.0)
    tta_u_l2_weight: float = field(default=0.01)
    # Normalized entropy H(p_teacher) / log(M).  A lower value means the
    # teacher distribution is more concentrated and therefore more reliable.
    tta_teacher_entropy_max: float = field(default=0.90)
    # Mean absolute rank displacement divided by M-1, comparing teacher and
    # initial DeAR orders on the same top-M candidates.
    tta_min_rank_disagreement: float = field(default=0.10)
    # Optional nDCG-oriented discount 1/log2(rank + 1), based on initial
    # DeAR rank.  Disabled by default to retain the exact KL objective.
    tta_position_discount: bool = field(default=False)
    # RankLLaMA continuous-feedback distribution TTA.  Unlike the legacy
    # pairwise and TOUR-style branches above, this keeps a separate switch so
    # existing experiments retain their exact behavior.
    use_distribution_tta: bool = field(default=False)
    tta_candidate_k: int = field(default=30)
    tta_lr: float = field(default=1e-2)
    tta_lambda_local: float = field(default=1.0)
    tta_window_size: int = field(default=10)
    tta_window_stride: int = field(default=5)
    tta_u_norm_weight: float = field(default=1e-4)
    tta_use_confidence_gate: bool = field(default=False)
    tta_min_teacher_confidence: float = field(default=0.0)
    tta_early_stop_patience: int = field(default=3)
    tta_early_stop_tolerance: float = field(default=1e-5)
    tta_teacher_cache_path: Optional[str] = field(default=None)
    tta_log_diagnostics: bool = field(default=True)
    # Holdout-entropy-consistent hard-routing TTA.  This is deliberately a
    # separate inference-only switch: the final query ranking is exactly the
    # raw DeAR+KK+SLL branch or exactly a freshly optimized query-vector
    # branch, never a score interpolation of the two.
    use_entropy_hard_gate_tta: bool = field(default=False)
    tta_entropy_temperature: float = field(default=2.0)
    tta_entropy_threshold: float = field(default=0.0)
    tta_force_adapted: bool = field(default=False)
    tta_single_fold_gate: bool = field(default=False)
    # TS-SetRank is an independent, inference-only Bayesian post-processor.
    # Keeping this separate from TTA means disabling it preserves DeAR exactly.
    use_ts_setrank: bool = field(default=False)
    ts_candidate_topk: int = field(default=100)
    ts_budget: int = field(default=10)
    ts_batch_size: int = field(default=10)
    ts_exploration_rounds: int = field(default=2)
    ts_prior_mode: str = field(default="dear")  # uniform | dear
    ts_prior_strength: float = field(default=4.0)
    ts_prior_temperature: float = field(default=1.0)
    ts_final_mode: str = field(default="fusion")  # posterior | fusion
    ts_fusion_weight: float = field(default=0.4)
    ts_seed: int = field(default=42)
    # The mock judge is only for deterministic unit/integration tests; it has
    # no relevance knowledge and is not a valid DL19 experiment.
    ts_judge_type: str = field(default="generative")  # generative | contextual | mock
    # Deliberately distinct from model_name_or_path: the latter is always the
    # frozen DeAR pointwise checkpoint, while this is a causal setwise judge.
    setwise_model_name_or_path: Optional[str] = field(default=None)
    ts_judge_temperature: float = field(default=0.6)
    ts_judge_load_in_4bit: bool = field(default=False)
    # Compact P-ID answers need only a few tokens; a small cap reduces
    # unsolicited explanations that can contain misleading extra IDs.
    ts_judge_max_new_tokens: int = field(default=32)
    ts_judge_max_input_tokens: int = field(default=4096)
    ts_judge_document_max_chars: int = field(default=1600)
    # Defaults to <encoded_save_path>.ts_setrank_trace.jsonl when omitted.
    ts_trace_path: Optional[str] = field(default=None)
    teacher_only: bool = field(default=False)

@dataclass
class DataArguments:
    train_dir: str = field(default=None, metadata={"help": "Path OR file to json/jsonl (optional if using load_from_disk)."})
    dataset_name: str = field(default=None, metadata={"help": "Path to HF dataset saved with save_to_disk."})
    passage_field_separator: str = field(default=' ')
    dataset_proc_num: int = field(default=12)
    train_n_passages: int = field(default=8)
    positive_passage_no_shuffle: bool = field(default=False)
    negative_passage_no_shuffle: bool = field(default=False)

    encode_in_path: List[str] = field(default=None)
    encoded_save_path: str = field(default=None)
    encode_is_qry: bool = field(default=False)
    encode_num_shard: int = field(default=1)
    encode_shard_index: int = field(default=0)
    dev_path: Optional[str] = field(default=None)
    train_path: Optional[str] = field(
            default=None,
            metadata={
                "help": "Path to the local training JSON or JSONL file."
            },
        )
    dataset_split: str = field(
    default="train",
    metadata={
        "help": "Dataset split used for training."
    },
        )
    q_max_len: int = field(default=32)
    p_max_len: int = field(default=128)
    data_cache_dir: Optional[str] = field(default=None)

    def __post_init__(self):
        if self.dataset_name is not None:
            # using load_from_disk; we still set split names for convenience
            self.dataset_split = 'train'
            self.dataset_language = 'default'
        else:
            self.dataset_name = 'json'
            self.dataset_split = 'train'
            self.dataset_language = 'default'

        if self.train_dir is not None:
            if os.path.isdir(self.train_dir):
                files = os.listdir(self.train_dir)
                self.train_dir = os.path.join(os.path.abspath(os.getcwd()), self.train_dir)
                self.train_path = [
                    os.path.join(self.train_dir, f)
                    for f in files if f.endswith(('jsonl', 'json'))
                ]
            else:
                self.train_path = [self.train_dir]
        else:
            self.train_path = None

        self.dev_split = 'dev' if self.dev_path is not None else None

from transformers import IntervalStrategy

@dataclass
class ModelTrainingArguments(TrainingArguments):
    # names kept for backwards compat with your script/flags
    warmup_ratio: float = field(default=0.1)
    negatives_x_device: bool = field(default=False)
    do_encode: bool = field(default=False)

    grad_cache: bool = field(default=False)
    gc_q_chunk_size: int = field(default=4)
    gc_p_chunk_size: int = field(default=32)
    num_questions_per_batch: int = field(default=4)
    num_answers_per_question: int = field(default=4)
    eval_interval: int = field(default=100)
    ckpt_save_interval: int = field(default=1000)
    skip_unfinished_episodes: bool = field(default=True)
    max_grad_norm: float = field(default=1.0)
