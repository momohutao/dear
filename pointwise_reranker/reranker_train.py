# reranker_train.py
import logging
import os
import sys
import hashlib
import random

import numpy as np
import torch
from transformers import AutoTokenizer, HfArgumentParser, set_seed
from transformers.integrations import is_deepspeed_zero3_enabled
from arguments import ModelArguments, DataArguments, ModelTrainingArguments as TrainingArguments
from modeling import DistillRerankerModel
from data import HFRerankerTrainDataset, RerankerTrainDataset, RerankerTrainCollator
from trainer import RerankerTrainer, save_zero3_peft_adapter

logger = logging.getLogger(__name__)


def _hash_trainable_parameters(model) -> tuple[str, list[str], int]:
    """Return a stable SHA256 fingerprint without changing any RNG state."""
    digest = hashlib.sha256()
    names = []
    numel = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        tensor = parameter.detach().cpu().contiguous()
        names.append(name)
        numel += tensor.numel()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest(), names, numel

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            f"Use --overwrite_output_dir to overwrite."
        )

    # Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bit: fp16=%s bf16=%s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
        training_args.bf16,
    )

    logger.info("Training args: %s", training_args)
    logger.info("Model args: %s", model_args)
    logger.info("Data args: %s", data_args)
    logger.info("Teacher: %s | T=%s | alpha=%s | loss_type=%s",
                model_args.teacher_model_name_or_path, model_args.temperature, model_args.alpha, model_args.loss_type)

    set_seed(training_args.seed)
    # Make CUDA/PyTorch operations reproducible across repeated runs with the
    # same data and worker configuration.  This must be enabled before model
    # construction and DataLoader creation.
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(
        "REPRO seed=%s data_seed=%s local_rank=%s pythonhashseed=%s "
        "dataloader_workers=%s full_determinism=%s",
        training_args.seed,
        training_args.data_seed,
        training_args.local_rank,
        os.environ.get("PYTHONHASHSEED", "<unset>"),
        training_args.dataloader_num_workers,
        training_args.full_determinism,
    )
    python_rng_hash = hashlib.sha256(
        repr(random.getstate()).encode("utf-8")
    ).hexdigest()
    numpy_rng_hash = hashlib.sha256(
        np.random.get_state()[1].tobytes()
    ).hexdigest()
    torch_rng_hash = hashlib.sha256(
        torch.random.get_rng_state().numpy().tobytes()
    ).hexdigest()
    cuda_rng_hash = "n/a"
    if torch.cuda.is_available():
        cuda_rng_hash = hashlib.sha256(
            torch.cuda.get_rng_state().cpu().numpy().tobytes()
        ).hexdigest()
    logger.info(
        "REPRO rng_state_sha256 python=%s numpy=%s torch=%s cuda=%s",
        python_rng_hash,
        numpy_rng_hash,
        torch_rng_hash,
        cuda_rng_hash,
    )
    logger.info(
        "REPRO torch_initial_seed=%s cuda_initial_seed=%s "
        "deterministic_algorithms=%s cudnn_deterministic=%s cudnn_benchmark=%s",
        torch.initial_seed(),
        torch.cuda.initial_seed() if torch.cuda.is_available() else "n/a",
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    logger.info(
        "REPRO optimization_config lr=%s per_device_batch=%s grad_accum=%s "
        "max_steps=%s scheduler=%s warmup_ratio=%s warmup_steps=%s",
        training_args.learning_rate,
        training_args.per_device_train_batch_size,
        training_args.gradient_accumulation_steps,
        training_args.max_steps,
        training_args.lr_scheduler_type,
        training_args.warmup_ratio,
        training_args.warmup_steps,
    )

    # Tokenizers
    tokenizer_student = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )
    tokenizer_teacher = AutoTokenizer.from_pretrained(model_args.teacher_model_name_or_path, cache_dir=model_args.cache_dir)

    # pad/bos safety
    if tokenizer_student.pad_token is None:
        tokenizer_student.pad_token = tokenizer_student.eos_token
    if tokenizer_teacher.pad_token is None:
        tokenizer_teacher.pad_token = tokenizer_teacher.eos_token
    tokenizer_student.padding_side = "right"
    tokenizer_teacher.padding_side = "right"

    # Build model (student LoRA + frozen teacher)
    model = DistillRerankerModel.build(
        model_args=model_args,
        train_args=training_args,
        student_pad_token_id=(
            tokenizer_student.pad_token_id
        ),
        teacher_pad_token_id=(
            tokenizer_teacher.pad_token_id
        ),
        temperature=model_args.temperature,
        alpha=model_args.alpha,
        cache_dir=model_args.cache_dir,
    )
    parameter_hash, trainable_names, trainable_numel = _hash_trainable_parameters(model)
    logger.info("REPRO trainable_parameter_names=%s", trainable_names)
    logger.info(
        "REPRO initial_trainable_parameter_sha256=%s tensors=%d numel=%d",
        parameter_hash,
        len(trainable_names),
        trainable_numel,
    )

    # 训练时不需要生成 KV cache
    model.student.config.use_cache = False
    model.teacher.config.use_cache = False
   
    if (
        model.student.config.pad_token_id
        != tokenizer_student.pad_token_id
    ):
        raise ValueError(
            "Student model/tokenizer pad_token_id mismatch: "
            f"model={model.student.config.pad_token_id}, "
            f"tokenizer={tokenizer_student.pad_token_id}"
        )

    if (
        model.teacher.config.pad_token_id
        != tokenizer_teacher.pad_token_id
    ):
        raise ValueError(
            "Teacher model/tokenizer pad_token_id mismatch: "
            f"model={model.teacher.config.pad_token_id}, "
            f"tokenizer={tokenizer_teacher.pad_token_id}"
        )

    logger.info(
        "PAD alignment: student=%s, teacher=%s",
        tokenizer_student.pad_token_id,
        tokenizer_teacher.pad_token_id,
    )
    # Dataset
    hf_ds = HFRerankerTrainDataset(
        tokenizer_student=tokenizer_student,
        tokenizer_teacher=tokenizer_teacher,
        data_args=data_args,
        cache_dir=data_args.data_cache_dir or model_args.cache_dir,
    ).process()

    train_dataset = RerankerTrainDataset(
        data_args=data_args,
        dataset=hf_ds,
        tokenizer_student=tokenizer_student,
        tokenizer_teacher=tokenizer_teacher
    )

    if training_args.local_rank > 0:
        torch.distributed.barrier()
    if training_args.local_rank == 0:
        torch.distributed.barrier()

    # Trainer
    trainer = RerankerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=RerankerTrainCollator(
            tokenizer_student=tokenizer_student,
            tokenizer_teacher=tokenizer_teacher,
            max_p_len=data_args.p_max_len,
            max_q_len=data_args.q_max_len
        ),
    )

    # HF Trainer expects model attribute on dataset sometimes (for gradient cache patterns, etc.)
    train_dataset.trainer = trainer

    # HF 4.44+ sometimes adds gradient_checkpointing_kwargs which DS doesn't like; remove if present
    if hasattr(trainer.args, "gradient_checkpointing_kwargs"):
        delattr(trainer.args, "gradient_checkpointing_kwargs")

    # Train
    trainer.train()
    if is_deepspeed_zero3_enabled():
        save_zero3_peft_adapter(
            model,
            training_args.output_dir,
        )
    else:
        trainer.save_model()

    if trainer.is_world_process_zero():
        tokenizer_student.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()
