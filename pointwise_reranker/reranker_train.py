# reranker_train.py
import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoTokenizer, HfArgumentParser, set_seed
from transformers.integrations import is_deepspeed_zero3_enabled
from arguments import ModelArguments, DataArguments, ModelTrainingArguments as TrainingArguments
from modeling import DistillRerankerModel
from data import HFRerankerTrainDataset, RerankerTrainDataset, RerankerTrainCollator
from trainer import (
    RerankerTrainer,
    save_zero3_peft_adapter,
    PeriodicAdapterSaveCallback,
    DLAverageEarlyStoppingCallback,
)

logger = logging.getLogger(__name__)

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
        train_n_passages=data_args.train_n_passages,
        temperature=model_args.temperature,
        alpha=model_args.alpha,
        cache_dir=model_args.cache_dir,
    )

    # 训练时不需要生成KV cache
    model.student.config.use_cache = False
    model.teacher.config.use_cache = False

    # Dataset
    # 分布式训练时只允许主进程先构建Arrow缓存。
    # 其他rank等待主进程完成，然后直接读取同一份缓存。
    with training_args.main_process_first(
        desc="tokenize training dataset",
    ):
        hf_ds = HFRerankerTrainDataset(
            tokenizer_student=tokenizer_student,
            tokenizer_teacher=tokenizer_teacher,
            data_args=data_args,
            cache_dir=(
                data_args.data_cache_dir
                or model_args.cache_dir
            ),
        ).process()

    train_dataset = RerankerTrainDataset(
        data_args=data_args,
        dataset=hf_ds,
        tokenizer_student=tokenizer_student,
        tokenizer_teacher=tokenizer_teacher,
    )
    # Trainer
    callbacks = []

    if training_args.early_stop_eval_steps > 0:
        callbacks.append(
            DLAverageEarlyStoppingCallback(
                tokenizer=tokenizer_student,
                eval_steps=training_args.early_stop_eval_steps,
                patience=training_args.early_stop_patience,
                min_delta=training_args.early_stop_min_delta,
                batch_size=training_args.early_stop_eval_batch_size,
                q_max_len=data_args.q_max_len,
                p_max_len=data_args.p_max_len,
                dl19_path=training_args.early_stop_dl19_path,
                dl20_path=training_args.early_stop_dl20_path,
                dl19_qrels=training_args.early_stop_dl19_qrels,
                dl20_qrels=training_args.early_stop_dl20_qrels,
                metric=training_args.early_stop_metric,
                beir_data_dir=training_args.early_stop_beir_data_dir,
                beir_qrels_dir=training_args.early_stop_beir_qrels_dir,
            )
        )
    elif training_args.adapter_save_steps > 0:
        callbacks.append(
            PeriodicAdapterSaveCallback(
                save_steps=training_args.adapter_save_steps,
            )
        )

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
        callbacks=callbacks,
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
