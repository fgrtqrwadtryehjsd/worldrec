"""
OneReason-0.8B SFT LoRA 训练脚本

适用于快手 LLM-Rec 挑战赛 2026，基于 OneReason-0.8B-pretrain-competition 模型。
支持 8GB 显存的 RTX 4060 Laptop GPU，使用 LoRA + gradient checkpointing。
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_utils import (
    load_dataset_from_dir,
    format_chat_sample,
    SFTDataCollator,
)


def parse_args():
    parser = argparse.ArgumentParser(description="OneReason SFT LoRA Training")

    # 模型与数据
    parser.add_argument(
        "--model_path",
        type=str,
        default="D:/models/OneReason-0.8B-pretrain-competition",
        help="Pretrained model path",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset",
        help="Dataset directory containing JSONL files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/lora_v1",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="Categories to load (e.g. 懂推荐 懂物料 懂用户). None = all",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples per file (for debugging)",
    )

    # LoRA 配置
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha")
    parser.add_argument(
        "--lora_dropout", type=float, default=0.05, help="LoRA dropout"
    )
    parser.add_argument(
        "--target_modules",
        type=str,
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        help="LoRA target modules",
    )

    # 训练参数
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=16, help="Gradient accumulation"
    )
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument(
        "--use_gradient_checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing",
    )
    parser.add_argument("--save_steps", type=int, default=500, help="Save steps")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use bf16")

    args = parser.parse_args()
    return args


def prepare_dataset(args, tokenizer) -> Dataset:
    """加载并预处理数据集。"""
    print("=" * 60)
    print("Loading dataset...")

    # 如果数据在子目录 dataset/ 下
    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        data_dir = os.path.join(os.getcwd(), data_dir)

    raw_dataset = load_dataset_from_dir(
        data_dir=data_dir,
        categories=args.categories,
        max_samples=args.max_samples,
    )

    print(f"Raw dataset size: {len(raw_dataset)}")

    # 格式化
    def tokenize_fn(sample):
        return format_chat_sample(sample, tokenizer, args.max_seq_length)

    print("Tokenizing dataset...")
    processed = raw_dataset.map(
        tokenize_fn,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing",
        num_proc=4,
    )

    # 过滤掉过长的样本
    processed = processed.filter(
        lambda x: len(x["input_ids"]) > 10,
        desc="Filtering",
    )

    print(f"Processed dataset size: {len(processed)}")
    print("=" * 60)
    return processed


def main():
    args = parse_args()
    print("Training Configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 60)

    # 1. 加载 tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Vocab size: {len(tokenizer)}")

    # 2. 加载模型
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    if args.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        print("  Gradient checkpointing enabled")

    # 3. 配置 LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. 准备数据
    train_dataset = prepare_dataset(args, tokenizer)
    data_collator = SFTDataCollator(tokenizer, max_seq_length=args.max_seq_length)

    # 5. 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=args.bf16,
        gradient_checkpointing=args.use_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="tensorboard",
        save_safetensors=True,
        save_strategy="steps",
        dataloader_num_workers=2,
        remove_unused_columns=False,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
    )

    # 6. 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # 7. 训练
    print("\n" + "=" * 60)
    print("Starting training...")
    print(f"  Model: {args.model_path}")
    print(f"  Dataset: {len(train_dataset)} samples")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch size: {args.batch_size} x {args.gradient_accumulation_steps} (effective: {args.batch_size * args.gradient_accumulation_steps})")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    print("=" * 60 + "\n")

    trainer.train()

    # 8. 保存最终模型
    final_dir = os.path.join(args.output_dir, "final")
    print(f"\nSaving final model to {final_dir}...")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("Done!")

    # 9. 打印上传提示
    print("\n" + "=" * 60)
    print("训练完成！上传至万擎平台需要以下文件：")
    print(f"  {final_dir}/adapter_model.safetensors")
    print(f"  {final_dir}/adapter_config.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
