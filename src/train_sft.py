"""
OneReason-0.8B SFT LoRA 训练脚本

适用于快手 LLM-Rec 挑战赛 2026，基于 OneReason-0.8B-pretrain-competition 模型。
支持 RTX 3080 10GB（单卡/双卡），使用 LoRA + gradient checkpointing。

数据集分布（粗估 token 数）：
  懂推荐 1-4 : 19204 条, avg~4800, p95~8000, max~10865  → 推荐 max_seq_length >= 8192
  懂物料 1-7 : 10384 条, avg~150,  p95~200,  max~245    → 极短，注意配比
  懂用户     :  2892 条, avg~7155, p95~11002, max~15441 → 最长，需上调采样权重
"""

import argparse
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

# Single-GPU: disable distributed init before importing torch/accelerate
if "LOCAL_RANK" not in os.environ:
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")

# Windows: force gloo backend (no NCCL available)
os.environ.setdefault("TORCH_DISTRIBUTED_BACKEND", "gloo")

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


def resolve_attn_implementation(requested: str) -> str:
    if requested != "auto":
        return requested
    return "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"


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
    parser.add_argument(
        "--sample_weights",
        type=str,
        default=None,
        help='JSON string of category weights, e.g. \'{"懂推荐":1.0,"懂物料":1.5,"懂用户":3.0}\'',
    )
    parser.add_argument(
        "--filter_long_samples",
        action="store_true",
        default=False,
        help="Hard-filter samples longer than max_seq_length (default: truncate)",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="Single JSONL train file (overrides --data_dir). Each line is a dict with system/prompt/response.",
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        default=None,
        help="Single JSONL eval file for held-out evaluation during training.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Evaluate every N steps (only if --eval_file provided).",
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
    parser.add_argument("--max_seq_length", type=int, default=8192, help="Max sequence length (懂用户 p95~11002, 懂推荐 p95~8000)")
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


def apply_sample_weights(dataset, weights_json: str):
    """按类别对数据集进行加权重采样（与 train_full.py 保持一致）。"""
    weights = json.loads(weights_json)
    random.seed(42)

    samples = list(dataset)
    weights_list = []
    for s in samples:
        combined = s.get("system", "") + s.get("prompt", "") + s.get("response", "")
        w = 1.0
        for cat, cw in weights.items():
            if cat in combined:
                w = cw
                break
        weights_list.append(w)

    total_weight = sum(weights_list)
    probs = [w / total_weight for w in weights_list]
    indices = random.choices(range(len(samples)), weights=probs, k=len(samples))
    return Dataset.from_list([samples[i] for i in indices])


def load_single_jsonl(file_path: str) -> Dataset:
    """从单个 JSONL 文件加载数据（每行一个 dict）。"""
    from datasets import Dataset
    samples = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if isinstance(d, list):
                samples.extend(d)
            elif isinstance(d, dict):
                samples.append(d)
    print(f"  Loaded {file_path}: {len(samples)} samples")
    return Dataset.from_list(samples)


def prepare_dataset(args, tokenizer) -> Dataset:
    """加载并预处理训练数据集。"""
    print("=" * 60)
    print("Loading training dataset...")

    # 优先使用 --train_file（单文件模式）
    if args.train_file:
        raw_dataset = load_single_jsonl(args.train_file)
    else:
        # 目录模式
        data_dir = args.data_dir
        if not os.path.isdir(data_dir):
            data_dir = os.path.join(os.getcwd(), data_dir)
        raw_dataset = load_dataset_from_dir(
            data_dir=data_dir,
            categories=args.categories,
            max_samples=args.max_samples,
        )

    print(f"Raw dataset size: {len(raw_dataset)}")

    # 应用数据配比加权
    if args.sample_weights:
        print(f"Applying sample weights: {args.sample_weights}")
        raw_dataset = apply_sample_weights(raw_dataset, args.sample_weights)
        print(f"Resampled dataset size: {len(raw_dataset)}")

    # 格式化
    def tokenize_fn(sample):
        return format_chat_sample(sample, tokenizer, args.max_seq_length)

    print("Tokenizing dataset...")
    processed = raw_dataset.map(
        tokenize_fn,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing",
        num_proc=1,
    )

    # 过滤掉过长的样本
    if args.filter_long_samples:
        before = len(processed)
        processed = processed.filter(
            lambda x: len(x["input_ids"]) <= args.max_seq_length,
            desc="Filtering long samples",
        )
        print(f"  Filtered {before - len(processed)} samples > {args.max_seq_length} tokens")

    processed = processed.filter(
        lambda x: len(x["input_ids"]) > 10,
        desc="Filtering",
    )

    print(f"Processed dataset size: {len(processed)}")
    print("=" * 60)
    return processed


def prepare_eval_dataset(args, tokenizer) -> Dataset:
    """加载并预处理 eval 数据集（不应用 sample_weights）。"""
    print("=" * 60)
    print("Loading eval dataset...")
    raw_dataset = load_single_jsonl(args.eval_file)
    print(f"Raw eval size: {len(raw_dataset)}")

    def tokenize_fn(sample):
        return format_chat_sample(sample, tokenizer, args.max_seq_length)

    print("Tokenizing eval dataset...")
    processed = raw_dataset.map(
        tokenize_fn,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing eval",
        num_proc=1,
    )

    if args.filter_long_samples:
        before = len(processed)
        processed = processed.filter(
            lambda x: len(x["input_ids"]) <= args.max_seq_length,
            desc="Filtering long eval samples",
        )
        print(f"  Filtered {before - len(processed)} eval samples > {args.max_seq_length} tokens")

    processed = processed.filter(
        lambda x: len(x["input_ids"]) > 10,
        desc="Filtering eval",
    )

    print(f"Processed eval size: {len(processed)}")
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
    # DDP (torchrun) 下每个进程只绑定自己的 GPU；单卡时 fallback 到 auto
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank >= 0:
        device_map = {"": local_rank}
    else:
        device_map = "auto"

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation=resolve_attn_implementation("auto"),
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

    # 4.5 准备 eval 数据集（可选）
    eval_dataset = None
    if args.eval_file:
        eval_dataset = prepare_eval_dataset(args, tokenizer)

    # 5. 训练参数
    training_args_kwargs = dict(
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
        report_to="none",
        save_strategy="steps",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
    )
    if eval_dataset is not None:
        training_args_kwargs["eval_strategy"] = "steps"
        training_args_kwargs["eval_steps"] = args.eval_steps
        training_args_kwargs["per_device_eval_batch_size"] = args.batch_size

    training_args = TrainingArguments(**training_args_kwargs)

    # 6. 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
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
