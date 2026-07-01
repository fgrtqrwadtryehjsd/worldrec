"""
OneReason-0.8B 全参微调训练脚本

适用于快手 LLM-Rec 挑战赛 2026，基于 OneReason-0.8B-pretrain-competition 模型。
全参微调需要更大显存（建议 >=24GB），支持多卡训练。

数据集分布（粗估 token 数）：
  懂推荐 1-4 : 19204 条, avg~4800, p95~8000, max~10865  → 推荐 max_seq_length >= 8192
  懂物料 1-7 : 10384 条, avg~150,  p95~200,  max~245    → 极短，注意配比
  懂用户     :  2892 条, avg~7155, p95~11002, max~15441 → 最长，需上调采样权重

特性：
  - 全参微调（更新所有参数）
  - 支持 8-bit AdamW 优化器 (bitsandbytes) 以节省显存
  - 支持 DeepSpeed ZeRO Stage 2/3 分布式训练
  - 支持 gradient checkpointing
  - 可选 Flash Attention 2
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_utils import (
    load_dataset_from_dir,
    format_chat_sample,
    SFTDataCollator,
)


def parse_args():
    parser = argparse.ArgumentParser(description="OneReason SFT Full Fine-tuning")

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
        default="outputs/full_v1",
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

    # 数据配比控制
    parser.add_argument(
        "--sample_weights",
        type=str,
        default=None,
        help="JSON string of category weights, e.g. '{\"懂推荐\":1.0,\"懂物料\":1.0,\"懂用户\":3.0}'",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=8192,
        help="Max sequence length (懂用户 p95~11002, 懂推荐 p95~8000, 懂物料 max~245)",
    )
    parser.add_argument(
        "--filter_long_samples",
        action="store_true",
        default=False,
        help="Filter out samples longer than max_seq_length",
    )

    # 训练参数
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device batch size")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation"
    )
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max grad norm")

    # 优化器配置
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw_torch",
        choices=["adamw_torch", "adamw_8bit", "adamw_bnb_8bit"],
        help="Optimizer type (8bit saves memory)",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        choices=["cosine", "linear", "constant", "constant_with_warmup"],
    )

    # 显存优化
    parser.add_argument(
        "--use_gradient_checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing",
    )
    parser.add_argument("--bf16", action="store_true", default=True, help="Use bf16")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
    )

    # 保存与日志
    parser.add_argument("--save_steps", type=int, default=500, help="Save steps")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument(
        "--save_total_limit", type=int, default=3, help="Total save limit"
    )

    # DeepSpeed
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="Path to DeepSpeed config JSON file",
    )

    args = parser.parse_args()
    return args


def apply_sample_weights(dataset, weights_json):
    """按类别对数据集进行加权采样。

    weights_json: JSON string, e.g. '{"懂推荐": 1.0, "懂物料": 1.0, "懂用户": 3.0}'
    """
    import json
    import random

    weights = json.loads(weights_json)
    random.seed(42)

    # 为每个样本分配类别权重
    samples = list(dataset)
    weighted_samples = []

    for s in samples:
        # 根据 prompt 内容判断类别
        prompt = s.get("prompt", "")
        response = s.get("response", "")
        combined = prompt + response

        weight = 1.0
        for cat, w in weights.items():
            if cat in combined or cat in s.get("system", ""):
                weight = w
                break
        weighted_samples.append((s, weight))

    # 加权采样
    total = len(weighted_samples)
    weights_list = [w for _, w in weighted_samples]
    # 归一化
    total_weight = sum(weights_list)
    probs = [w / total_weight for w in weights_list]

    # 按概率采样（放回）
    indices = random.choices(range(total), weights=probs, k=total)
    resampled = [weighted_samples[i][0] for i in indices]

    return Dataset.from_list(resampled)


def prepare_dataset(args, tokenizer) -> Dataset:
    """加载并预处理数据集。"""
    print("=" * 60)
    print("Loading dataset...")

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
        num_proc=4,
    )

    # 过滤
    if args.filter_long_samples:
        before = len(processed)
        processed = processed.filter(
            lambda x: len(x["input_ids"]) <= args.max_seq_length,
            desc="Filtering long samples",
        )
        print(f"  Filtered {before - len(processed)} samples > {args.max_seq_length} tokens")

    processed = processed.filter(
        lambda x: len(x["input_ids"]) > 10,
        desc="Filtering empty",
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
        attn_implementation=args.attn_implementation,
    )

    if args.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        print("  Gradient checkpointing enabled")

    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params / 1e9:.2f}B")
    print(f"  Trainable params: {trainable_params / 1e9:.2f}B ({100 * trainable_params / total_params:.1f}%)")

    # 3. 准备数据
    train_dataset = prepare_dataset(args, tokenizer)
    data_collator = SFTDataCollator(tokenizer, max_seq_length=args.max_seq_length)

    # 4. 训练参数
    training_args_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        gradient_checkpointing=args.use_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="tensorboard",
        save_safetensors=True,
        save_strategy="steps",
        dataloader_num_workers=4,
        remove_unused_columns=False,
        optim=args.optim,
        lr_scheduler_type=args.lr_scheduler_type,
        ddp_find_unused_parameters=False,
    )

    if args.deepspeed:
        training_args_kwargs["deepspeed"] = args.deepspeed

    training_args = TrainingArguments(**training_args_kwargs)

    # 5. 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # 6. 训练
    print("\n" + "=" * 60)
    print("Starting FULL fine-tuning...")
    print(f"  Model: {args.model_path}")
    print(f"  Dataset: {len(train_dataset)} samples")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch size: {args.batch_size} x {args.gradient_accumulation_steps} (effective: {args.batch_size * args.gradient_accumulation_steps})")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Max seq length: {args.max_seq_length}")
    print(f"  Optimizer: {args.optim}")
    print(f"  Trainable: {trainable_params / 1e9:.2f}B / {total_params / 1e9:.2f}B")
    print("=" * 60 + "\n")

    trainer.train()

    # 7. 保存最终模型
    final_dir = os.path.join(args.output_dir, "final")
    print(f"\nSaving final model to {final_dir}...")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("Done!")

    # 8. 打印上传提示
    print("\n" + "=" * 60)
    print("训练完成！全参微调上传至万擎平台需要以下文件：")
    print(f"  {final_dir}/model.safetensors")
    print(f"  {final_dir}/model.safetensors.index.json  (如分片)")
    print(f"  {final_dir}/config.json")
    print(f"  {final_dir}/generation_config.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
