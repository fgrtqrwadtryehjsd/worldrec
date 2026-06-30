"""
数据集分析脚本

分析数据集各维度的样本分布、token 长度、内容模式，
为训练数据配比提供决策依据。
"""

import json
import glob
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def analyze_dataset(data_dir="dataset", model_path="D:/models/OneReason-0.8B-pretrain-competition"):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
    if not files:
        print(f"No JSONL files found in {data_dir}")
        return

    categories = {
        "懂推荐": {"samples": 0, "token_lens": [], "subtypes": Counter()},
        "懂物料": {"samples": 0, "token_lens": [], "subtypes": Counter()},
        "懂用户": {"samples": 0, "token_lens": [], "subtypes": Counter()},
    }

    for fp in files:
        fname = os.path.basename(fp)
        cat = None
        for key in categories:
            if fname.startswith(key):
                cat = key
                break
        if cat is None:
            continue

        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                samples = data if isinstance(data, list) else [data]

                for s in samples:
                    categories[cat]["samples"] += 1

                    # Token 长度
                    text = s.get("system", "") + s.get("prompt", "") + s.get("response", "")
                    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                    categories[cat]["token_lens"].append(len(ids))

                    # 子类型分析（基于系统提示词）
                    sys_prompt = s.get("system", "")[:30]
                    categories[cat]["subtypes"][sys_prompt] += 1

    # 输出分析结果
    print("=" * 70)
    print("数据集分析报告")
    print("=" * 70)

    total_samples = sum(c["samples"] for c in categories.values())
    print(f"\n总样本数: {total_samples}\n")

    for cat, info in categories.items():
        n = info["samples"]
        lens = sorted(info["token_lens"])
        if n == 0:
            continue

        p50 = lens[n // 2]
        p90 = lens[int(n * 0.9)]
        p99 = lens[int(n * 0.99)]

        print(f"{'─' * 50}")
        print(f"【{cat}】 {n} 样本 ({n * 100 // total_samples}% of total)")
        print(f"{'─' * 50}")
        print(f"  Token 长度:")
        print(f"    min={min(lens)}, max={max(lens)}, avg={sum(lens) // n}")
        print(f"    P50={p50}, P90={p90}, P99={p99}")
        print(f"  超长样本占比:")
        for threshold in [1024, 2048, 4096, 8192]:
            over = sum(1 for l in lens if l > threshold)
            print(f"    >{threshold}: {over} ({over * 100 // n}%)")
        print(f"  子类型分布 (top 5):")
        for subtype, count in info["subtypes"].most_common(5):
            print(f"    [{count}x] {subtype}...")
        print()

    # 配比建议
    print("=" * 70)
    print("数据配比建议")
    print("=" * 70)
    print()
    print("基于评测四维度（懂物料、懂用户、懂推荐、懂世界）和数据分析：")
    print()
    print("┌──────────┬──────────┬──────────┬──────────────────────────────┐")
    print("│ 维度     │ 样本数   │ 占比     │ 建议                         │")
    print("├──────────┼──────────┼──────────┼──────────────────────────────┤")
    print("│ 懂推荐   │ ~19,200  │ 59%      │ 核心，保持高权重             │")
    print("│ 懂物料   │ ~10,400  │ 32%      │ 物料理解，保持中权重         │")
    print("│ 懂用户   │ ~2,900   │  9%      │ 少量但重要，建议上采样 2-3x │")
    print("│ 懂世界   │ 0        │  0%      │ 缺失，建议引入外部通识数据   │")
    print("└──────────┴──────────┴──────────┴──────────────────────────────┘")
    print()
    print("推荐配比方案:")
    print()
    print("方案 A（均衡型，推荐初期）:")
    print('  --sample_weights \'{"懂推荐": 1.0, "懂物料": 1.0, "懂用户": 3.0}\'')
    print("  → 懂用户上采样 3x，使各维度 token 总量更均衡")
    print()
    print("方案 B（推荐优先型）:")
    print('  --sample_weights \'{"懂推荐": 2.0, "懂物料": 1.0, "懂用户": 4.0}\'')
    print("  → 加大推荐和用户理解权重")
    print()
    print("方案 C（物料优先型）:")
    print('  --sample_weights \'{"懂推荐": 1.0, "懂物料": 2.0, "懂用户": 3.0}\'')
    print("  → 加大物料理解权重，适合需要强 itemic token 能力时")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset")
    parser.add_argument("--model_path", default="D:/models/OneReason-0.8B-pretrain-competition")
    args = parser.parse_args()

    analyze_dataset(args.data_dir, args.model_path)
