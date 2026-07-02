"""
准备训练数据 v2：
1. 转换懂世界格式 (instruction/input/output → system/prompt/response)
2. 按长度过滤（<=4096 tok 估算，即 <=12288 字符）+ 抽样 30000 条
3. 从所有训练数据中拆分 5% 作为测试集（eval）
4. 输出到 dataset_v2/ 目录

用法：python prepare_data_v2.py
"""
import json
import os
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)

SRC_DS = Path("/home/yuanyi/ZDM/worldrec/dataset")
SRC_EXT = Path("/home/yuanyi/ZDM/worldrec/dataset_expend")
DST = Path("/home/yuanyi/ZDM/worldrec/dataset_v2")
DST.mkdir(parents=True, exist_ok=True)

# char 长度阈值（token 估算 × 3，4096 tok ≈ 12288 char）
MAX_CHAR = 12288
WORLD_SAMPLE = 30000
EVAL_RATIO = 0.05  # 5%


def load_jsonl(path):
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            items = d if isinstance(d, list) else [d]
            samples.extend(items)
    return samples


def convert_world(sample):
    """转换懂世界格式: instruction/input/output/history → system/prompt/response"""
    instr = sample.get("instruction", "").strip()
    inp = sample.get("input", "").strip()
    out = sample.get("output", "").strip()
    hist = sample.get("history", [])

    # instruction 作为 system，input 作为 prompt
    system = instr if instr else ""
    prompt = inp
    response = out

    # 如果有 history，拼接到 prompt 前面（简化处理，只保留单轮）
    # history 格式: [[user1, assistant1], [user2, assistant2], ...]
    if hist:
        history_text = ""
        for h_user, h_assistant in hist:
            history_text += f"Previous User: {h_user}\nPrevious Assistant: {h_assistant}\n\n"
        prompt = history_text + prompt

    return {"system": system, "prompt": prompt, "response": response}


def char_len(s):
    return len(s.get("system", "")) + len(s.get("prompt", "")) + len(s.get("response", ""))


# ========== 1. 加载并准备各类数据 ==========

print("=" * 60)
print("准备训练数据 v2")
print("=" * 60)

all_train = []  # 训练集
all_eval = []   # 测试集
category_stats = defaultdict(lambda: {"train": 0, "eval": 0})

# --- 原始 dataset（懂推荐、懂物料、懂用户）---
raw_files = {
    "懂推荐": [SRC_DS / f"懂推荐{i}.jsonl" for i in range(1, 5)],
    "懂物料": [SRC_DS / f"懂物料part{i}.jsonl" for i in range(1, 8)],
    "懂用户": [SRC_DS / "懂用户.jsonl"],
}

for cat, files in raw_files.items():
    samples = []
    for fp in files:
        if fp.exists():
            samples.extend(load_jsonl(fp))
    # 按长度过滤
    filtered = [s for s in samples if char_len(s) <= MAX_CHAR]
    # 拆分 train/eval
    random.shuffle(filtered)
    n_eval = max(1, int(len(filtered) * EVAL_RATIO))
    eval_samples = filtered[:n_eval]
    train_samples = filtered[n_eval:]
    all_train.extend(train_samples)
    all_eval.extend(eval_samples)
    category_stats[cat]["train"] = len(train_samples)
    category_stats[cat]["eval"] = len(eval_samples)
    print(f"[{cat}] 原始 {len(samples)} → 过滤后 {len(filtered)} → train {len(train_samples)} / eval {len(eval_samples)}")

# --- 懂世界（从 dataset_expend 转换 + 抽样）---
print(f"\n[懂世界] 加载 + 转换格式...")
world_raw = load_jsonl(SRC_EXT / "懂世界.jsonl")
print(f"  原始: {len(world_raw)}")

world_converted = [convert_world(s) for s in world_raw]
# 按长度过滤
world_filtered = [s for s in world_converted if char_len(s) <= MAX_CHAR and char_len(s) > 50]
print(f"  过滤后 (<= {MAX_CHAR} char): {len(world_filtered)}")

# 抽样
if len(world_filtered) > WORLD_SAMPLE:
    world_sample = random.sample(world_filtered, WORLD_SAMPLE)
else:
    world_sample = world_filtered
print(f"  抽样: {len(world_sample)}")

# 拆分 train/eval
random.shuffle(world_sample)
n_eval = max(1, int(len(world_sample) * EVAL_RATIO))
world_eval = world_sample[:n_eval]
world_train = world_sample[n_eval:]
all_train.extend(world_train)
all_eval.extend(world_eval)
category_stats["懂世界"]["train"] = len(world_train)
category_stats["懂世界"]["eval"] = len(world_eval)
print(f"  train {len(world_train)} / eval {len(world_eval)}")

# ========== 2. 保存 ==========
print("\n" + "=" * 60)
print("保存数据集")
print("=" * 60)

random.shuffle(all_train)
random.shuffle(all_eval)

train_path = DST / "train.jsonl"
eval_path = DST / "eval.jsonl"

with open(train_path, "w", encoding="utf-8") as f:
    for s in all_train:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

with open(eval_path, "w", encoding="utf-8") as f:
    for s in all_eval:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"训练集: {train_path} ({len(all_train)} 条)")
print(f"测试集: {eval_path} ({len(all_eval)} 条)")

print("\n各类别统计:")
print(f"  {'类别':<10} {'train':>8} {'eval':>6}")
print(f"  {'-'*26}")
for cat, stats in category_stats.items():
    print(f"  {cat:<10} {stats['train']:>8} {stats['eval']:>6}")
print(f"  {'合计':<10} {len(all_train):>8} {len(all_eval):>6}")

# 保存统计信息
stats_path = DST / "stats.json"
with open(stats_path, "w", encoding="utf-8") as f:
    json.dump({
        "total_train": len(all_train),
        "total_eval": len(all_eval),
        "categories": dict(category_stats),
        "max_char": MAX_CHAR,
        "world_sample_size": WORLD_SAMPLE,
        "eval_ratio": EVAL_RATIO,
    }, f, ensure_ascii=False, indent=2)
print(f"\n统计: {stats_path}")
