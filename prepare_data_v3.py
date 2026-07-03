"""
准备训练数据 v3：
1. 合并基础 dataset + 全部 dataset_expend 扩展数据（懂世界、懂推荐_ext、懂物料_ext、懂用户_ext）。
2. 懂世界和其它扩展数据需做格式转换。
3. 不再按 4096 硬过滤！为保留懂用户长序列，仅过滤超过 12000 tok（约 36000 字符）的极端异常数据，主要依靠训练时的 max_seq_length=8192 进行自然截断。
4. 拆分 5% 作为测试集（eval）。
"""
import json
import os
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)

SRC_DS = Path("/home/yuanyi/ZDM/worldrec/dataset")
SRC_EXT = Path("/home/yuanyi/ZDM/worldrec/dataset_expend")
DST = Path("/home/yuanyi/ZDM/worldrec/dataset_v3")
DST.mkdir(parents=True, exist_ok=True)

# 放宽字符限制，保留绝大多数长序列 (约 12000 tokens)
MAX_CHAR = 36000 
EVAL_RATIO = 0.05

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

def convert_format(sample):
    """转换扩展数据格式: instruction/input/output/history → system/prompt/response"""
    instr = sample.get("instruction", "").strip()
    inp = sample.get("input", "").strip()
    out = sample.get("output", "").strip()
    hist = sample.get("history", [])

    system = instr if instr else ""
    prompt = inp
    response = out

    if hist:
        history_text = ""
        for h_user, h_assistant in hist:
            history_text += f"Previous User: {h_user}\nPrevious Assistant: {h_assistant}\n\n"
        prompt = history_text + prompt

    return {"system": system, "prompt": prompt, "response": response}

def char_len(s):
    return len(s.get("system", "")) + len(s.get("prompt", "")) + len(s.get("response", ""))

print("=" * 60)
print("准备训练数据 v3 (全量扩展数据 + 宽松截断)")
print("=" * 60)

all_train = []
all_eval = []
category_stats = defaultdict(lambda: {"train": 0, "eval": 0})

# 1. 基础数据
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
    filtered = [s for s in samples if char_len(s) <= MAX_CHAR]
    
    random.shuffle(filtered)
    n_eval = max(1, int(len(filtered) * EVAL_RATIO))
    all_eval.extend(filtered[:n_eval])
    all_train.extend(filtered[n_eval:])
    category_stats[cat]["train"] += len(filtered) - n_eval
    category_stats[cat]["eval"] += n_eval

# 2. 扩展数据 (需要转换格式)
ext_files = {
    "懂推荐_ext": SRC_EXT / "懂推荐_ext.jsonl",
    "懂物料_ext": SRC_EXT / "懂物料_ext.jsonl",
    "懂用户_ext": SRC_EXT / "懂用户_ext.jsonl",
    "懂世界_ext": SRC_EXT / "懂世界.jsonl",
}

for cat, fp in ext_files.items():
    if not fp.exists():
        continue
    raw_samples = load_jsonl(fp)
    converted = [convert_format(s) for s in raw_samples]
    
    # 懂世界_ext 数据量太大（125万），这里依然抽样 30000 条，其他全量
    if cat == "懂世界_ext":
        valid = [s for s in converted if 50 < char_len(s) <= MAX_CHAR]
        if len(valid) > 30000:
            filtered = random.sample(valid, 30000)
        else:
            filtered = valid
        base_cat = "懂世界"
    else:
        filtered = [s for s in converted if char_len(s) <= MAX_CHAR]
        base_cat = cat.replace("_ext", "")

    random.shuffle(filtered)
    n_eval = max(1, int(len(filtered) * EVAL_RATIO))
    all_eval.extend(filtered[:n_eval])
    all_train.extend(filtered[n_eval:])
    category_stats[base_cat]["train"] += len(filtered) - n_eval
    category_stats[base_cat]["eval"] += n_eval
    print(f"[{cat}] 加载转换 {len(raw_samples)} → 过滤后 {len(filtered)}")

# 保存
print("\n保存数据集...")
random.shuffle(all_train)
random.shuffle(all_eval)

train_path = DST / "train.jsonl"
with open(train_path, "w", encoding="utf-8") as f:
    for s in all_train:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

eval_path = DST / "eval.jsonl"
with open(eval_path, "w", encoding="utf-8") as f:
    for s in all_eval:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"训练集: {train_path} ({len(all_train)} 条)")
print(f"测试集: {eval_path} ({len(all_eval)} 条)")
for cat, stats in category_stats.items():
    print(f"  {cat:<10} train:{stats['train']:>6}  eval:{stats['eval']:>6}")
