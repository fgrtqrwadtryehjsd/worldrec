"""数据加载与预处理工具"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset


def load_jsonl(file_path: str) -> List[Dict]:
    """加载单个 JSONL 文件，每行是一个 JSON 数组。"""
    samples = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if isinstance(data, list):
                samples.extend(data)
            elif isinstance(data, dict):
                samples.append(data)
    return samples


def load_dataset_from_dir(
    data_dir: str,
    categories: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
) -> Dataset:
    """从目录加载所有 JSONL 数据并转为 HuggingFace Dataset。

    Args:
        data_dir: 数据目录路径
        categories: 要加载的类别前缀列表，如 ["懂推荐", "懂物料", "懂用户"]。
                    None 表示加载全部。
        max_samples: 每个文件最大样本数，None 表示不限制。

    Returns:
        HuggingFace Dataset 对象
    """
    data_dir = Path(data_dir)
    all_samples = []

    jsonl_files = sorted(data_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in {data_dir}")

    for file_path in jsonl_files:
        fname = file_path.name

        # 按类别过滤
        if categories is not None:
            if not any(fname.startswith(cat) for cat in categories):
                continue

        samples = load_jsonl(str(file_path))

        if max_samples is not None:
            samples = samples[:max_samples]

        all_samples.extend(samples)
        print(f"  Loaded {fname}: {len(samples)} samples")

    print(f"Total samples: {len(all_samples)}")

    # 转为 Dataset
    dataset = Dataset.from_list(all_samples)
    return dataset


def format_chat_sample(
    sample: Dict,
    tokenizer,
    max_seq_length: int = 2048,
) -> Dict:
    """将单条样本格式化为 chat 模板的 input_ids 和 labels。

    使用 Qwen3 的 chat template 格式：
    <|im_start|>system\n{system}<|im_end|>
    <|im_start|>user\n{prompt}<|im_end|>
    <|im_start|>assistant\n{response}<|im_end|>

    只对 assistant 部分计算 loss。
    """
    messages = [
        {"role": "system", "content": sample.get("system", "")},
        {"role": "user", "content": sample["prompt"]},
        {"role": "assistant", "content": sample["response"]},
    ]

    # 使用 tokenizer 的 chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # 手动构建 labels：只对 response 部分计算 loss
    # 构建无 response 的前缀
    prefix_messages = [
        {"role": "system", "content": sample.get("system", "")},
        {"role": "user", "content": sample["prompt"]},
    ]
    prefix_text = tokenizer.apply_chat_template(
        prefix_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 完整文本和前缀文本分别编码
    full_ids = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
        return_tensors=None,
    )["input_ids"]

    prefix_ids = tokenizer(
        prefix_text,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
        return_tensors=None,
    )["input_ids"]

    # 构建 labels：前缀部分设为 -100，response 部分保留
    labels = [-100] * len(full_ids)
    prefix_len = len(prefix_ids)
    for i in range(prefix_len, len(full_ids)):
        labels[i] = full_ids[i]

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


class SFTDataCollator:
    """SFT 数据整理器，支持动态 padding。"""

    def __init__(self, tokenizer, max_seq_length: int = 2048):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.max_seq_length = max_seq_length

    def __call__(self, batch: List[Dict]) -> Dict:
        max_len = max(len(x["input_ids"]) for x in batch)
        max_len = min(max_len, self.max_seq_length)

        input_ids = []
        attention_mask = []
        labels = []

        for x in batch:
            ids = x["input_ids"][:max_len]
            lbls = x["labels"][:max_len]
            pad_len = max_len - len(ids)

            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
            labels.append(lbls + [-100] * pad_len)

        import torch

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
