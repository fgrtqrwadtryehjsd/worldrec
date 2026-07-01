# 实验记录

> 每次训练实验在此记录配置、数据、训练日志路径和评测结果。
> 训练日志存放在 `logs/exp{ID}_*/` 目录，模型输出在 `outputs/exp{ID}_*/`。

---

## Exp01 — 双卡 LoRA 基线 (RTX 3080 Ti ×2)

| 项目 | 值 |
|------|-----|
| **状态** | 训练中 |
| **开始时间** | 2026-07-02 01:35 |
| **脚本** | `scripts/windows/run_train_2gpu.bat 01 2048` |
| **模型** | `E:/zdm/models/OneReason-0.8B-pretrain-competition` (OneReason-0.8B) |
| **数据** | `dataset/` (32480 条，12 个 JSONL) |
| **输出目录** | `outputs/exp01_lora_2gpu/` |
| **训练日志** | `logs/exp01_lora_2gpu/train.log` |

### 训练配置

| 参数 | 值 | 备注 |
|------|-----|------|
| 微调方式 | LoRA | r=64, alpha=128, dropout=0.05 |
| target_modules | q,k,v,o,gate,up,down_proj | 全投影层 |
| epochs | 3 | |
| batch_size | 1 (per device) | |
| gradient_accumulation_steps | 8 | effective batch = 2×1×8 = 16 |
| learning_rate | 2e-4 | cosine, warmup_ratio=0.03 |
| **max_seq_length** | **2048** | 原计划 8192，因 12GB 显存 OOM 降至 2048 |
| bf16 | True | |
| gradient_checkpointing | True | use_reentrant=False |
| DDP backend | gloo | Windows 无 NCCL |
| optim | adamw_torch | |
| save_steps | 200 | |
| logging_steps | 10 | |

### 环境

- torch 2.5.1+cu124, transformers 4.57.6, peft 0.19.1, accelerate 1.14.0
- deepspeed 0.14.5, flash-attn 2.7.4.post1 (forward 可用，backward 有 bug 未启用)
- CUDA Toolkit 12.1 (conda 安装在 worldrec 环境)
- GPU: 2× NVIDIA GeForce RTX 3080 Ti (12 GB each)

### Windows 适配说明

原 `scripts/train_lora_2gpu.sh` 为 Linux 设计，Windows 下需以下修改：
1. `TCPStore` 调用加 `use_libuv=False`（PyTorch Windows 版无 libuv，需 patch `static_tcp_rendezvous.py` 和 `rendezvous.py`）
2. DDP 后端强制 `gloo`（Windows 无 NCCL）
3. `dataloader_num_workers=0`（Windows 多进程 spawn 不稳定）
4. `datasets.map(num_proc=1)`（同上）
5. `device_map` 按 `LOCAL_RANK` 绑定单卡（DDP 下不能用 "auto"）
6. `report_to="none"`（禁用 tensorboard，避免目录操作导致崩溃）
7. flash-attn 2.7.4 Windows wheel backward 产生 nan（不可用），改用 `sdpa`
8. 单卡模式需设置 `RANK`/`WORLD_SIZE` 等环境变量避免 accelerate 初始化分布式报错

### 训练过程

| Step | Loss | Grad Norm | LR | 备注 |
|------|------|-----------|-----|------|
| 10 | 2.768 | 3.743 | 9.836e-06 | |

（训练进行中，step 11/6090，~6.5s/step，预计 ~11 小时完成）

### 评测结果

> 训练完成后在此记录万擎平台评测分数。

| 指标 | 值 | 提交时间 | 备注 |
|------|-----|---------|------|
| | | | |

### 上传文件

- `outputs/exp01_lora_2gpu/final/adapter_model.safetensors`
- `outputs/exp01_lora_2gpu/final/adapter_config.json`

---

<!-- 后续实验模板：

## Exp02 — 

| 项目 | 值 |
|------|-----|
| **状态** | 待开始 |
| **开始时间** | |
| **脚本** | `scripts/windows/run_train_2gpu.bat 02` |
| **模型** | |
| **数据** | |
| **输出目录** | `outputs/exp02_lora_2gpu/` |
| **训练日志** | `logs/exp02_lora_2gpu/train.log` |

### 训练配置

| 参数 | 值 | 备注 |
|------|-----|------|
| max_seq_length | | |
| | | |

### 训练过程

| Step | Loss | 备注 |
|------|------|------|
| | | |

### 评测结果

| 指标 | 值 | 提交时间 | 备注 |
|------|-----|---------|------|
| | | | |

-->
