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
| 600 | 1.48 | ~1.0 | | 快速下降期 |
| 3000 | 0.61 | ~1.0 | | 持续下降 |
| 5400 | 0.33 | ~1.0 | | 趋于平稳 |
| 6090 | 0.25 | 0.83 | 1.4e-11 | 完成 |

- **总耗时**: 13h01m, 6.16 s/it
- **最终 train_loss**: 0.678 (avg), 末段 ~0.32
- **grad_norm**: 全程稳定 avg=1.084, max=3.277, 无爆炸
- **过拟合判断**: 末段 loss=0.32 偏低，模型对训练数据记忆过深 → 疑似过拟合

### 评测结果

| 指标 | Baseline(未微调) | Exp01 | 变化 |
|------|-----------------|-------|------|
| 总分 | 0.6731 | **0.7256** | +0.0525 |
| 懂物料 | 0.1533 | 0.1226 | -0.0307 ⚠️ |
| 懂用户-1 | 0.0000 | 0.0867 | +0.0867 ✅ |
| 懂用户-2 | 0.0055 | 0.0377 | +0.0322 ✅ |
| 懂推荐-1 | 0.0960 | 0.0488 | -0.0472 ⚠️ |
| 懂推荐-2 | 0.0544 | 0.0955 | +0.0411 ✅ |
| 懂推荐-3 | 0.1330 | 0.1233 | -0.0097 |
| 懂推荐-4 | 0.0900 | 0.0900 | 0 |
| 懂世界 | 0.1409 | 0.1123 | -0.0286 ⚠️ |

**问题分析**:
1. **灾难性遗忘**: 懂物料/懂世界倒退，微调丢了预训练能力
2. **懂世界未训练**: 无训练数据，纯依赖预训练 → 遗忘后分数下降
3. **懂推荐-1 倒退**: 某推荐子域（直播/广告）训练不充分
4. **sample_weights 不合理**: 懂物料权重过高(1.5)，懂推荐过低(1.0)

### 上传文件

- `outputs/exp01_lora_2gpu/final/adapter_model.safetensors` → 实际路径 `outputs/lora_2gpu_3090_v1/final/`
- `outputs/exp01_lora_2gpu/final/adapter_config.json`

---

## Exp02 — 双卡 LoRA + 懂世界 (RTX 3090 ×2, Linux)

| 项目 | 值 |
|------|-----|
| **状态** | 待开始 |
| **开始时间** | 2026-07-02 |
| **脚本** | `scripts/train_lora_2gpu_3090_v2.sh` |
| **模型** | `~/models/OneReason-0.8B-pretrain-competition` |
| **数据** | `dataset_v2/train.jsonl` (53551 条) + `dataset_v2/eval.jsonl` (2817 条) |
| **输出目录** | `outputs/exp02_lora_world/` |

### 数据准备

由 `prepare_data_v2.py` 生成，合并原始数据 + 懂世界抽样：

| 类别 | train | eval | 来源 |
|------|-------|------|------|
| 懂推荐 | 14240 | 749 | dataset/懂推荐1-4 (过滤 <=4096 tok) |
| 懂物料 | 9865 | 519 | dataset/懂物料part1-7 |
| 懂用户 | 946 | 49 | dataset/懂用户 (过滤 <=4096 tok, 仅34%保留) |
| 懂世界 | 28500 | 1500 | dataset_expend/懂世界.jsonl (转换格式+抽样30000) |
| **合计** | **53551** | **2817** | |

注：懂世界原始格式为 `{instruction, input, output, history}`，已转换为 `{system, prompt, response}`。

### 训练配置（vs Exp01 调整）

| 参数 | Exp01 | Exp02 | 调整理由 |
|------|-------|-------|---------|
| max_seq_length | 8192 | **4096** | 提速2x, 配合 filter |
| filter_long_samples | False | **True** | 硬过滤超长, 避免截断信息丢失 |
| lora_r | 64 | **32** | 降低容量, 防过拟合 |
| lora_alpha | 128 | **64** | 保持 alpha=2×r |
| lora_dropout | 0.05 | **0.1** | 增加dropout防过拟合 |
| learning_rate | 2e-4 | **1e-4** | 降LR防遗忘 |
| num_epochs | 3 | **2** | Exp01 loss=0.32过拟合 |
| weight_decay | 0.01 | **0.05** | L2正则防遗忘 |
| sample_weights | 推荐1/物料1.5/用户3 | **推荐2/物料1/用户5/世界1** | 按baseline建议, 懂用户最弱 |
| eval | 无 | **每500步** | 监控eval_loss |
| flash-attn | SDPA | **auto(FA2)** | 已安装 |

### 训练目标

- 预计 ~2.8 s/it, 2 epochs × 3347 steps = 6694 steps, ~5.2h
- 关键观察点：eval_loss 是否随 train_loss 下降（防过拟合）
- 对比 Exp01：懂世界是否提升，懂物料/懂推荐是否止跌

### 评测结果

> 训练完成后在此记录万擎平台评测分数。

| 指标 | Baseline | Exp01 | **Exp02** | 备注/变化(vs Exp01) |
|------|----------|-------|-------|------|
| **总分** | 0.6731 | 0.7256 | **0.7910** | 🚀 大幅提升 **+0.0654** |
| 懂物料 | 0.1533 | 0.1226 | **0.1840** | 🚀 恢复并超越 Baseline (+0.0614) |
| 懂用户-1 | 0.0000 | 0.0867 | **0.0514** | ⚠️ 下降 (-0.0353) |
| 懂用户-2 | 0.0055 | 0.0377 | **0.0394** | 稳定/略升 (+0.0017) |
| 懂推荐-1 | 0.0960 | 0.0488 | **0.0384** | ⚠️ 继续下降 (-0.0104) |
| 懂推荐-2 | 0.0544 | 0.0955 | **0.1088** | 🚀 提升 (+0.0133) |
| 懂推荐-3 | 0.1330 | 0.1233 | **0.1260** | 稳定/略升 (+0.0027) |
| 懂推荐-4 | 0.0900 | 0.0900 | **0.1044** | 🚀 提升 (+0.0144) |
| 懂世界 | 0.1409 | 0.1123 | **0.1337** | 🚀 显著恢复 (+0.0214) |

### 📈 Exp02 结果诊断

1. **整体表现极佳**：总分直接从 `0.725` 跃升至 `0.791`。这证明了我们的**防过拟合策略**（降 r、增 dropout、减 epoch）和**加入懂世界数据**（防遗忘）非常成功。
2. **懂世界与懂物料双双止跌回升**：懂世界恢复到接近 baseline 的水平，懂物料不仅恢复甚至超越了 baseline。
3. **隐患 1：懂用户-1 明显下降**。原因很明显：我们在 Exp02 中将 `max_seq_length` 从 8192 降到了 4096，并开启了 `filter_long_samples`。这导致**65%的优质长序列“懂用户”数据被直接抛弃了**（训练集里只剩不到1000条），因此用户理解能力出现倒退。
4. **隐患 2：懂推荐-1 持续低迷**。该类目的训练数据可能覆盖不足（例如特定类型的推荐：如广告或直播等在原始数据中偏少）。

---

## 🚀 Exp03 — 引入全量扩展数据 + 恢复长序列 (计划)

针对 Exp02 暴露的短板（懂用户截断丢失、推荐子域覆盖不足），提出下一步 Exp03 的优化计划：

1. **引入官方扩展数据 (`_ext`)**：
   - 将 `dataset_expend/懂推荐_ext.jsonl` (2万条)
   - `dataset_expend/懂物料_ext.jsonl` (2万条)
   - `dataset_expend/懂用户_ext.jsonl` (2万条)
   **全部加入训练集**。极大丰富“懂推荐”和“懂用户”的多样性，有望彻底解决“懂推荐-1”持续低迷的问题。
2. **恢复长序列，拯救“懂用户”**：
   - 将 `max_seq_length` 恢复为 **8192**。
   - **取消硬过滤** (`filter_long_samples`)，采用默认的尾部截断。这样即使部分超过 8192 的序列被轻微截断，也能保留绝大部分用户的长序列历史。
3. **保持 Exp02 优秀的超参数**：
   - 继续使用 LoRA r=32, lr=1e-4, dropout=0.1, epochs=2。
   - `sample_weights` 维持 `{"懂推荐": 2.0, "懂物料": 1.0, "懂用户": 5.0, "懂世界": 1.0}`。

---

<!-- 后续实验模板：

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
