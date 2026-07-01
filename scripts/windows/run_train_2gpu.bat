@echo off
setlocal enabledelayedexpansion

REM ============================================
REM 双卡 LoRA 训练 (2x RTX 3080 Ti 12GB)
REM 用法: scripts\windows\run_train_2gpu.bat [EXP_ID] [MAX_SEQ_LEN]
REM   EXP_ID       实验编号，默认 01（对应 outputs/exp01, logs/exp01）
REM   MAX_SEQ_LEN  最大序列长度，默认 2048
REM 示例: scripts\windows\run_train_2gpu.bat 02 4096
REM ============================================

set "EXP_ID=%~1"
if "%EXP_ID%"=="" set "EXP_ID=01"
set "MAX_SEQ_LEN=%~2"
if "%MAX_SEQ_LEN%"=="" set "MAX_SEQ_LEN=2048"

set "EXP_DIR=outputs\exp%EXP_ID%_lora_2gpu"
set "LOG_DIR=logs\exp%EXP_ID%_lora_2gpu"
set "LOG_FILE=%LOG_DIR%\train.log"

echo ============================================
echo   双卡 LoRA 训练 (2x RTX 3080 Ti 12GB)
echo   Exp ID   : %EXP_ID%
echo   Output   : %EXP_DIR%
echo   Log      : %LOG_FILE%
echo   Seq Len  : %MAX_SEQ_LEN%
echo   Eff batch: 2 x 1 x 8 = 16
echo ============================================

call "E:\Soft\anaconda3\Scripts\activate.bat" worldrec
if errorlevel 1 (
  echo [ERR] conda activate failed
  exit /b 1
)

REM Windows PyTorch torchrun needs libuv disabled for DDP rendezvous
set "USE_LIBUV=0"
REM Reduce CUDA memory fragmentation
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

cd /d "E:\zdm\worldrec\worldrec"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

torchrun ^
    --nproc_per_node=2 ^
    --master_port=29500 ^
    src/train_sft.py ^
        --model_path   "E:/zdm/models/OneReason-0.8B-pretrain-competition" ^
        --data_dir     "dataset" ^
        --output_dir   "%EXP_DIR%" ^
        --lora_r       64 ^
        --lora_alpha   128 ^
        --lora_dropout 0.05 ^
        --learning_rate 2e-4 ^
        --num_epochs   3 ^
        --batch_size   1 ^
        --gradient_accumulation_steps 8 ^
        --max_seq_length %MAX_SEQ_LEN% ^
        --save_steps   200 ^
        --logging_steps 10 ^
        --bf16 > "%LOG_FILE%" 2>&1

echo "TRAIN_EXIT=%ERRORLEVEL%"
endlocal
