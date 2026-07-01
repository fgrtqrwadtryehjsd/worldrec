#!/bin/bash
# 解压数据集
# 用法：bash scripts/prepare_data.sh [DATASET_TAR] [TARGET_DIR]
set -e

DATASET_TAR="${1:-dataset.tar.gz}"
TARGET_DIR="${2:-dataset}"

if [ -d "${TARGET_DIR}" ] && [ "$(ls -A ${TARGET_DIR}/*.jsonl 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "[INFO] ${TARGET_DIR}/ 已存在且包含 jsonl 文件，跳过解压"
    ls -lh "${TARGET_DIR}"/*.jsonl
    exit 0
fi

if [ ! -f "${DATASET_TAR}" ]; then
    echo "[ERROR] 找不到 ${DATASET_TAR}"
    echo "请将 dataset.tar.gz 放到项目根目录后重试"
    exit 1
fi

echo "解压 ${DATASET_TAR} 到 ${TARGET_DIR}/ ..."
mkdir -p "${TARGET_DIR}"
tar -xzf "${DATASET_TAR}" -C "${TARGET_DIR}" --strip-components=1 2>/dev/null || \
    tar -xzf "${DATASET_TAR}" -C "${TARGET_DIR}"

echo ""
echo "数据集文件："
ls -lh "${TARGET_DIR}"/*.jsonl 2>/dev/null || ls -lh "${TARGET_DIR}"
echo "解压完成！"
