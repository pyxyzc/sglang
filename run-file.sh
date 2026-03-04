#!/bin/bash

cd /sgl-workspace/file/
rm -rf *

cd /sgl-workspace/sglang/
# 模型路径（改成你本地的，比如 /data/models/Llama-2-7b-hf）
MODEL_PATH=/home/models/QwQ-32B

# 端口号（可以改，比如 30000）
PORT=30000
# 卡数量
TP=2

export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/sgl-workspace/file
# 只使用
export CUDA_VISIBLE_DEVICES=2,3

# 启动命令
python python/sglang/launch_server.py \
    --model-path $MODEL_PATH \
    --page-size 128 \
    --tp $TP \
    --port $PORT \
    --enable-hierarchical-cache \
    --hicache-write-policy write_through \
    --hicache-storage-backend file \
    --hicache-storage-prefetch-policy wait_complete

