#!/bin/bash

cd /sgl-workspace/kvcache/
rm -rf *

cd /sgl-workspace/sglang/
# 模型路径（改成你本地的，比如 /data/models/Llama-2-7b-hf）
MODEL_PATH=/home/models/QwQ-32B

# 端口号（可以改，比如 30000）
PORT=30000
# 卡数量
TP=2

HICACHE_CONFIG='{"kv_connector_extra_config":{"ucm_connector_name":"UcmPipelineStore","ucm_connector_config":{"storage_backends":"/sgl-workspace/kvcache"}}}'

# 只使用
export CUDA_VISIBLE_DEVICES=2,3

# 启动命令
python python/sglang/launch_server.py \
    --model-path $MODEL_PATH \
    --page-size 128 \
    --tp $TP \
    --port $PORT \
    --enable-hierarchical-cache \
    --hicache-mem-layout page_first \
    --hicache-write-policy write_through \
    --hicache-storage-backend unifiedcache \
    --hicache-storage-prefetch-policy wait_complete \
    --hicache-storage-backend-extra-config "$HICACHE_CONFIG"
