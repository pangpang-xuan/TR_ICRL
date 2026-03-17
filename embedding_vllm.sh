#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

model_path="models/huggingface.co/Qwen/Qwen3-Embedding-8B"

nohup vllm serve "$model_path" \
    --task embed \
    --gpu_memory_utilization 0.95 \
    --tensor_parallel_size 1 \
    --max_num_seqs 20480 \
    --seed 42 \
    --enforce-eager \
    > outputs/reasoning.log 2>&1 &

# bash embedding_vllm.sh