#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

model_path="models/huggingface.co/Qwen/Qwen2.5-7B-Instruct"

vllm serve "$model_path" \
    --gpu_memory_utilization 0.88 \
    --tensor_parallel_size 2 \
    --port 8848 \
    --trust_remote_code \
    --max_num_seqs 512



# bash reasoning_vllm.sh