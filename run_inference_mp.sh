#!/bin/bash
# 自动检测可用的GPU数量并使用它们进行模型并行推理
gpu_count=$(nvidia-smi --list-gpus | wc -l)
prompt="${1:-raining, sea}"
echo "使用 $gpu_count 个GPU进行模型并行推理"
echo "提示词: $prompt"
torchrun --nproc_per_node=$gpu_count --standalone scripts/diffusion/inference.py configs/diffusion/inference/t2i2v_768px.py --save-dir samples --prompt "$prompt"
