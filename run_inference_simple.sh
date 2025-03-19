#!/bin/bash
# 简单版本的模型并行推理脚本 - 假设您已经激活了opensora环境

# 检测可用GPU数量
gpu_count=$(nvidia-smi --list-gpus | wc -l)
# 获取提示词，默认为"raining, sea"
prompt="${1:-raining, sea}"
# 获取配置文件，默认为t2i2v_768px.py
config="${2:-configs/diffusion/inference/t2i2v_768px.py}"

echo "====================================================="
echo "  Open-Sora 模型并行推理 (简化版)"
echo "====================================================="
echo "使用 $gpu_count 个GPU进行模型并行推理"
echo "配置文件: $config"
echo "提示词: $prompt"
echo "====================================================="

# 添加当前目录到PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 运行推理 - 注意脚本现在已经有了防止tp_size>world_size的保护措施
torchrun --nproc_per_node=$gpu_count --standalone scripts/diffusion/inference.py $config --save-dir samples --prompt "$prompt"

# 显示运行结果
if [ $? -eq 0 ]; then
    echo "====================================================="
    echo "推理成功完成！结果保存在 samples 目录中"
    echo "====================================================="
else
    echo "====================================================="
    echo "推理过程中出现错误"
    echo "====================================================="
fi 