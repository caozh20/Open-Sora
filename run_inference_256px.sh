#!/bin/bash
# 专为256px分辨率设计的模型并行推理脚本

# 检测可用GPU数量
gpu_count=$(nvidia-smi --list-gpus | wc -l)
# 获取提示词，默认为"raining, sea"
prompt="${1:-raining, sea}"
# 固定使用256px配置
config="configs/diffusion/inference/t2i2v_256px.py"

echo "====================================================="
echo "  Open-Sora 256px 模型并行推理"
echo "====================================================="
echo "使用 $gpu_count 个GPU进行模型并行推理"
echo "配置文件: $config (固定使用256px配置)"
echo "提示词: $prompt"
echo "====================================================="

# 添加当前目录到PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 运行推理 - 使用256px配置
torchrun --nproc_per_node=$gpu_count --standalone scripts/diffusion/inference.py $config --save-dir samples --prompt "$prompt"

# 显示运行结果
if [ $? -eq 0 ]; then
    echo "====================================================="
    echo "推理成功完成！结果保存在 samples 目录中"
    echo "查看生成的视频: samples/video_256px/"
    echo "====================================================="
else
    echo "====================================================="
    echo "推理过程中出现错误"
    echo "====================================================="
fi 