#!/bin/bash
# 自动检测可用的GPU数量并使用它们进行模型并行推理

# 检测可用GPU数量
gpu_count=$(nvidia-smi --list-gpus | wc -l)
# 获取提示词，默认为"raining, sea"
prompt="${1:-raining, sea}"

echo "====================================================="
echo "  Open-Sora 模型并行推理"
echo "====================================================="
echo "使用 $gpu_count 个GPU进行模型并行推理"
echo "提示词: $prompt"
echo "====================================================="

# 检查环境和依赖
if ! command -v torchrun &> /dev/null; then
    echo "错误: 找不到torchrun命令，请确保PyTorch正确安装"
    exit 1
fi

# 设置信号处理
cleanup() {
    echo "接收到中断信号，正在安全清理资源..."
    pkill -P $$
    exit 1
}

# 注册信号处理
trap cleanup SIGINT SIGTERM

# 运行推理
torchrun --nproc_per_node=$gpu_count --standalone scripts/diffusion/inference.py configs/diffusion/inference/t2i2v_768px.py --save-dir samples --prompt "$prompt"

# 检查运行结果
result=$?
if [ $result -eq 0 ]; then
    echo "====================================================="
    echo "推理成功完成！"
    echo "结果保存在 samples 目录中"
    echo "====================================================="
else
    echo "====================================================="
    echo "推理过程中出现错误，退出代码: $result"
    echo "====================================================="
    exit $result
fi 