#!/bin/bash
# 自动检测可用的GPU数量并使用它们进行模型并行推理

# 激活虚拟环境
if [ -n "$CONDA_PREFIX" ]; then
    # 如果已经在conda环境中，检查是否是正确的环境
    if [ "$(basename $CONDA_PREFIX)" != "opensora" ]; then
        echo "当前激活的环境不是opensora，尝试激活opensora环境..."
        conda activate opensora || {
            echo "错误: 无法激活opensora环境，请确保环境存在或手动激活后再运行"
            echo "可以运行: conda activate opensora"
            exit 1
        }
    fi
else
    # 如果没有激活conda环境，尝试激活opensora
    echo "尝试激活opensora环境..."
    # 检查conda是否在PATH中
    if command -v conda &> /dev/null; then
        eval "$(conda shell.bash hook)"
        conda activate opensora || {
            echo "错误: 无法激活opensora环境，请确保环境存在或手动激活后再运行"
            echo "可以运行: conda activate opensora"
            exit 1
        }
    else
        echo "警告: 找不到conda命令，请手动激活opensora环境后再运行此脚本"
        echo "可以运行: conda activate opensora"
        exit 1
    fi
fi

# 检测可用GPU数量
gpu_count=$(nvidia-smi --list-gpus | wc -l)
# 获取提示词，默认为"raining, sea"
prompt="${1:-raining, sea}"
# 获取配置文件，默认为t2i2v_768px.py
config="${2:-configs/diffusion/inference/t2i2v_768px.py}"

echo "====================================================="
echo "  Open-Sora 模型并行推理 (opensora环境)"
echo "====================================================="
echo "使用 $gpu_count 个GPU进行模型并行推理"
echo "配置文件: $config"
echo "提示词: $prompt"
echo "====================================================="

# 检查环境和依赖
if ! command -v torchrun &> /dev/null; then
    echo "错误: 找不到torchrun命令，请确保PyTorch正确安装在opensora环境中"
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

# 添加当前目录到PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 运行推理 - 注意脚本现在已经有了防止tp_size>world_size的保护措施
torchrun --nproc_per_node=$gpu_count --standalone scripts/diffusion/inference.py $config --save-dir samples --prompt "$prompt"

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