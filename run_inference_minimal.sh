#!/bin/bash
# 极低内存消耗版本的模型并行推理脚本 - 适用于内存受限的系统

# 固定使用2个GPU
gpu_count=2
# 获取提示词，默认为"raining, sea"
prompt="${1:-raining, sea}"
# 强制使用256px配置以减少内存需求
config="configs/diffusion/inference/t2i2v_256px.py"

echo "====================================================="
echo "  Open-Sora 模型并行推理 (极低内存版)"
echo "====================================================="
echo "固定使用 $gpu_count 个GPU进行最小资源消耗的推理"
echo "配置文件: $config (256px低分辨率)"
echo "提示词: $prompt"
echo "====================================================="

# 添加当前目录到PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 确保目录存在
mkdir -p samples

# 设置内存优化参数
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# 运行前清理 GPU 内存
echo "清理GPU内存缓存..."
nvidia-smi --gpu-reset 2>/dev/null || echo "无法重置GPU，继续执行..."

# 运行推理 - 使用自定义参数
echo "启动极简推理过程..."
echo "提示: 该模式专为内存严重受限的系统设计"

# 设置小帧数和更少的采样步骤，降低内存需求
torchrun --nproc_per_node=$gpu_count --standalone scripts/diffusion/inference.py $config \
  --save-dir samples \
  --prompt "$prompt" \
  --sampling_option.num_frames 21 \
  --sampling_option.num_steps 30

# 显示运行结果
result=$?
if [ $result -eq 0 ]; then
    echo "====================================================="
    echo "✓ 推理成功完成！"
    echo "结果保存在 samples 目录中"
    echo "====================================================="
else
    echo "====================================================="
    echo "✗ 推理过程中出现错误 (退出代码: $result)"
    echo ""
    echo "可能的解决方法:"
    echo "1. 尝试使用单个GPU: 将gpu_count修改为1"
    echo "2. 增加系统交换空间: sudo fallocate -l 24G /swapfile"
    echo "3. 运行前彻底重启电脑，关闭所有其他程序"
    echo "====================================================="
    exit $result
fi 