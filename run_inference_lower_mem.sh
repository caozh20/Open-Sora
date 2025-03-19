#!/bin/bash
# 低内存消耗版本的模型并行推理脚本

# 设置最大使用的GPU数量（通常4-6个较为合适）
MAX_GPUS=4
# 检测可用GPU数量
available_gpus=$(nvidia-smi --list-gpus | wc -l)
# 使用较少的GPU数量
gpu_count=$(( available_gpus > MAX_GPUS ? MAX_GPUS : available_gpus ))
# 获取提示词，默认为"raining, sea"
prompt="${1:-raining, sea}"
# 强制使用256px配置以减少内存需求
config="configs/diffusion/inference/t2i2v_256px.py"

echo "====================================================="
echo "  Open-Sora 模型并行推理 (低内存优化版)"
echo "====================================================="
echo "可用GPU: $available_gpus, 实际使用: $gpu_count"
echo "配置文件: $config (强制使用256px配置以减少内存需求)"
echo "提示词: $prompt"
echo "====================================================="

# 添加当前目录到PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 确保目录存在
mkdir -p samples

# 设置信号处理
cleanup() {
    echo "接收到中断信号，正在清理资源..."
    pkill -P $$
    exit 1
}

# 注册信号处理
trap cleanup SIGINT SIGTERM

# 设置PyTorch内存分配策略
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 运行推理 - 带有错误处理
echo "启动推理过程 (低内存优化模式)..."
echo "内存优化提示: 使用更少的GPU，低分辨率模型和优化的内存分配策略"

torchrun --nproc_per_node=$gpu_count --standalone scripts/diffusion/inference.py $config \
  --save-dir samples \
  --prompt "$prompt"

# 显示运行结果
result=$?
if [ $result -eq 0 ]; then
    echo "====================================================="
    echo "✓ 推理成功完成！"
    echo "结果保存在 samples 目录中"
    echo "查看生成的视频: samples/video_256px/"

    # 列出生成的文件
    if [ -d "samples/video_256px" ]; then
        echo "生成的文件:"
        ls -la samples/video_256px/ | grep -E '.mp4|.gif|.png' | head -n 5
    fi
    
    echo "====================================================="
else
    echo "====================================================="
    echo "✗ 推理过程中出现错误 (退出代码: $result)"
    echo ""
    echo "可能的解决方法:"
    echo "1. 进一步减少GPU使用数量(修改脚本中的MAX_GPUS参数)"
    echo "2. 检查是否有足够的系统内存"
    echo "3. 尝试增加系统交换空间"
    echo "4. 运行前清空GPU内存: nvidia-smi -r"
    echo "====================================================="
    exit $result
fi 