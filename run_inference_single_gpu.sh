#!/bin/bash
# 单GPU版本的推理脚本 - 为单GPU系统优化

# 固定使用1个GPU
gpu_count=1
# 获取提示词，默认为"raining, sea"
prompt="${1:-raining, sea}"
# 强制使用256px配置以减少内存需求
config="configs/diffusion/inference/t2i2v_256px.py"

echo "====================================================="
echo "  Open-Sora 单GPU推理"
echo "====================================================="
echo "使用单GPU进行推理 (无模型并行)"
echo "配置文件: $config (256px低分辨率)"
echo "提示词: $prompt"
echo "====================================================="

# 添加当前目录到PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 确保目录存在
mkdir -p samples

# 设置内存优化参数
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# 清理GPU内存
echo "清理GPU内存..."
nvidia-smi --gpu-reset 2>/dev/null || echo "无法重置GPU，继续执行..."

# 运行前额外释放内存
echo "释放系统缓存..."
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || echo "无法释放系统缓存，继续执行..."

# 运行推理 - 使用极简设置
echo "启动单GPU推理进程..."
echo "注意: 需要一张至少有12GB内存的高端GPU"

# 使用最小帧数和最少采样步骤
torchrun --nproc_per_node=$gpu_count --standalone scripts/diffusion/inference.py $config \
  --save-dir samples \
  --prompt "$prompt" \
  --sampling_option.num_frames 17 \
  --sampling_option.num_steps 20 \
  --sampling_option.guidance 5.0 \
  --offload_model True

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
    echo "这个模型可能太大，无法在单GPU上运行。您可以尝试："
    echo "1. 使用高端GPU (如A100, A6000或RTX 4090等)"
    echo "2. 使用更简单的模型或降低分辨率"
    echo "3. 增加系统交换空间和内存"
    echo "====================================================="
    exit $result
fi 