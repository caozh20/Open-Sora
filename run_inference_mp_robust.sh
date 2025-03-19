#!/bin/bash
# 自动检测可用的GPU数量并使用它们进行模型并行推理
gpu_count=$(nvidia-smi --list-gpus | wc -l)
prompt="${1:-raining, sea}"
echo "使用 $gpu_count 个GPU进行模型并行推理"
echo "提示词: $prompt"
# 设置捕获错误信号
cleanup() {
  echo "接收到中断信号，清理资源..."
  # 杀死所有相关进程
  pkill -P $$
  exit 1
}
# 注册信号处理函数
trap cleanup SIGINT SIGTERM
# 运行推理命令
torchrun --nproc_per_node=$gpu_count --standalone scripts/diffusion/inference.py configs/diffusion/inference/t2i2v_768px.py --save-dir samples --prompt "$prompt"
# 检查程序退出状态
exit_code=$?
if [ $exit_code -ne 0 ]; then
  echo "推理失败，退出代码: $exit_code"
  exit $exit_code
else
  echo "推理成功完成！"
  echo "结果保存在 samples 目录中"
fi
