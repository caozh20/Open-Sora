# Open-Sora 推理脚本说明

本文档介绍了用于 Open-Sora 项目的各种推理脚本，每个脚本针对不同的硬件配置和内存需求进行了优化。

## 脚本概览

| 脚本名称 | GPU需求 | 分辨率 | 内存要求 | 适用场景 |
|---------|--------|-------|---------|---------|
| `run_inference_simple.sh` | 多GPU (默认全部) | 768px | 高 | 高端多GPU服务器 |
| `run_inference_lower_mem.sh` | 最多4个GPU | 256px | 中等 | 消费级多GPU设备 |
| `run_inference_minimal.sh` | 2个GPU | 256px | 低 | 内存受限的双GPU系统 |
| `run_inference_single_gpu.sh` | 1个GPU | 256px | 低 | 单一高端GPU设备 |

## 详细说明

### 1. `run_inference_simple.sh`

**描述**：标准推理脚本，使用所有可用GPU进行模型并行推理。

**用法**：
```bash
./run_inference_simple.sh "你的提示词" [可选:配置文件路径]
```

**适用场景**：
- 高性能计算集群
- 具有多张高端GPU的服务器 (如8×A100)
- 当需要高分辨率输出时

**内存要求**：
- 每个GPU至少需要24GB内存
- 推荐系统内存≥64GB

### 2. `run_inference_lower_mem.sh`

**描述**：优化后的推理脚本，限制使用最多4个GPU，并降低分辨率以减少内存消耗。

**用法**：
```bash
./run_inference_lower_mem.sh "你的提示词"
```

**适用场景**：
- 消费级工作站
- 多GPU游戏电脑 (如4×RTX 3090/4080等)
- 需要在有限资源下运行模型

**内存要求**：
- 每个GPU至少需要16GB内存
- 推荐系统内存≥32GB

### 3. `run_inference_minimal.sh`

**描述**：极简配置的推理脚本，固定使用2个GPU，配合最低资源设置（更少的帧数和步骤）。

**用法**：
```bash
./run_inference_minimal.sh "你的提示词"
```

**适用场景**：
- 入门级工作站
- 双GPU设置 (如2×RTX 3070等)
- 需要在内存严重受限的环境中运行

**内存要求**：
- 每个GPU至少需要8GB内存
- 推荐系统内存≥16GB

### 4. `run_inference_single_gpu.sh`

**描述**：单GPU版本的推理脚本，针对不支持模型并行的环境优化。

**用法**：
```bash
./run_inference_single_gpu.sh "你的提示词"
```

**适用场景**：
- 单GPU高端电脑
- 云端GPU实例 (如单个A100或T4)
- 资源有限但希望尝试模型的场景

**内存要求**：
- GPU至少需要12GB内存
- 推荐系统内存≥16GB
- 可能需要启用模型卸载到CPU内存功能

## 故障排除

如果在运行任何脚本时遇到内存不足错误：

1. **尝试使用更低资源的脚本**：按照 `simple` → `lower_mem` → `minimal` → `single_gpu` 的顺序尝试。

2. **增加系统交换空间**：
   ```bash
   sudo fallocate -l 32G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

3. **减少批处理大小和其他参数**：可以手动编辑脚本，降低帧数和采样步骤。

4. **检查系统资源**：使用 `nvidia-smi` 和 `free -h` 命令监控资源使用情况。

5. **优化CUDA内存分配**：脚本已配置 `PYTORCH_CUDA_ALLOC_CONF`，但您可能需要根据特定硬件进行调整。

## 自定义脚本

您可以根据自己的硬件配置修改现有脚本。主要可调参数包括：

- `gpu_count`：使用的GPU数量
- 配置文件：选择不同分辨率的配置文件
- 采样参数：修改帧数、步骤数和指导尺度
- 内存优化选项：调整 `PYTORCH_CUDA_ALLOC_CONF` 和 `--offload_model` 参数

希望这些脚本能帮助您在各种硬件配置上成功运行 Open-Sora 模型！ 