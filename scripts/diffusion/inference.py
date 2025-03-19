import os
import time
import warnings
from pprint import pformat

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.distributed as dist
from colossalai.utils import set_seed
from tqdm import tqdm

# 导入补丁并确保它被应用
from opensora.patches.hybrid_plugin_patch import apply_hybrid_plugin_patches
# 确保补丁被应用
_ = apply_hybrid_plugin_patches()

from opensora.acceleration.parallel_states import get_data_parallel_group, get_tensor_parallel_group, set_tensor_parallel_group
from opensora.datasets.dataloader import prepare_dataloader
from opensora.registry import DATASETS, build_module
from opensora.utils.cai import (
    get_booster,
    get_is_saving_process,
    init_inference_environment,
    set_group_size,
)
from opensora.utils.config import parse_alias, parse_configs
from opensora.utils.inference import (
    add_fps_info_to_text,
    add_motion_score_to_text,
    create_tmp_csv,
    modify_option_to_t2i,
    process_and_save,
)
from opensora.utils.logger import create_logger, is_main_process
from opensora.utils.misc import log_cuda_max_memory, to_torch_dtype
from opensora.utils.prompt_refine import refine_prompts
from opensora.utils.sampling import (
    SamplingOption,
    prepare_api,
    prepare_models,
    sanitize_sampling_option,
)


@torch.inference_mode()
def main():
    # ======================================================
    # 1. configs & runtime variables
    # ======================================================
    torch.set_grad_enabled(False)

    # == parse configs ==
    cfg = parse_configs()
    cfg = parse_alias(cfg)

    # == device and dtype ==
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    seed = cfg.get("seed", 1024)
    if seed is not None:
        set_seed(seed)

    # == init distributed env ==
    # 设置默认的模型并行配置
    if "plugin_config" not in cfg:
        cfg.plugin_config = {}
    # 设置张量并行度，默认为可用GPU数量
    tp_size = cfg.plugin_config.get("tp_size", torch.cuda.device_count())
    cfg.plugin_config["tp_size"] = tp_size
    # 删除skip_pg_mesh参数，采用另一种方法解决
    if "skip_pg_mesh" in cfg.plugin_config:
        del cfg.plugin_config["skip_pg_mesh"]
    # 添加初始化相关参数，确保pg_mesh正确创建
    cfg.plugin_config["enable_all_optimization"] = False
    cfg.plugin_config["use_cpuoffload"] = False
    # 将插件类型设置为hybrid以启用模型并行
    cfg.plugin = "hybrid"
    
    # 初始化分布式环境
    init_inference_environment()
    # 设置张量并行组
    if tp_size > 1:
        # 创建并设置tensor parallel组
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        # 确保tp_size不超过world_size，避免除零错误
        if tp_size > world_size:
            logger.warning(f"指定的tp_size ({tp_size}) 大于world_size ({world_size})，将自动调整为world_size")
            tp_size = world_size
            cfg.plugin_config["tp_size"] = tp_size
        
        # 计算步长，避免除零
        stride = max(1, world_size // tp_size)
        tp_group_ranks = list(range(0, world_size, stride))
        
        # 如果组大小不符合预期，进行调整
        if len(tp_group_ranks) != tp_size:
            logger.warning(f"由于world_size ({world_size}) 不能被tp_size ({tp_size}) 整除，实际的张量并行组大小为 {len(tp_group_ranks)}")
            
        # 创建新的处理器组    
        tp_group = dist.new_group(tp_group_ranks)
        set_tensor_parallel_group(tp_group)
        
    logger = create_logger()
    logger.info("Inference configuration:\n %s", pformat(cfg.to_dict()))
    is_saving_process = get_is_saving_process(cfg)
    booster = get_booster(cfg)
    booster_ae = get_booster(cfg, ae=True)
    
    # 输出模型并行信息
    if tp_size > 1:
        logger.info(f"Using Tensor Parallelism with {tp_size} GPUs")

    # ======================================================
    # 2. build dataset and dataloader
    # ======================================================
    logger.info("Building dataset...")

    # save directory
    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # == build dataset ==
    if cfg.get("prompt"):
        cfg.dataset.data_path = create_tmp_csv(save_dir, cfg.prompt, cfg.get("ref", None), create=is_main_process())
    dist.barrier()
    dataset = build_module(cfg.dataset, DATASETS)

    # range selection
    start_index = cfg.get("start_index", 0)
    end_index = cfg.get("end_index", None)
    if end_index is None:
        end_index = start_index + cfg.get("num_samples", len(dataset.data) + 1)
    dataset.data = dataset.data[start_index:end_index]
    logger.info("Dataset contains %s samples.", len(dataset))

    # == build dataloader ==
    dataloader_args = dict(
        dataset=dataset,
        batch_size=cfg.get("batch_size", 1),
        num_workers=cfg.get("num_workers", 4),
        seed=cfg.get("seed", 1024),
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        process_group=get_data_parallel_group(),
        prefetch_factor=cfg.get("prefetch_factor", None),
    )
    dataloader, _ = prepare_dataloader(**dataloader_args)

    # == prepare default params ==
    sampling_option = SamplingOption(**cfg.sampling_option)
    sampling_option = sanitize_sampling_option(sampling_option)

    cond_type = cfg.get("cond_type", "t2v")
    prompt_refine = cfg.get("prompt_refine", False)
    fps_save = cfg.get("fps_save", 16)
    num_sample = cfg.get("num_sample", 1)

    type_name = "image" if cfg.sampling_option.num_frames == 1 else "video"
    sub_dir = f"{type_name}_{cfg.sampling_option.resolution}"
    os.makedirs(os.path.join(save_dir, sub_dir), exist_ok=True)
    use_t2i2v = cfg.get("use_t2i2v", False)
    img_sub_dir = os.path.join(sub_dir, "generated_condition")
    if use_t2i2v:
        os.makedirs(os.path.join(save_dir, sub_dir, "generated_condition"), exist_ok=True)

    # ======================================================
    # 3. build model
    # ======================================================
    logger.info("Building models...")

    # == build flux model ==
    if tp_size > 1:
        logger.info(f"Model will be split across {tp_size} GPUs using Tensor Parallelism")
        
    # 确保模型配置考虑到张量并行
    if tp_size > 1 and "model" in cfg:
        # 获取实际的张量并行组大小
        if hasattr(dist, "get_world_size") and get_tensor_parallel_group() is not None:
            actual_tp_size = len(tp_group_ranks)
        else:
            actual_tp_size = tp_size
        
        # 为模型添加并行配置
        if "parallel_config" not in cfg.model:
            cfg.model.parallel_config = {}
        cfg.model.parallel_config["tensor_parallel_size"] = actual_tp_size
        
        # 如果存在自动编码器，也为其添加并行配置
        if "ae" in cfg:
            if "parallel_config" not in cfg.ae:
                cfg.ae.parallel_config = {}
            cfg.ae.parallel_config["tensor_parallel_size"] = actual_tp_size
            
            # 如果存在图像flux模型，也为其添加并行配置
            if "img_flux" in cfg:
                if "parallel_config" not in cfg.img_flux:
                    cfg.img_flux.parallel_config = {}
                cfg.img_flux.parallel_config["tensor_parallel_size"] = actual_tp_size
                
                if "img_flux_ae" in cfg:
                    if "parallel_config" not in cfg.img_flux_ae:
                        cfg.img_flux_ae.parallel_config = {}
                    cfg.img_flux_ae.parallel_config["tensor_parallel_size"] = actual_tp_size

    model, model_ae, model_t5, model_clip, optional_models = prepare_models(
        cfg, device, dtype, offload_model=cfg.get("offload_model", False)
    )
    log_cuda_max_memory("build model")

    if booster:
        logger.info("Applying booster to model...")
        model, _, _, _, _ = booster.boost(model=model)
        model = model.unwrap()
    if booster_ae:
        logger.info("Applying booster to autoencoder model...")
        model_ae, _, _, _, _ = booster_ae.boost(model=model_ae)
        model_ae = model_ae.unwrap()

    # 如果使用张量并行，记录内存使用情况
    if tp_size > 1:
        log_cuda_max_memory("after model parallel boost")
        if is_main_process():
            logger.info("Model has been split across multiple GPUs using Tensor Parallelism")

    api_fn = prepare_api(model, model_ae, model_t5, model_clip, optional_models)

    # prepare image flux model if t2i2v
    if use_t2i2v:
        api_fn_img = prepare_api(
            optional_models["img_flux"], optional_models["img_flux_ae"], model_t5, model_clip, optional_models
        )

    # ======================================================
    # 4. inference
    # ======================================================
    for epoch in range(num_sample):  # generate multiple samples with different seeds
        dataloader_iter = iter(dataloader)
        with tqdm(
            enumerate(dataloader_iter, start=0),
            desc="Inference progress",
            disable=not is_main_process(),
            initial=0,
            total=len(dataloader),
        ) as pbar:
            for _, batch in pbar:
                original_text = batch.pop("text")
                if use_t2i2v:
                    batch["text"] = original_text if not prompt_refine else refine_prompts(original_text, type="t2i")
                    sampling_option_t2i = modify_option_to_t2i(
                        sampling_option,
                        distilled=True,
                        img_resolution=cfg.get("img_resolution", "768px"),
                    )
                    if cfg.get("offload_model", False):
                        model_move_start = time.time()
                        
                        # 处理张量并行情况下的模型加载与卸载
                        if tp_size > 1:
                            logger.info("Loading video diffusion model back to GPU in tensor parallel mode")
                            # 加载视频模型到GPU
                            if hasattr(model, "to_tensor_parallel"):
                                model.to_tensor_parallel(device, dtype)
                            else:
                                model = model.to(device, dtype)
                                
                            if hasattr(model_ae, "to_tensor_parallel"):
                                model_ae.to_tensor_parallel(device, dtype)
                            else:
                                model_ae = model_ae.to(device, dtype)
                                
                            # 卸载图像flux模型到CPU
                            logger.info("Offloading image flux models to CPU in tensor parallel mode")
                            if hasattr(optional_models["img_flux"], "to_tensor_parallel"):
                                optional_models["img_flux"].to_tensor_parallel("cpu", dtype)
                            else:
                                optional_models["img_flux"].to("cpu", dtype)
                                
                            if hasattr(optional_models["img_flux_ae"], "to_tensor_parallel"):
                                optional_models["img_flux_ae"].to_tensor_parallel("cpu", dtype)
                            else:
                                optional_models["img_flux_ae"].to("cpu", dtype)
                        else:
                            # 非并行模式下的处理
                            model = model.to(device, dtype)
                            model_ae = model_ae.to(device, dtype)
                            optional_models["img_flux"].to("cpu", dtype)
                            optional_models["img_flux_ae"].to("cpu", dtype)
                        
                        logger.info(
                            "load video diffusion model to gpu, offload image flux model to cpu: %s s",
                            time.time() - model_move_start,
                        )

                    logger.info("Generating image condition by flux...")
                    x_cond = api_fn_img(
                        sampling_option_t2i,
                        "t2v",
                        seed=sampling_option.seed + epoch if sampling_option.seed else None,
                        channel=cfg["img_flux"]["in_channels"],
                        **batch,
                    ).cpu()

                    # save image to disk
                    batch["name"] = process_and_save(
                        x_cond,
                        batch,
                        cfg,
                        img_sub_dir,
                        sampling_option_t2i,
                        epoch,
                        start_index,
                        saving=is_saving_process,
                    )
                    dist.barrier()

                    if cfg.get("offload_model", False):
                        model_move_start = time.time()
                        model = model.to(device, dtype)
                        model_ae = model_ae.to(device, dtype)
                        optional_models["img_flux"].to("cpu", dtype)
                        optional_models["img_flux_ae"].to("cpu", dtype)
                        logger.info(
                            "load video diffusion model to gpu, offload image flux model to cpu: %s s",
                            time.time() - model_move_start,
                        )

                    ref_dir = os.path.join(save_dir, os.path.join(sub_dir, "generated_condition"))
                    batch["ref"] = [os.path.join(ref_dir, f"{x}.png") for x in batch["name"]]
                    cond_type = "i2v_head"

                batch["text"] = original_text
                if prompt_refine:
                    batch["text"] = refine_prompts(
                        original_text, type="t2v" if cond_type == "t2v" else "t2i", image_paths=batch.get("ref", None)
                    )
                batch["text"] = add_fps_info_to_text(batch.pop("text"), fps=fps_save)
                if "motion_score" in cfg:
                    batch["text"] = add_motion_score_to_text(batch.pop("text"), cfg.get("motion_score", 5))

                logger.info("Generating video...")
                x = api_fn(
                    sampling_option,
                    cond_type,
                    seed=sampling_option.seed + epoch if sampling_option.seed else None,
                    patch_size=cfg.get("patch_size", 2),
                    save_prefix=cfg.get("save_prefix", ""),
                    channel=cfg["model"]["in_channels"],
                    **batch,
                ).cpu()

                if is_saving_process:
                    process_and_save(x, batch, cfg, sub_dir, sampling_option, epoch, start_index)
                dist.barrier()

    logger.info("Inference finished.")
    log_cuda_max_memory("inference")
    
    # 确保清理所有资源
    if tp_size > 1:
        logger.info("Cleaning up tensor parallel resources...")
        # 清理模型
        del model
        del model_ae
        if 'model_t5' in locals():
            del model_t5
        if 'model_clip' in locals():
            del model_clip
        if 'optional_models' in locals():
            del optional_models
        # 清理booster
        if booster:
            del booster
        if booster_ae:
            del booster_ae
        # 强制GC回收内存
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        # 等待所有进程完成清理
        dist.barrier()
        
    logger.info("All resources cleaned up successfully.")


if __name__ == "__main__":
    main()
