# From https://github.com/showlab/Tune-A-Video/blob/main/train_tuneavideo.py
import argparse
import logging
import inspect
import math
import os
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.checkpoint
import numpy as np
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from video_diffusion.models_tune.unet  import UNet3DConditionModel
from video_diffusion.data.dataset_tune import TuneAVideoDataset
from einops import rearrange
import grid
check_min_version("0.10.0.dev0")
logger = get_logger(__name__, log_level="INFO")

# 分块梯度累积：单卡且 n_sample_frames >= ACCUM_ABOVE_FRAMES（含 200f 本身）时，
# 训练步内按网格图切片逐块 forward+backward（梯度累积），激活峰值只与块大小相关，
# 避免 200f+ 单卡一次性为全部网格图建计算图导致 OOM；多卡/多进程路径不受影响。
# 块大小可用 ELVE_TUNE_ACCUM_GRIDS 覆盖（每格 = grid_sizeH*grid_sizeW 帧）。
ACCUM_ABOVE_FRAMES = 200
ACCUM_CHUNK_GRIDS = max(1, int(os.environ.get("ELVE_TUNE_ACCUM_GRIDS", "16") or 16))


def main(
    pretrained_model_path: str,
    output_dir: str,
    train_data: Dict,
    validation_data: Dict,  # 仅用于吸收 yaml 字段，训练中已不使用（验证/反演逻辑已剥离）
    validation_steps: int = 100,
    trainable_modules: Tuple[str] = (
        "attn1.to_q",
        "attn2.to_q",
        "attn_temp",
    ),
    train_batch_size: int = 1,
    max_train_steps: int = 500,
    ddim_grid_steps: int = 50,  # timestep 与推理对齐：训练只在 DDIM 推理格点上采样，此值=编辑 yaml 的 num_inference_steps
    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing=True,
    checkpointing_steps: int = 500,
    resume_from_checkpoint: Optional[str] = None,
    mixed_precision: Optional[str] = "fp16",
    use_8bit_adam: bool = False,
    enable_xformers_memory_efficient_attention: bool = True,
    seed: Optional[int] = None,
    grid_sizeH: int=1,
    grid_sizeW: int=4,
    model_config: Dict={}
):
    *_, config = inspect.getargvalues(inspect.currentframe())
    n_sample_frames=train_data['n_sample_frames']
    print("n_sample_frames=",n_sample_frames)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )
  
    def encode_in_f_batches(image_grid, model):
        device=accelerator.device
        # 获取输入的 batch 大小
        batches = torch.split(image_grid, 1, dim=0)  # 分批
        #print("-------------------------------------")
        #print("image_grid=",image_grid.shape)
        #print("batch_size_grid=",batch_size_grid)
        # 初始化一个空列表，用于存储每个批次的编码结果
        encoded_batches = []
        # 遍历每个批次，进行编码并计算均值
        for batch in batches:
            # 对当前批次进行编码并计算均值
            #print("batch.shape=",batch.shape)
            
            batch=batch.half()
            latent_dist = model.encode(batch).latent_dist.sample()
            #print("***************")
            #mean_latents = latent_dist.mean
            encoded_batches.append(latent_dist)

        # 将分批的编码结果合并回原来的第一个维度
        latents_grid = torch.cat(encoded_batches, dim=0)
        return latents_grid

    def process_with_unet(unet, latent_model_input, t, text_embeddings):
        # 网格图之间相互独立：按 chunk 张沿 batch 维打包前向（数学等价于逐图串行，kernel 利用率更高，
        # 激活总量不变——本来就是全部前向建一张图后一次 backward）。timesteps/text 条件对每张网格图相同，
        # 打包时沿 batch 维复制。ELVE_TUNE_FWD_CHUNK=1 可退回逐图串行（等价性验证用）。
        assert latent_model_input.shape[0] == 1, "process_with_unet 按 batch=1 设计"
        n_grids = latent_model_input.shape[2]
        chunk = int(os.environ.get("ELVE_TUNE_FWD_CHUNK", "4") or 4)
        chunk = max(1, min(chunk, n_grids))
        noise_preds = []
        for s in range(0, n_grids, chunk):
            g = latent_model_input[:, :, s:s + chunk, :, :]            # [1, C, k, H, W]
            k = g.shape[2]
            flat = g.permute(2, 0, 1, 3, 4).reshape(k, g.shape[1], g.shape[3], g.shape[4]).unsqueeze(2)  # [k, C, 1, H, W]
            tt = t.reshape(-1)[:1].repeat(k)                           # 同一 t 复制给 k 张网格图
            eh = text_embeddings.expand(k, -1, -1)
            pred = unet(flat, tt, encoder_hidden_states=eh).sample     # [k, C, 1, H, W]
            noise_preds.append(pred)
        # [G, C, 1, H, W] -> [1, C, G, H, W]
        noise_pred = torch.cat(noise_preds, dim=0).permute(2, 1, 0, 3, 4)
        return noise_pred

    # 设置日志记录格式，并根据是否是主进程来设置不同的日志级别。
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # 设置随机数种子。
    if seed is not None:
        set_seed(seed)

    # 创建输出目录，并保存配置文件。
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/inv_latents", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    # 加载预训练模型中的调度器、标记器和模型。
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    # 训练 timestep 与推理对齐：只从 DDIM 推理实际访问的格点采样。
    # SD1.5 scheduler（steps_offset=1 + leading 间距）50 步格点 = {981,961,...,21,1}，
    # 推理也只在这些 t 上查询模型，训练分布与查询分布一致。
    _grid_sched = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    _grid_sched.set_timesteps(ddim_grid_steps)
    ddim_timestep_grid = torch.as_tensor(_grid_sched.timesteps, dtype=torch.long, device=accelerator.device)
    print(f"[tune] timestep 对齐推理格点: {ddim_grid_steps} 个 t, min={int(ddim_timestep_grid.min())} max={int(ddim_timestep_grid.max())}")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet",grid_sizeH=grid_sizeH,grid_sizeW=grid_sizeW)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    
    for name, module in unet.named_modules():
        if name.endswith(tuple(trainable_modules)):
            for params in module.parameters():
                params.requires_grad = True

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # 单卡定制：gradient_checkpointing 启用阈值从 >16f 降到 >8f（yaml 里 16f 及以下为
    # false）；双卡仍走 yaml 原值，多卡路径不受影响
    if gradient_checkpointing or (
        torch.cuda.device_count() == 1 and n_sample_frames > 8
    ):
        unet.enable_gradient_checkpointing()
        print("gradient_checkpointing: on"
              + (" (单卡>8f 自动启用)" if not gradient_checkpointing else " (yaml)"))

    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )
    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        unet.parameters(),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    train_dataset = TuneAVideoDataset(**train_data,GPU_num=torch.cuda.device_count())
    train_dataset.prompt_ids = tokenizer(
        train_dataset.prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids[0]

    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size,shuffle=False
    )

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2video-fine-tune")


    def alternate(latents,flag):
        #print("latents.shape=",latents.shape)
        # 与推理一致：按补齐后的总帧数置换（latents 为网格合并后 b c F h w，F*网格面积=补齐后帧数）
        n_frames = latents.shape[2] * grid_sizeH * grid_sizeW
        org = list(range(n_frames))
        #print("n_sample_frame=",n_frames)
        alt = org[grid_sizeH:] + org[:grid_sizeH]
        #alt = [0,1,4,5,2,3,6,7]
        if n_frames==grid_sizeH*grid_sizeW:
            sqc=org
        elif flag%2==0 and n_frames!=grid_sizeH*grid_sizeW:
            sqc=alt
        else:
            sqc=torch.argsort(torch.tensor(alt))
        
        latents=grid.restore_frames_from_grid(latents,grid_sizeH,grid_sizeW)
        latents=latents[:, :, sqc, :, :]
        latents=grid.combine_frames_in_grid(latents,grid_sizeH,grid_sizeW)
        
        return latents

    def alternate_multiGPU(latents, flag):
        rank = dist.get_rank()
        device = latents.device  # 确保设备一致
        # if latents.shape[2] == 1:
        #     print("No change")
        
        if flag % 2 == 0:
            # 偶数变
            if rank == 0:
                latents = grid.restore_frames_from_grid(latents, grid_sizeH,grid_sizeW)
                data = latents[:, :, :grid_sizeH, :, :].to(device)  # 确保设备一致
                data1 = torch.zeros_like(data)
                data = data.contiguous()  # 确保张量是连续的
                latents = latents[:, :, grid_sizeH:, :, :]
                dist.send(data, dst=1)  # Rank 0 发送数据到 Rank 1
                dist.recv(data1, src=1)  # Rank 0 接收来自 Rank 1 的数据
                latents = torch.cat([latents,data1], dim=2).to(device)
                latents = grid.combine_frames_in_grid(latents, grid_sizeH,grid_sizeW)
            elif rank == 1:
                latents = grid.restore_frames_from_grid(latents, grid_sizeH,grid_sizeW)
                data = latents[:, :, :grid_sizeH, :, :].to(device)  # 确保设备一致
                data1 = torch.zeros_like(data)
                data = data.contiguous()  # 确保张量是连续的
                latents = latents[:, :, grid_sizeH:, :, :]
                dist.recv(data1, src=0)  # Rank 1 接收来自 Rank 0 的数据
                dist.send(data, dst=0)  # Rank 1 发送数据到 Rank 0
                latents = torch.cat([latents,data1], dim=2).to(device)
                latents = grid.combine_frames_in_grid(latents, grid_sizeH,grid_sizeW)
        else:
            # 奇数不变
            if rank == 0:
                latents = grid.restore_frames_from_grid(latents, grid_sizeH,grid_sizeW)
                data = latents[:, :, -grid_sizeH:, :, :].to(device)  # 确保设备一致
                data1 = torch.zeros_like(data)
                data = data.contiguous()  # 确保张量是连续的
                latents = latents[:, :, :-grid_sizeH, :, :]
                dist.send(data, dst=1)  # Rank 0 发送数据到 Rank 1
                dist.recv(data1, src=1)  # Rank 0 接收来自 Rank 1 的数据
                latents = torch.cat([data1,latents], dim=2).to(device)
                latents = grid.combine_frames_in_grid(latents, grid_sizeH,grid_sizeW)
            elif rank == 1:
                latents = grid.restore_frames_from_grid(latents, grid_sizeH,grid_sizeW)
                data = latents[:, :, -grid_sizeH:, :, :].to(device)  # 确保设备一致
                data1 = torch.zeros_like(data)
                data = data.contiguous()  # 确保张量是连续的
                latents = latents[:, :, :-grid_sizeH, :, :]
                dist.recv(data1, src=0)  # Rank 1 接收来自 Rank 0 的数据
                dist.send(data, dst=0)  # Rank 1 发送数据到 Rank 0
                latents = torch.cat([data1,latents], dim=2).to(device)
                latents = grid.combine_frames_in_grid(latents, grid_sizeH,grid_sizeW)
        dist.barrier()  # 同步所有进程
        
        return latents

    # 预编码本 rank 分片的 latents，全程复用（不再像 Tune-A-Video 原版那样每步重新 VAE 编码）。
    # 帧补齐与推理（p2p_ddim_fatezero.prepare_latents_ddim_inverted）完全一致——
    # 复制末帧到 grid_sizeH*grid_sizeW 的整数倍，保证微调学到的帧编排与推理相同
    batch = next(iter(train_dataloader))
    pixel_values = batch["pixel_values"]
    pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
    n_real = pixel_values.shape[2]
    gA = grid_sizeH * grid_sizeW
    n_pad = (-n_real) % gA
    if n_pad:
        print(f'[tune] 帧数 {n_real} 非 {gA} 整数倍，复制末帧补齐到 {n_real + n_pad}')
        pixel_values = torch.cat([pixel_values, pixel_values[:, :, -1:, :, :].expand(-1, -1, n_pad, -1, -1)], dim=2)
    pixel_values = grid.combine_frames_in_grid(pixel_values, grid_sizeH,grid_sizeW)
    pixel_values = rearrange(pixel_values, "b c f h w -> b f c h w")
    video_length = pixel_values.shape[1]
    pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
    latents = encode_in_f_batches(pixel_values, vae)
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
    latents = latents * 0.18215
    print("latents.shape=",latents.shape)

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if resume_from_checkpoint:
        if resume_from_checkpoint != "latest":
            path = os.path.basename(resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1]
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(output_dir, path))
        global_step = int(path.split("-")[1])

        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch
    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    if torch.cuda.device_count()>1:
        # alternate_multiGPU 的 send/recv 与数据集帧分片均按双卡硬编码：
        # 非双卡分布式启动在此拦下，避免静默帧划分错乱
        assert dist.is_available() and dist.is_initialized() and dist.get_world_size() == 2, \
            "多卡 tune 只支持 accelerate launch --num_processes=2（alternate_multiGPU 按双卡设计）"
        print("启用静态图")
        unet._set_static_graph()
    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        train_loss = 0.0
        if torch.cuda.device_count()>1:
            latents= alternate_multiGPU(latents=latents,flag=epoch)
        else:
            latents=alternate(latents=latents,flag=epoch)

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each video
                # 与推理对齐：从 DDIM 推理格点均匀采样（而非全区间 [0,1000) 均匀）
                timesteps = ddim_timestep_grid[torch.randint(0, ddim_timestep_grid.shape[0], (bsz,), device=latents.device)]
                # Add noise to the latents according to the noise magnitude at each timestep
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["prompt_ids"])[0]
                # Get the target for loss depending on the prediction type
                if noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")
                # Predict the noise residual and compute loss
                # 单卡 & >=ACCUM_ABOVE_FRAMES：按网格图切片逐块 forward+backward 累积梯度，
                # 单块计算图随 backward 释放（激活峰值只与块大小相关）；块权重 = 块内格数/总格数，
                # 累积梯度与全帧一次 mean loss 严格等价
                n_grids_total = noisy_latents.shape[2]
                if (torch.cuda.device_count() == 1
                        and n_sample_frames >= ACCUM_ABOVE_FRAMES):
                    loss = None
                    for s in range(0, n_grids_total, ACCUM_CHUNK_GRIDS):
                        g_pred = process_with_unet(
                            unet, noisy_latents[:, :, s:s + ACCUM_CHUNK_GRIDS],
                            timesteps, encoder_hidden_states)
                        g_loss = F.mse_loss(
                            g_pred.float(), target[:, :, s:s + ACCUM_CHUNK_GRIDS].float(),
                            reduction="mean")
                        w = g_pred.shape[2] / n_grids_total
                        accelerator.backward(g_loss * w)
                        loss = (g_loss * w).detach() if loss is None else loss + (g_loss * w).detach()
                else:
                    model_pred = process_with_unet(unet, noisy_latents, timesteps, encoder_hidden_states)
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    # Backpropagate
                    accelerator.backward(loss)
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break
   
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    from diffusers.pipeline_utils import DiffusionPipeline
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        pipeline = DiffusionPipeline.from_pretrained(
            pretrained_model_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
        )
        pipeline.save_pretrained(output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/tuneavideo.yaml")
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))
