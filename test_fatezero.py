import os
import sys
import importlib

# 1. 定义你的本地diffusers路径
local_diffusers_path = '/mnt/fspace/hproject/ELVE/diffusersF/'

# 2. 确保该路径在sys.path的最前面
if local_diffusers_path in sys.path:
    sys.path.remove(local_diffusers_path)
sys.path.insert(0, local_diffusers_path)

# 3. 彻底清理所有相关模块缓存
def clear_diffusers_cache():
    modules_to_remove = []
    for module_name in list(sys.modules.keys()):
        if (module_name.startswith('diffusers') or 
            module_name.startswith('transformers') or
            module_name.startswith('torch')):
            modules_to_remove.append(module_name)
    
    for module_name in modules_to_remove:
        del sys.modules[module_name]
    
    print(f"进程 {os.getpid()} 已清理 {len(modules_to_remove)} 个模块缓存")

# 立即清理缓存
clear_diffusers_cache()

import diffusers
print("________________",diffusers.__file__)
from glob import glob
import copy
import imageio
from typing import Optional,Dict
from tqdm.auto import tqdm
from omegaconf import OmegaConf
import click
import torch.distributed as dist
import torch
import torch.utils.data
import torch.utils.checkpoint
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
)
from video_diffusion.pipelines.p2p_ddim_spatial_temp import P2pDDIMSpatiotempPipeline
# from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
import warnings
warnings.filterwarnings("ignore", message="Failed to load image Python extension")
from diffusers.utils.import_utils import is_xformers_available
from transformers import AutoTokenizer, CLIPTextModel
from einops import rearrange
from video_diffusion.models.unet_3d_condition import UNet3DConditionModel
from video_diffusion.data.dataset import ImageSequenceDataset
from video_diffusion.common.util import get_time_string, get_function_args
from video_diffusion.common.logger import get_logger_config_path
from video_diffusion.common.image_util import log_train_samples
from video_diffusion.common.instantiate_from_config import instantiate_from_config
from video_diffusion.pipelines.p2p_validation_loop import P2pSampleLogger



def init_model(pretrained_model_path,grid_sizeH,grid_sizeW,model_config
               ,test_pipeline_config,video_length,accelerator,data_path,num_inference_steps,device):
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    #F/C FateZero的模型初始化
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_path,
        subfolder="tokenizer",
        use_fast=False,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_path,
        subfolder="text_encoder",
    )
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_path,
        subfolder="vae",
    )
    unet = UNet3DConditionModel.from_2d_model(
        os.path.join(pretrained_model_path, "unet"), model_config=model_config, 
        grid_sizeH=grid_sizeH , grid_sizeW=grid_sizeW
    )
    scheduler=DDIMScheduler.from_pretrained(
            pretrained_model_path,
            subfolder="scheduler",
        )
    controlnet=None
    pipeline = instantiate_from_config(
        test_pipeline_config,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        controlnet=controlnet, 
        disk_store= False,
        video_length=video_length,
        grid_sizeH=grid_sizeH,
        grid_sizeW=grid_sizeW,
        accelerator=accelerator,
        data_path = data_path,
        num_inference_steps = num_inference_steps
    )

    # C/F ControlNetModel的模型初始化
    # controlnet = ControlNetModel.from_pretrained("/mnt/fspace/vidtome_model/models--lllyasviel"
    # "--control_v11f1p_sd15_depth/snapshots/539f99181d33db39cf1af2e517cd8056785f0a87/", 
    # torch_dtype= weight_dtype).to(device, weight_dtype)
    # pipeline_cn = StableDiffusionControlNetImg2ImgPipeline.from_pretrained("/mnt/fspace/hproject/model/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype= weight_dtype).to(device,  weight_dtype)
   
    # vae=pipeline_cn.vae
    # unet=pipeline_cn.unet
    # text_encoder=pipeline_cn.text_encoder
    # scheduler=pipeline_cn.scheduler
    # tokenizer=pipeline_cn.tokenizer
    # pipeline_cn.set_progress_bar_config(disable=True)
   

    # pipeline = P2pDDIMSpatiotempPipeline(
    #     vae=vae,
    #     text_encoder=text_encoder,
    #     tokenizer=tokenizer,
    #     unet=unet,
    #     scheduler=scheduler,
    #     controlnet=controlnet,  
    #     disk_store=False,
    #     video_length=video_length,
    #     grid_sizeH=grid_sizeH,
    #     grid_sizeW=grid_sizeW,
    #     accelerator=accelerator,
    #     data_path=data_path,
    #     num_inference_steps=num_inference_steps,
    #     # pipeline_cn=pipeline_cn  # 如果需要ControlNet管道，取消注释
    # )
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    scheduler.set_timesteps(num_inference_steps)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    if is_xformers_available():
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print("xformers加载异常")  

    return pipeline,unet,vae,text_encoder,tokenizer,weight_dtype,controlnet 

def test(
    config: str,
    pretrained_model_path: str,
    dataset_config: Dict,
    logdir: str = None,
    editing_config: Optional[Dict] = None,
    test_pipeline_config: Optional[Dict] = None,
    gradient_accumulation_steps: int = 1,
    seed: Optional[int] = None,
    mixed_precision: Optional[str] = "fp16",
    batch_size: int = 1,
    model_config: dict={},
    verbose: bool=True,
    **kwargs
):
    args = get_function_args()
    grid_sizeH = editing_config["grid_sizeH"]
    grid_sizeW = editing_config["grid_sizeW"]
    controlnet_conditioning_scale=1.0
    video_length = dataset_config["n_sample_frame"]
    time_string = get_time_string()
    if logdir is None:
        logdir = config.replace('config', 'result').replace('.yml', '').replace('.yaml', '')
    logdir += f"_{time_string}"
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )
    if accelerator.is_main_process:
        os.makedirs(logdir, exist_ok=True)
        OmegaConf.save(args, os.path.join(logdir, "config.yml"))
    
    pipeline,unet,vae,text_encoder,tokenizer,weight_dtype,controlnet=init_model(pretrained_model_path,grid_sizeH,grid_sizeW,model_config
               ,test_pipeline_config,video_length,accelerator,dataset_config["path"],
               editing_config['num_inference_steps'],accelerator.device)

    prompt_ids = tokenizer(
        dataset_config["prompt"],
        truncation=True,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids

    video_dataset = ImageSequenceDataset(**dataset_config, prompt_ids=prompt_ids,GPU_num=torch.cuda.device_count())
    train_dataloader = torch.utils.data.DataLoader(
        video_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        
    )
    unet, train_dataloader = accelerator.prepare(
        unet, train_dataloader
    )
    if accelerator.is_main_process:
        accelerator.init_trackers("video")  

    validation_sample_logger = P2pSampleLogger(**editing_config, logdir=logdir, source_prompt=dataset_config['prompt'],controlnet_conditioning_scale=controlnet_conditioning_scale)
   
    for step, batch in enumerate(train_dataloader):
        assert batch["images"].shape[0] == 1, "Only support, overfiting on a single video"
        print("batch['images'].shape=",batch['images'].shape)
        vae.eval()
        text_encoder.eval()
        unet.eval()

        #C/F controlnet的eval，生成深度图作为控制条件
        #controlnet.eval()

        images_for_depth = rearrange(batch["images"].to(dtype=weight_dtype), "b c f h w -> (b f) c h w")
        depth_images = P2pDDIMSpatiotempPipeline.depth_preprocess(images_for_depth)
        depth_images = depth_images.to(device=accelerator.device, dtype=weight_dtype)

        #F/C
        depth_images=None
        
        text_embeddings = pipeline._encode_prompt(
                dataset_config.prompt,
                device = accelerator.device,
                num_images_per_prompt = 1,
                do_classifier_free_guidance = True,
                negative_prompt=None
        )
        use_inversion_attention =  editing_config.get('use_inversion_attention', False)
        controlnet_conditioning_scale = 1.0
        batch['ddim_init_latents']  = pipeline.prepare_latents_ddim_inverted(
            rearrange(batch["images"].to(dtype=weight_dtype), "b c f h w -> (b f) c h w"),
            batch_size = 1,
            num_images_per_prompt = 1,  # not sure how to use it
            text_embeddings = text_embeddings,
            prompt = dataset_config.prompt,
            store_attention=use_inversion_attention,
            LOW_RESOURCE = True, # not classifier-free guidance
            save_path = logdir if verbose else None,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            control_image=depth_images,  # 传入深度图作为控制条件
            )
        
        vae.eval()
        text_encoder.eval()
        unet.eval()
        #controlnet.eval()
        images = batch["images"].to(dtype=weight_dtype)
        images = rearrange(images, "b c f h w -> (b f) c h w")

        if accelerator is not None:
            validation_sample_logger.log_sample_images(
                image=images, # torch.Size([8, 3, 512, 512])
                pipeline=pipeline,
                device=accelerator.device,
                step=0,
                latents = batch['ddim_init_latents'],
                save_dir = logdir if verbose else None,
                control_image=depth_images,  # 传入控制图像
            )

    if torch.cuda.device_count()>1:
   
        dist.barrier()
        rank = dist.get_rank()
        if rank==0:
            gif1 = Image.open(f"{logdir}/sample/step_0_GPU0.gif")
            gif2 = Image.open(f"{logdir}/sample/step_0_GPU1.gif")
            # 提取两个 GIF 的所有帧
            frames1 = []
            frames2 = []
            try:
                while True:
                    frames1.append(gif1.copy())
                    gif1.seek(len(frames1))  # 移动到下一帧
            except EOFError:
                pass
            try:
                while True:
                    frames2.append(gif2.copy())
                    gif2.seek(len(frames2))  # 移动到下一帧
            except EOFError:
                pass

            # 合并帧
            combined_frames = frames1 + frames2
            output_path=f"{logdir}/sample/step1.gif"
            # 保存为新的 GIF
            combined_frames[0].save(
                output_path,
                save_all=True,
                append_images=combined_frames[1:],
                loop=0,  # 无限循环
                duration=gif1.info['duration']  # 使用第一个 GIF 的帧延迟
            )
    accelerator.end_training()
     
        


@click.command()
@click.option("--config", type=str, default="config/sample.yml")
def run(config):
    Omegadict = OmegaConf.load(config)
    if 'unet' in os.listdir(Omegadict['pretrained_model_path']):
        test(config=config, **Omegadict)


if __name__ == "__main__":
    run()
