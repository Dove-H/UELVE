import os
import sys
from glob import glob
from typing import Optional, Dict
from omegaconf import OmegaConf
import click
import torch.distributed as dist
import torch
import torch.utils.data
import PIL
import grid
from PIL import Image
from accelerate import Accelerator
import warnings
from contextlib import contextmanager
warnings.filterwarnings("ignore", message="Failed to load image Python extension")
from transformers import AutoTokenizer, CLIPTextModel
from einops import rearrange
import torchvision.transforms as T
import transformers

from video_diffusion.data.dataset import ImageSequenceDataset
from video_diffusion.common.util import get_time_string, get_function_args
from video_diffusion.common.instantiate_from_config import instantiate_from_config
from video_diffusion.pipelines.p2p_validation_loop import P2pSampleLogger
import numpy as np
import utils.preprocesser_utils as pu
import utils.image_process_utils as ipu
import utils.feature_utils as fu


# diffusers 路径管理函数
def setup_diffusers_path(method):
    """根据 method 参数设置不同的 diffusers 路径"""
    if method == "FateZero":
        local_diffusers_path = '/mnt/fspace/hproject/ELVE/diffusersF/'
    elif method == "ControlNet":
        local_diffusers_path = '/mnt/fspace/hproject/ELVE/diffusersC/'
    else:
        raise ValueError(f"Unknown method: {method}")
    if local_diffusers_path in sys.path:
        sys.path.remove(local_diffusers_path)  
    sys.path.insert(0, local_diffusers_path)
    return local_diffusers_path

def clear_diffusers_cache():
    """只清理 diffusers 和 transformers 相关模块缓存，不清理 torch"""
    modules_to_remove = []
    for module_name in list(sys.modules.keys()):
        if (module_name.startswith('diffusers') or 
            module_name.startswith('transformers')):
            if module_name != 'transformers' and not module_name.startswith('transformers.'):
                modules_to_remove.append(module_name)
    for module_name in modules_to_remove:
        del sys.modules[module_name]

@contextmanager
def diffusers_context(method):
    """diffusers 版本切换的上下文管理器"""
    local_diffusers_path = setup_diffusers_path(method)
    clear_diffusers_cache()
    
    try:
        yield local_diffusers_path
    finally:
        pass



def depth_preprocess(image_tensor, accelerator, controlnet, prepare_control_image,Control_image_path):
    """
    处理输入的四维张量，转换为PIL图像列表进行深度预处理
    image_tensor: 形状为 (batch_size * frames, channels, height, width) 的张量
    """
    if len(os.listdir(Control_image_path)) > 0:
                control_torch = torch.load(os.path.join(Control_image_path, 'control.pt')).to(accelerator.device)
    else:
        # 将张量转换为PIL图像列表
        image_pil_list = []
        
        # 遍历张量中的每个图像
        for i in range(image_tensor.shape[0]):
            # 获取单个图像张量 [C, H, W]
            img_tensor = image_tensor[i]
            
            # 将张量从 [C, H, W] 转换为 [H, W, C]
            if img_tensor.dim() == 3:
                img_tensor = img_tensor.permute(1, 2, 0)
            
            # 转换为numpy数组并确保值范围在[0, 255]
            if img_tensor.max() <= 1.0:
                img_array = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            else:
                img_array = img_tensor.cpu().numpy().astype(np.uint8)
            
            # 转换为PIL图像
            pil_image = Image.fromarray(img_array)
            image_pil_list.append(pil_image)
        
        control_torch_list = []
        # 遍历输入的PIL图像列表
        for image_pil in image_pil_list:
            # 获取PIL图像的宽度和高度
            width, height = image_pil.size
            # 对PIL图像进行预处理，生成控制图像
            # 将一个PIL图像转换为一系列帧并放入帧列表，grid_size=[2,2]
            list_of_image_pils = fu.pil_grid_to_frames(image_pil, grid_size=[2,2]) # List[C, W, H] -> len = num_frames
            # 对每个帧进行像素完美处理，处理方式由self.preprocess_name指定
            list_of_pils = [pu.pixel_perfect_process(np.array(frame_pil, dtype='uint8'), "canny") for frame_pil in list_of_image_pils]
            # 将处理后的帧列表转换为numpy数组
            control_images = np.array(list_of_pils)
            # 从numpy数组创建一个新的网格（四合一的大numpy向量），网格大小由self.grid决定
            control_img = ipu.create_grid_from_numpy(control_images, grid_size=[2,2])
            # 将网格转换为灰度图像
            control_pil = PIL.Image.fromarray(control_img).convert("L")
            # 根据控制图像、宽度和高度准备控制图像
            control_image = prepare_control_image(
                image=control_pil,
                width=width,
                height=height,
                device=accelerator.device,
                dtype=controlnet.dtype,
                batch_size=1,
                num_images_per_prompt=1
            )
            # 将控制图像添加到控制图像列表中
            control_torch_list.append(control_image)

        # 将控制图像列表中的torch张量按行拼接，并移动到指定设备上
        control_torch = torch.cat(control_torch_list, dim=0).to(accelerator.device)
        torch.save(control_torch, os.path.join(Control_image_path, 'control.pt'))
        print("control_torch.shape=",control_torch.shape)
    return control_torch

def init_model(pretrained_model_path, method, grid_sizeH, grid_sizeW, model_config,
               test_pipeline_config, video_length, accelerator, data_path, num_inference_steps, device):
    
    # 动态导入 diffusers 相关模块
    import diffusers
    from diffusers import AutoencoderKL, DDIMScheduler
    from diffusers.utils.import_utils import is_xformers_available

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if method == "FateZero":
        # FateZero的模型初始化
        from video_diffusion.models.unet_3d_condition import UNet3DConditionModel
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
            grid_sizeH=grid_sizeH, grid_sizeW=grid_sizeW
        )
        scheduler = DDIMScheduler.from_pretrained(
            pretrained_model_path,
            subfolder="scheduler",
        )
        controlnet = None
        prepare_control_image = None
        scheduler_config=None
        pipeline = instantiate_from_config(
            test_pipeline_config,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            controlnet=controlnet, 
            method=method,
            disk_store=False,
            video_length=video_length,
            grid_sizeH=grid_sizeH,
            grid_sizeW=grid_sizeW,
            accelerator=accelerator,
            data_path=data_path,
            num_inference_steps=num_inference_steps
        )

    elif method == "ControlNet":
        # ControlNet的模型初始化
        from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11f1p_sd15_depth", 
            torch_dtype=weight_dtype
        ).to(device, weight_dtype)
        pipeline_cn = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "/mnt/fspace/hproject/model/stable-diffusion-v1-5", 
            controlnet=controlnet, 
            torch_dtype=weight_dtype
        ).to(device, weight_dtype)
        prepare_control_image = pipeline_cn.prepare_control_image
        vae = pipeline_cn.vae
        unet = pipeline_cn.unet
        text_encoder = pipeline_cn.text_encoder
        scheduler = pipeline_cn.scheduler
        tokenizer = pipeline_cn.tokenizer
        scheduler_config = pipeline_cn.scheduler.config
        pipeline_cn.set_progress_bar_config(disable=True)

        pipeline = instantiate_from_config(
            test_pipeline_config,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            controlnet=controlnet, 
            method=method,
            disk_store=False,
            video_length=video_length,
            grid_sizeH=grid_sizeH,
            grid_sizeW=grid_sizeW,
            accelerator=accelerator,
            data_path=data_path,
            num_inference_steps=num_inference_steps
        )
    
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

    return pipeline, unet, vae, text_encoder, tokenizer, weight_dtype, controlnet, prepare_control_image,scheduler_config

def test(
    config: str,
    pretrained_model_path: str,
    method: str,
    dataset_config: Dict,
    logdir: str = None,
    editing_config: Optional[Dict] = None,
    test_pipeline_config: Optional[Dict] = None,
    gradient_accumulation_steps: int = 1,
    seed: Optional[int] = None,
    mixed_precision: Optional[str] = "fp16",
    batch_size: int = 1,
    model_config: dict = {},
    verbose: bool = True,
    **kwargs
):
    # 使用上下文管理器设置 diffusers 环境
    with diffusers_context(method):
        import diffusers
        print("________________", diffusers.__file__)
    
    args = get_function_args()
    grid_sizeH = editing_config["grid_sizeH"]
    grid_sizeW = editing_config["grid_sizeW"]
    controlnet_conditioning_scale = 1.0
    video_length = dataset_config["n_sample_frame"]
    time_string = get_time_string()
    GPU_num=torch.cuda.device_count()
    if GPU_num>1:
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
        Control_image_path = os.path.join('Control', str(GPU_num) + "GPU",str(dist.get_rank()), os.path.basename(dataset_config["path"]),str(video_length) )
    else:
        Control_image_path = os.path.join('Control',str(GPU_num) + "GPU",os.path.basename(dataset_config["path"]),str(video_length)  )
    os.makedirs(Control_image_path, exist_ok = True)
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
    
    pipeline, unet, vae, text_encoder, tokenizer, weight_dtype, controlnet, prepare_control_image,scheduler_config = init_model(
        pretrained_model_path, method, grid_sizeH, grid_sizeW, model_config,
        test_pipeline_config, video_length, accelerator, dataset_config["path"],
        editing_config['num_inference_steps'], accelerator.device
    )

    prompt_ids = tokenizer(
        dataset_config["prompt"],
        truncation=True,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids

    video_dataset = ImageSequenceDataset(**dataset_config, prompt_ids=prompt_ids, GPU_num=torch.cuda.device_count())
    train_dataloader = torch.utils.data.DataLoader(
        video_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
    )
    
    unet, train_dataloader = accelerator.prepare(unet, train_dataloader)
    
    if accelerator.is_main_process:
        accelerator.init_trackers("video")  

    validation_sample_logger = P2pSampleLogger(
        **editing_config, 
        logdir=logdir, 
        source_prompt=dataset_config['prompt'],
        controlnet_conditioning_scale=controlnet_conditioning_scale
    )
   
    for step, batch in enumerate(train_dataloader):
        assert batch["images"].shape[0] == 1, "Only support, overfiting on a single video"
        print("batch['images'].shape=", batch['images'].shape)
        
        vae.eval()
        text_encoder.eval()
        unet.eval()
        
        if method == "ControlNet":
            # ControlNet: 生成深度图作为控制条件
            images=batch["images"]
            images=grid.combine_frames_in_grid(images,grid_sizeH,grid_sizeW)
            images_for_depth = rearrange(images.to(dtype=weight_dtype), "b c f h w -> (b f) c h w")
            depth_images = depth_preprocess(images_for_depth,accelerator,controlnet,prepare_control_image,Control_image_path)
            depth_images=rearrange(depth_images,"(b f) c h w -> b c f h w", f=1)
            depth_images=grid.restore_frames_from_grid(depth_images,grid_sizeH,grid_sizeW)
            depth_images=rearrange(depth_images,"b c f h w -> (b f) c h w")
            depth_images = depth_images.to(device=accelerator.device, dtype=weight_dtype)
        elif method == "FateZero":
            # FateZero: 不需要深度图
            depth_images = None
        
        text_embeddings = pipeline._encode_prompt(
            dataset_config.prompt,
            device=accelerator.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=None
        )
        
        use_inversion_attention = editing_config.get('use_inversion_attention', False)
        controlnet_conditioning_scale = 1.0
        
        batch['ddim_init_latents'] = pipeline.prepare_latents_ddim_inverted(
            rearrange(batch["images"].to(dtype=weight_dtype), "b c f h w -> (b f) c h w"),
            batch_size=1,
            num_images_per_prompt=1,
            text_embeddings=text_embeddings,
            prompt=dataset_config.prompt,
            store_attention=use_inversion_attention,
            LOW_RESOURCE=True,
            save_path=logdir if verbose else None,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            control_image=depth_images,  # 传入深度图作为控制条件
            scheduler_config=scheduler_config
        )
        
        vae.eval()
        text_encoder.eval()
        unet.eval()
        
        images = batch["images"].to(dtype=weight_dtype)
        images = rearrange(images, "b c f h w -> (b f) c h w")

        if accelerator is not None:
            validation_sample_logger.log_sample_images(
                image=images,
                pipeline=pipeline,
                device=accelerator.device,
                step=0,
                latents=batch['ddim_init_latents'],
                save_dir=logdir if verbose else None,
                control_image=depth_images,  # 传入控制图像
            )

    if torch.cuda.device_count() > 1:
        dist.barrier()
        rank = dist.get_rank()
        if rank == 0:
            gif1 = Image.open(f"{logdir}/sample/step_0_GPU0.gif")
            gif2 = Image.open(f"{logdir}/sample/step_0_GPU1.gif")
            
            frames1 = []
            frames2 = []
            try:
                while True:
                    frames1.append(gif1.copy())
                    gif1.seek(len(frames1))
            except EOFError:
                pass
            try:
                while True:
                    frames2.append(gif2.copy())
                    gif2.seek(len(frames2))
            except EOFError:
                pass

            combined_frames = frames1 + frames2
            output_path = f"{logdir}/sample/step1.gif"
            combined_frames[0].save(
                output_path,
                save_all=True,
                append_images=combined_frames[1:],
                loop=0,
                duration=gif1.info['duration']
            )
    
    accelerator.end_training()

@click.command()
@click.option("--config", type=str, default="config/sample.yml")
def run(config):
    # 首先读取配置获取 method
    Omegadict = OmegaConf.load(config)
    method = Omegadict.get('method')
    
    if not method:
        raise ValueError("Method parameter is required in config")
    
    # 使用上下文管理器运行测试
    with diffusers_context(method):
        # 重新加载配置以确保一致性
        Omegadict = OmegaConf.load(config)
        
        if 'unet' in os.listdir(Omegadict['pretrained_model_path']):
            test(config=config, **Omegadict)

if __name__ == "__main__":
    run()