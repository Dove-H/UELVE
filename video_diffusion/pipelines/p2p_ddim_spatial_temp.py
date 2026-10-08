from typing import Callable, List, Optional, Union
import os, sys
import PIL
import time
from accelerate import Accelerator
import torch
import random
import torch.distributed as dist
import grid
import time
import numpy as np
from einops import rearrange
from tqdm import trange, tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.utils import deprecate, logging
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.models import AutoencoderKL
from diffusers.schedulers import (DDIMScheduler)
from .stable_diffusion import SpatiotempStableDiffusionPipeline
from video_diffusion.prompt_attention import attention_util
import torchvision.transforms as T
import transformers


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
class P2pDDIMSpatiotempPipeline(SpatiotempStableDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet,
        scheduler: Union[DDIMScheduler],
        controlnet, 
        method:str,
        control_class:str,
        disk_store: bool=False,
        video_length: int=8,
        grid_sizeH: int=2,
        grid_sizeW: int=2,
        accelerator : Accelerator=None,
        data_path : str="",
        num_inference_steps : int=0,
        ):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler)
        self.method=method
        self.control_class=control_class
        self.scheduler=scheduler
        self.controlnet = controlnet
        if method=="FateZero":
            #F/C FateZero需要初始化controller
            self.store_controller = attention_util.AttentionStore(disk_store=disk_store,stage=0)
            self.empty_controller = attention_util.EmptyControl()
        elif method=="ControlNet":
            #C/F controlnet不需要controller
            self.store_controller = None
            self.empty_controller = None

        self.grid_sizeH=grid_sizeH
        self.grid_sizeW=grid_sizeW
        self.accelerator=accelerator
        self.video_length=video_length//torch.cuda.device_count()
        device = accelerator.device
        
        self.num_inference_steps=num_inference_steps
        GPU_num=torch.cuda.device_count()
        if method=="FateZero":
            if GPU_num>1:
                GPU_id=dist.get_rank()
                self.attention_path = os.path.join('Attentions', str(GPU_num) + "GPU","GPU"+str(GPU_id), os.path.basename(data_path),str(self.video_length))
                self.latents_path = os.path.join('Latents',str(GPU_num) + "GPU", "GPU"+str(GPU_id), os.path.basename(data_path),str(self.video_length))
            else:
                self.attention_path = os.path.join('Attentions', str(GPU_num) + "GPU", os.path.basename(data_path),str(self.video_length))
                self.latents_path = os.path.join('Latents',str(GPU_num) + "GPU", os.path.basename(data_path),str(self.video_length))
            # latent 由 store_controller 在 step_callback 中落盘（flag=0）
            self.store_controller.latents_path = self.latents_path
            

        # 初始化分布式环境
        
    r"""
    Pipeline for text-to-video generation using Spatio-temp Stable Diffusion.
    """

 
    def encode_in_f_batches(self,image_grid, model):
        batches = torch.split(image_grid,1, dim=0)  # 分批
        encoded_batches = []
        for batch in batches:
            latent_dist = model(batch).latent_dist.sample()
            encoded_batches.append(latent_dist)
        latents_grid = torch.cat(encoded_batches, dim=0)
        return latents_grid
    
    @torch.no_grad()
    def prepare_latents_ddim_inverted(self, image, batch_size,num_images_per_prompt, 
                                        text_embeddings,
                                        store_attention=False, prompt=None,
                                        generator=None,
                                        LOW_RESOURCE = True,
                                        save_path = None,
                                        controlnet_conditioning_scale=1.0,
                                        control_image=None,
                                        scheduler_config=None
                                      ):
 
        self.prepare_before_train_loop()
        #F/C controller初始化
        self.scheduler_config=scheduler_config
        self.inverse_scheduler = DDIMScheduler.from_config(self.scheduler_config)  # 从配置中创建DDIMScheduler对象
        self.inverse_scheduler.set_timesteps(self.num_inference_steps, device=self.device)  # 设置时间步长
        self.timesteps = reversed(self.inverse_scheduler.timesteps)  # 获取倒序的时间步长
        #print(self.timesteps)
        if self.method=="FateZero":
            if store_attention:
                attention_util.register_attention_control(self, self.store_controller)
            resource_default_value = self.store_controller.LOW_RESOURCE
            self.store_controller.LOW_RESOURCE = LOW_RESOURCE  # in inversion, no CFG, record all latents attention

        batch_size = batch_size * num_images_per_prompt
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        image=rearrange(image,"(b f) c h w -> b c f h w", b=1)#[1,3,8,512,512]
        init_latents=grid.combine_frames_in_grid(image,self.grid_sizeH,self.grid_sizeW) 
        init_latents=rearrange(init_latents,"b c f h w -> (b f) c h w")#[2, 3, 1024, 1024]
        init_latents=self.encode_in_f_batches(init_latents,self.vae.encode)
        init_latents = 0.18215 * init_latents
        init_latents = torch.cat([init_latents], dim=0)
        init_latents_bcfhw = rearrange(init_latents, "(b f) c h w -> b c f h w", b=batch_size)
        if self.method=="FateZero":
            os.makedirs(self.latents_path, exist_ok = True)
            os.makedirs(self.attention_path, exist_ok = True)
        if self.method=="ControlNet":
            #C/F control_image需要匹配网格的数据格式
            control_image = rearrange(control_image, "(b f) c h w -> b c f h w", b=batch_size)
            control_image = grid.combine_frames_in_grid(control_image, self.grid_sizeH, self.grid_sizeW)
        
        ddim_latent = self.ddim_clean2noisy_loop(init_latents_bcfhw, text_embeddings, self.store_controller, controlnet_conditioning_scale=controlnet_conditioning_scale,
            control_image=control_image)
        if self.method=="FateZero":
            #F/C controller操作
            if store_attention and (save_path is not None) :
                attention_util.register_attention_control(self, self.empty_controller)
            self.store_controller.LOW_RESOURCE = resource_default_value
        return ddim_latent
    

    def process_with_unet(self,unet, latent_model_input, t, text_embeddings,b_indices, control_image, controlnet_conditioning_scale):
        latent_grids = latent_model_input.split(1, dim=2)
        noise_preds = []

        for i, latent_grid in enumerate(latent_grids):
            if self.method=="FateZero":
                #F/C DDIM Inversion阶段的unet调用
                noise_pred = unet(latent_grid, t, encoder_hidden_states=text_embeddings,indices=b_indices[i]).sample
            elif self.method=="ControlNet":
                #C/F controlnet的unet结构缝合
                latent_grid = latent_grid.squeeze(2)
                latent_grid = latent_grid.to(self.controlnet.dtype)
                control_image=control_image.to(self.controlnet.dtype)
                controlnet_cond = control_image[:, :, i, :, :] if control_image.dim() == 5 else control_image
                
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    latent_grid,
                    t,
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=controlnet_cond,
                    conditioning_scale=controlnet_conditioning_scale,
                    return_dict=False,
                )
                noise_pred = self.unet(
                    latent_grid,
                    t,
                    encoder_hidden_states=text_embeddings,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample
                noise_pred=noise_pred.unsqueeze(2)
            noise_preds.append(noise_pred)
        noise_pred = torch.cat(noise_preds, dim=2)
        return noise_pred

    @torch.no_grad()
    def ddim_clean2noisy_loop(self, latent, text_embeddings, controller:attention_util.AttentionControl=None,controlnet_conditioning_scale=1.0, control_image=None):
        weight_dtype = latent.dtype
        uncond_embeddings, cond_embeddings = text_embeddings.chunk(2)
        latent = latent.clone().detach()
        print('Invert clean image to noise latents by DDIM and Unet')
        #self.scheduler.timesteps = torch.unique(self.scheduler.timesteps)
        
        for i, t in enumerate(tqdm(self.timesteps)):  # 遍历时间步长
            
         
           
            # latent 落盘已移至 AttentionStore.step_callback(flag=0)：保存"本步更新后"的 latent（org 帧布局），
            # 与旧版 latents_store 语义对齐（此处循环开头保存的是"步前"且处于上一步 indicate 布局，配对错一档）
            if torch.cuda.device_count()>1:
                latent,indices= self.alternate_multiGPU(latent,i)
                if self.method=="ControlNet" :
                    control_image,_=self.alternate_multiGPU(control_image,i)
            else:  
                latent,indices= self.alternate(latent,i)
                if self.method=="ControlNet" :
                                    control_image,_=self.alternate(control_image,i)
            if self.method=="FateZero":
                b_indices=self.reassign_values(indices, self.grid_sizeH,self.grid_sizeW)
            elif self.method=="ControlNet":
                #C/F 
                b_indices=None

            noise_pred=self.process_with_unet(self.unet,latent,t,cond_embeddings,b_indices,control_image, controlnet_conditioning_scale)
            latent = self.next_clean2noise_step(noise_pred, t, latent)

            if self.method=="FateZero":
                #F/C
                if controller is not None: 
                    latent=grid.restore_frames_from_grid(latent,self.grid_sizeH,self.grid_sizeW)
                    sqs=torch.argsort(torch.tensor(indices))
                    latent=latent[:, :, sqs, :, :]
                    latent=grid.combine_frames_in_grid(latent,self.grid_sizeH,self.grid_sizeW)
                    latent = controller.step_callback(latent,0,self.attention_path)
                    latent=grid.restore_frames_from_grid(latent,self.grid_sizeH,self.grid_sizeW)
                    latent=latent[:, :, indices, :, :]
                    latent=grid.combine_frames_in_grid(latent,self.grid_sizeH,self.grid_sizeW)
            latent = latent.to(weight_dtype)
            torch.cuda.empty_cache()
        return latent

    
    def next_clean2noise_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        """
        Assume the eta in DDIM=0
        """
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

    
    def get_timesteps(self, num_inference_steps, strength, device):
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start
    
    def p2preplace_edit(self, **kwargs):
       
        control_image = kwargs.pop('control_image', None)
        controlnet_conditioning_scale = kwargs.pop('controlnet_conditioning_scale', 1.0)
        len_source = {len(kwargs['source_prompt'].split(' '))}
        len_target = {len(kwargs['prompt'].split(' '))}
        equal_length = (len_source == len_target)
        # if self.accelerator.is_main_process:
        #     print(f" len_source: {len_source}, len_target: {len_target}, equal_length: {equal_length}")
        if self.method=="ControlNet":
            #C/F
            edit_controller=None
        elif self.method=="FateZero":
            #F/C
            edit_controller = attention_util.make_controller(
                                self.tokenizer, 
                                [ kwargs['source_prompt'], kwargs['prompt']],
                                NUM_DDIM_STEPS = kwargs['num_inference_steps'],
                                is_replace_controller=kwargs.get('is_replace_controller', True) and equal_length,
                                cross_replace_steps=kwargs['cross_replace_steps'], 
                                self_replace_steps=kwargs['self_replace_steps'], 
                                blend_words=kwargs.get('blend_words', None),
                                equilizer_params=kwargs.get('eq_params', None),
                                additional_attention_store=self.store_controller,
                                use_inversion_attention = kwargs['use_inversion_attention'],
                                blend_self_th = kwargs.get('blend_self_th', (0.3, 0.3)),
                                blend_temp_th = kwargs.get('blend_temp_th', (0.3, 0.3)),
                                blend_latents_th = kwargs.get('blend_latents_th', (0.3, 0.3)),
                                blend_self_attention = kwargs.get('blend_self_attention', None),
                                blend_temp_attention = kwargs.get('blend_temp_attention', None),
                                blend_latents=kwargs.get('blend_latents', None),
                                save_path=kwargs.get('save_path', None),
                                save_self_attention = kwargs.get('save_self_attention', True),
                                disk_store = kwargs.get('disk_store', False),
                                attention_path=self.attention_path,  # 传递参数
                                fixed_mask=getattr(self, 'fixed_mask', None),  # 外接掩码（GroundingDINO+FastSAM）
                                fuse_target_attention=getattr(self, 'fuse_target_attention', False)  # Option C: 外接mask∪cross-attention mask
                                )
            attention_util.register_attention_control(self, edit_controller)
            

        # In ddim inferece, no need source prompt
        sdimage_output = self.sd_ddim_pipeline(
            controller = edit_controller,
            control_image=control_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale, 
            # target_prompt = kwargs['prompts'][1],
            **kwargs)
        if self.method=="FateZero":
            #C/F edit_controller的
            if hasattr(edit_controller.latent_blend, 'mask_list'):
                mask_list = edit_controller.latent_blend.mask_list
            else:
                mask_list = None
        elif self.method=="ControlNet":
            #C/F
            mask_list = None

        attention_output = None
        dict_output = {
                "sdimage_output" : sdimage_output,
                "attention_output" : attention_output,
                "mask_list" : mask_list,
            }
        if self.method=="FateZero":
            #F/C
            attention_util.register_attention_control(self, self.empty_controller)
        return dict_output
    
    def alternate(self,latents,flag):

        org = list(range(self.video_length))
    
        alt = org[self.grid_sizeH:] + org[:self.grid_sizeH]
        # alt=org
        if self.video_length==self.grid_sizeH*self.grid_sizeW:
            indicate=org
            sqc=org
        elif flag%2==0 and self.video_length!=self.grid_sizeH*self.grid_sizeW:
            indicate=alt
            sqc=alt
        else:
            indicate = org
            sqc=torch.argsort(torch.tensor(alt))
        latents=grid.restore_frames_from_grid(latents,self.grid_sizeH,self.grid_sizeW)
        latents=latents[:, :, sqc, :, :]
        latents=grid.combine_frames_in_grid(latents,self.grid_sizeH,self.grid_sizeW)
        return latents,indicate
    
    def alternate_multiGPU(self, latents, flag):
        org = list(range(self.video_length))
        alt = org[self.grid_sizeH:] + org[:self.grid_sizeH]
        rank = dist.get_rank()
        device = self.accelerator.device  # 确保设备一致
        
        # if self.video_length == self.grid_sizeH *self.grid_sizeW:
        #     indicate = org
        #     sqc = org
        if flag % 2 == 0:
            # 偶数变
            sqc = alt
            if rank == 0:
                indicate = org
                latents = grid.restore_frames_from_grid(latents, self.grid_sizeH,self.grid_sizeW)
                data = latents[:, :, :self.grid_sizeH, :, :].to(device)  # 确保设备一致
                data1 = torch.zeros_like(data)
                data = data.contiguous()  # 确保张量是连续的
                latents = latents[:, :, self.grid_sizeH:, :, :]
                dist.send(data, dst=1)  # Rank 0 发送数据到 Rank 1
                dist.recv(data1, src=1)  # Rank 0 接收来自 Rank 1 的数据
                latents = torch.cat([latents,data1], dim=2).to(device)
                latents = grid.combine_frames_in_grid(latents, self.grid_sizeH,self.grid_sizeW)
            elif rank == 1:
                indicate = alt
                latents = grid.restore_frames_from_grid(latents, self.grid_sizeH,self.grid_sizeW)
                data = latents[:, :, :self.grid_sizeH, :, :].to(device)  # 确保设备一致
                data1 = torch.zeros_like(data)
                data = data.contiguous()  # 确保张量是连续的
                latents = latents[:, :, self.grid_sizeH:, :, :]
                dist.recv(data1, src=0)  # Rank 1 接收来自 Rank 0 的数据
                dist.send(data, dst=0)  # Rank 1 发送数据到 Rank 0
                latents = torch.cat([latents,data1], dim=2).to(device)
                latents = grid.combine_frames_in_grid(latents, self.grid_sizeH,self.grid_sizeW)
        else:
            # 奇数不变
            indicate = org
            sqc = torch.argsort(torch.tensor(alt))
            if rank == 0:
                latents = grid.restore_frames_from_grid(latents, self.grid_sizeH,self.grid_sizeW)
                data = latents[:, :, -self.grid_sizeH:, :, :].to(device)  # 确保设备一致
                data1 = torch.zeros_like(data)
                data = data.contiguous()  # 确保张量是连续的
                latents = latents[:, :, :-self.grid_sizeH, :, :]
                dist.send(data, dst=1)  # Rank 0 发送数据到 Rank 1
                dist.recv(data1, src=1)  # Rank 0 接收来自 Rank 1 的数据
                latents = torch.cat([data1,latents], dim=2).to(device)
                latents = grid.combine_frames_in_grid(latents, self.grid_sizeH,self.grid_sizeW)
            elif rank == 1:
                latents = grid.restore_frames_from_grid(latents, self.grid_sizeH,self.grid_sizeW)
                data = latents[:, :, -self.grid_sizeH:, :, :].to(device)  # 确保设备一致
                data1 = torch.zeros_like(data)
                data = data.contiguous()  # 确保张量是连续的
                latents = latents[:, :, :-self.grid_sizeH, :, :]
                dist.recv(data1, src=0)  # Rank 1 接收来自 Rank 0 的数据
                dist.send(data, dst=0)  # Rank 1 发送数据到 Rank 0
                latents = torch.cat([data1,latents], dim=2).to(device)
                latents = grid.combine_frames_in_grid(latents, self.grid_sizeH,self.grid_sizeW)
        dist.barrier()  # 同步所有进程
        
        return latents, indicate

    @torch.no_grad()
    def __call__(self, **kwargs):
        edit_type = kwargs['edit_type']
        control_image = kwargs.pop('control_image', None)
        controlnet_conditioning_scale = kwargs.pop('controlnet_conditioning_scale', 1.0)
        assert edit_type in ['save', 'swap', None]
        if edit_type == 'swap':
            return self.p2preplace_edit(control_image=control_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,**kwargs)

    @torch.no_grad()
    def sd_ddim_pipeline(
        self,
        prompt: Union[str, List[str]],
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        control_image: Union[torch.FloatTensor, PIL.Image.Image] = None,  # 添加控制图像参数
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        controller: attention_util.AttentionControl = None,
        controlnet_conditioning_scale: float = 1.0,
        **args
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )
        
        # 4. Prepare timesteps
        self.scheduler = DDIMScheduler.from_config(self.scheduler_config)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        
        #self.scheduler.timesteps = torch.unique(self.scheduler.timesteps)
        timesteps = self.scheduler.timesteps
        # 5. Prepare control image
        if control_image is not None:
            control_image = control_image.to(device=device, dtype=text_embeddings.dtype)
            if do_classifier_free_guidance:
                control_image = torch.cat([control_image] * 2)

        latents_dtype = latents.dtype
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        if self.method =="ControlNet":
            #C/F
            control_image = rearrange(control_image, "(b f) c h w -> b c f h w", b=2)
            control_image = grid.combine_frames_in_grid(control_image, self.grid_sizeH, self.grid_sizeW)
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in tqdm(enumerate(self.scheduler.timesteps), desc='reverse_diffusion'):
                #C/F
                if torch.cuda.device_count()>1:
                    latents,indices= self.alternate_multiGPU(latents,i)
                    if self.method =="ControlNet" :
                        control_image,_= self.alternate_multiGPU(control_image,i)
                else:
                    latents,indices= self.alternate(latents,i)
                    if self.method =="ControlNet" :
                        control_image,_= self.alternate(control_image,i)
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                if self.method =="FateZero":
                    #C/F
                    b_indices=self.reassign_values(indices, self.grid_sizeH,self.grid_sizeW)
                # predict the noise residual
                latent_grids_in_unet = latent_model_input.split(1, dim=2)
                latent_grid=latents.split(1, dim=2)
                # 存储每个网格批次的处理结果
                latents_b= []
                # 逐批次处理每个单帧网格
                
              
                for j, latent_grid_in_unet in enumerate(latent_grids_in_unet):
                    if controller is not None:
                        controller.cur_grid_idx = j   # tell the self-attn mask which grid/frame is being processed (grid 1x1 vs 2x2 adaptive)
                    if self.method =="ControlNet":
                        #C/F
                        latent_grid_in_unet = latent_grid_in_unet.squeeze(2)
                        latent_grid_in_unet=latent_grid_in_unet.to(self.controlnet.dtype)
                        control_image=control_image.to(self.controlnet.dtype)
                        controlnet_cond = None
                        if control_image is not None:
                            controlnet_cond = control_image[:, :, j, :, :] if control_image.dim() == 5 else control_image
                        if self.controlnet is not None and controlnet_cond is not None:
                            down_block_res_samples, mid_block_res_sample = self.controlnet(
                                latent_grid_in_unet,
                                t,
                                encoder_hidden_states=text_embeddings,
                                controlnet_cond=controlnet_cond,
                                conditioning_scale=controlnet_conditioning_scale,
                                return_dict=False,
                            )
                            noise_pred = self.unet(
                                latent_grid_in_unet,
                                t,
                                encoder_hidden_states=text_embeddings,
                                down_block_additional_residuals=down_block_res_samples,
                                mid_block_additional_residual=mid_block_res_sample,
                            ).sample
                        noise_pred=noise_pred.unsqueeze(2).to(dtype=latents_dtype)
                    elif self.method =="FateZero":
                        #F/C
                        noise_pred = self.unet(latent_grid_in_unet, t, encoder_hidden_states=text_embeddings, indices=b_indices[j]).sample.to(dtype=latents_dtype)
                    if do_classifier_free_guidance:
                       noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                       noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                    # compute the previous noisy sample x_t -> x_t-1
                    latent= self.scheduler.step(noise_pred, t, latent_grid[j], **extra_step_kwargs).prev_sample
                    
                    weight_type = latents.dtype
                    latents_b.append(latent)
                latents = torch.cat(latents_b, dim=2)
                if self.method=="FateZero":
                    #F/C
                    if controller is not None:
                                latents=grid.restore_frames_from_grid(latents,self.grid_sizeH,self.grid_sizeW)
                                sqs=torch.argsort(torch.tensor(indices))
                                latents=latents[:, :, sqs, :, :]
                                latents=grid.combine_frames_in_grid(latents,self.grid_sizeH,self.grid_sizeW)
                                latents = controller.step_callback(latents,self.num_inference_steps-i-1,self.latents_path).to(device, dtype=weight_type)
                                latents=grid.restore_frames_from_grid(latents,self.grid_sizeH,self.grid_sizeW)
                                latents=latents[:, :, indices, :, :]
                                latents=grid.combine_frames_in_grid(latents,self.grid_sizeH,self.grid_sizeW)
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
                torch.cuda.empty_cache()
        # 8. Post-processing
        latents = self.decode_in_f_batches(latents)
        image = grid.restore_frames_from_grid(latents,self.grid_sizeH,self.grid_sizeW)
        image = rearrange(image, "b c f h w -> b f h w c")
        # 9. Run safety checker
        has_nsfw_concept = None
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().float().numpy()
        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)
        if not return_dict:
            return (image, has_nsfw_concept)
        torch.cuda.empty_cache()
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
    




    def decode_in_f_batches(self, image_grid):  
            batches = torch.split(image_grid, 1, dim=2)  # 分批
            decoded_batches = []
            for batch in batches:
                latent_dist = self.decode_latents(batch,self.grid_sizeH,self.grid_sizeW)
                decoded_batches.append(latent_dist)
            latents_grid = torch.cat(decoded_batches, dim=2)
            return latents_grid
    
    def reassign_values(self, indices, grid_sizeH,grid_sizeW):
        result = []
        sublist_size = (grid_sizeH*grid_sizeW)
        # Step 1: Split the indices list into sublists
        sublists = [indices[i:i + sublist_size] for i in range(0, len(indices), sublist_size)]
        for sublist in sublists:
            # Step 2: Sort the sublist and assign new values (0, 1, 2, ...)
            sorted_sublist = sorted(sublist)
            reassigned_sublist = [sorted_sublist.index(x) for x in sublist]
            # Step 3: Add the new sublist to the result
            result.append(reassigned_sublist)
        return result

