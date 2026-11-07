
from typing import Callable, List, Optional, Union
import os

import PIL
import torch
import torch.distributed as dist
import grid
import numpy as np
from einops import rearrange
from tqdm import tqdm

from accelerate import Accelerator
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.utils import logging
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.models import AutoencoderKL
from diffusers.schedulers import (DDIMScheduler)

from .stable_diffusion import SpatiotempStableDiffusionPipeline

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class P2pDDIMSpatiotempPipeline(SpatiotempStableDiffusionPipeline):
    r"""ControlNet inverting-and-editing pipeline under the grid mechanism."""

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet,
        scheduler: Union[DDIMScheduler],
        controlnet,
        method: str = "ControlNet",
        control_class: str = None,
        disk_store: bool = False,
        video_length: int = 8,
        grid_sizeH: int = 2,
        grid_sizeW: int = 2,
        accelerator: Accelerator = None,
        data_path: str = "",
        num_inference_steps: int = 0,
        **kwargs,
    ):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler)
        self.method = method
        self.control_class = control_class
        self.scheduler = scheduler
        self.controlnet = controlnet
        self.grid_sizeH = grid_sizeH
        self.grid_sizeW = grid_sizeW
        self.accelerator = accelerator
        self.num_inference_steps = num_inference_steps
        self.video_length = video_length // torch.cuda.device_count()

    def encode_in_f_batches(self, image_grid, model):
        batches = torch.split(image_grid, 1, dim=0)
        encoded_batches = []
        for batch in batches:
            latent_dist = model(batch).latent_dist.sample()
            encoded_batches.append(latent_dist)
        latents_grid = torch.cat(encoded_batches, dim=0)
        return latents_grid

    @torch.no_grad()
    def prepare_latents_ddim_inverted(self, image, batch_size, num_images_per_prompt,
                                      text_embeddings,
                                      prompt=None,
                                      generator=None,
                                      LOW_RESOURCE=True,
                                      save_path=None,
                                      controlnet_conditioning_scale=1.0,
                                      control_image=None,
                                      scheduler_config=None):
        self.prepare_before_train_loop()
        self.scheduler_config = scheduler_config
        self.inverse_scheduler = DDIMScheduler.from_config(self.scheduler_config)
        self.inverse_scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        self.timesteps = reversed(self.inverse_scheduler.timesteps)

        batch_size = batch_size * num_images_per_prompt
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        image = rearrange(image, "(b f) c h w -> b c f h w", b=1)  # [1,3,8,512,512]
        n_real = image.shape[2]
        gA = self.grid_sizeH * self.grid_sizeW
        n_pad = (-n_real) % gA
        if n_pad:
            print(f'[ControlNet] frame count {n_real} not a multiple of {gA}; padded by repeating last frame to {n_real + n_pad}')
            image = torch.cat([image, image[:, :, -1:, :, :].expand(-1, -1, n_pad, -1, -1)], dim=2)
        self._n_real = n_real
        self.video_length = n_real + n_pad
        init_latents = grid.combine_frames_in_grid(image, self.grid_sizeH, self.grid_sizeW)
        init_latents = rearrange(init_latents, "b c f h w -> (b f) c h w")  # [2, 3, 1024, 1024]
        init_latents = self.encode_in_f_batches(init_latents, self.vae.encode)
        init_latents = 0.18215 * init_latents
        init_latents = torch.cat([init_latents], dim=0)
        init_latents_bcfhw = rearrange(init_latents, "(b f) c h w -> b c f h w", b=batch_size)

        control_image = rearrange(control_image, "(b f) c h w -> b c f h w", b=batch_size)
        _n_pad_ctrl = (-control_image.shape[2]) % gA
        if _n_pad_ctrl:
            control_image = torch.cat([control_image, control_image[:, :, -1:, :, :].expand(-1, -1, _n_pad_ctrl, -1, -1)], dim=2)
        control_image = grid.combine_frames_in_grid(control_image, self.grid_sizeH, self.grid_sizeW)

        ddim_latent = self.ddim_clean2noisy_loop(
            init_latents_bcfhw, text_embeddings,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            control_image=control_image)
        return ddim_latent

    def process_with_unet(self, unet, latent_model_input, t, text_embeddings,
                          control_image, controlnet_conditioning_scale):
        latent_grids = latent_model_input.split(1, dim=2)
        noise_preds = []
        for i, latent_grid in enumerate(latent_grids):
            latent_grid = latent_grid.squeeze(2)
            latent_grid = latent_grid.to(self.controlnet.dtype)
            control_image = control_image.to(self.controlnet.dtype)
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
            noise_pred = noise_pred.unsqueeze(2)
            noise_preds.append(noise_pred)
        noise_pred = torch.cat(noise_preds, dim=2)
        return noise_pred

    @torch.no_grad()
    def ddim_clean2noisy_loop(self, latent, text_embeddings,
                              controlnet_conditioning_scale=1.0, control_image=None):
        weight_dtype = latent.dtype
        uncond_embeddings, cond_embeddings = text_embeddings.chunk(2)
        latent = latent.clone().detach()
        print('Invert clean image to noise latents by DDIM and Unet')

        for i, t in enumerate(tqdm(self.timesteps)):
            if torch.cuda.device_count() > 1:
                latent, _ = self.alternate_multiGPU(latent, i)
                control_image, _ = self.alternate_multiGPU(control_image, i)
            else:
                latent, _ = self.alternate(latent, i)
                control_image, _ = self.alternate(control_image, i)

            noise_pred = self.process_with_unet(self.unet, latent, t, cond_embeddings,
                                                control_image, controlnet_conditioning_scale)
            latent = self.next_clean2noise_step(noise_pred, t, latent)
            latent = latent.to(weight_dtype)
            torch.cuda.empty_cache()
        return latent

    def next_clean2noise_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        """Assume the eta in DDIM=0"""
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

    def alternate(self, latents, flag):
        org = list(range(self.video_length))
        alt = org[self.grid_sizeH:] + org[:self.grid_sizeH]
        if self.video_length == self.grid_sizeH * self.grid_sizeW:
            indicate = org
            sqc = org
        elif flag % 2 == 0 and self.video_length != self.grid_sizeH * self.grid_sizeW:
            indicate = alt
            sqc = alt
        else:
            indicate = org
            sqc = torch.argsort(torch.tensor(alt))
        latents = grid.restore_frames_from_grid(latents, self.grid_sizeH, self.grid_sizeW)
        latents = latents[:, :, sqc, :, :]
        latents = grid.combine_frames_in_grid(latents, self.grid_sizeH, self.grid_sizeW)
        return latents, indicate

    def alternate_multiGPU(self, latents, flag):
        org = list(range(self.video_length))
        alt = org[self.grid_sizeH:] + org[:self.grid_sizeH]
        rank = dist.get_rank()
        device = self.accelerator.device

        if flag % 2 == 0:
            sqc = alt
            indicate = org if rank == 0 else alt
            latents = grid.restore_frames_from_grid(latents, self.grid_sizeH, self.grid_sizeW)
            data = latents[:, :, :self.grid_sizeH, :, :].to(device)
            data1 = torch.zeros_like(data)
            data = data.contiguous()
            latents = latents[:, :, self.grid_sizeH:, :, :]
            if rank == 0:
                dist.send(data, dst=1)
                dist.recv(data1, src=1)
            else:
                dist.recv(data1, src=0)
                dist.send(data, dst=0)
            latents = torch.cat([latents, data1], dim=2).to(device)
            latents = grid.combine_frames_in_grid(latents, self.grid_sizeH, self.grid_sizeW)
        else:
            indicate = org
            sqc = torch.argsort(torch.tensor(alt))
            latents = grid.restore_frames_from_grid(latents, self.grid_sizeH, self.grid_sizeW)
            data = latents[:, :, -self.grid_sizeH:, :, :].to(device)
            data1 = torch.zeros_like(data)
            data = data.contiguous()
            latents = latents[:, :, :-self.grid_sizeH, :, :]
            if rank == 0:
                dist.send(data, dst=1)
                dist.recv(data1, src=1)
            else:
                dist.recv(data1, src=0)
                dist.send(data, dst=0)
            latents = torch.cat([data1, latents], dim=2).to(device)
            latents = grid.combine_frames_in_grid(latents, self.grid_sizeH, self.grid_sizeW)
        dist.barrier()
        return latents, indicate

    @torch.no_grad()
    def __call__(self, **kwargs):
        edit_type = kwargs['edit_type']
        assert edit_type in ['save', 'swap', None]
        if edit_type == 'swap':
            kwargs.pop('edit_type')
            control_image = kwargs.pop('control_image', None)
            controlnet_conditioning_scale = kwargs.pop('controlnet_conditioning_scale', 1.0)
            sdimage_output = self.sd_ddim_pipeline(control_image=control_image,
                                                   controlnet_conditioning_scale=controlnet_conditioning_scale,
                                                   **kwargs)
            return {"sdimage_output": sdimage_output, "attention_output": None, "mask_list": None}

    @torch.no_grad()
    def sd_ddim_pipeline(
        self,
        prompt: Union[str, List[str]],
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        control_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
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
        controlnet_conditioning_scale: float = 1.0,
        **args
    ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        text_embeddings = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        self.scheduler = DDIMScheduler.from_config(self.scheduler_config)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        if control_image is not None:
            control_image = control_image.to(device=device, dtype=text_embeddings.dtype)
            if do_classifier_free_guidance:
                control_image = torch.cat([control_image] * 2)

        latents_dtype = latents.dtype
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        control_image = rearrange(control_image, "(b f) c h w -> b c f h w", b=2)
        _n_pad_c = (-control_image.shape[2]) % (self.grid_sizeH * self.grid_sizeW)
        if _n_pad_c:
            print(f'[ControlNet] editing-stage control_image frames {control_image.shape[2]} padded to '
                  f'{control_image.shape[2] + _n_pad_c}')
            control_image = torch.cat([control_image, control_image[:, :, -1:, :, :].expand(-1, -1, _n_pad_c, -1, -1)], dim=2)
        control_image = grid.combine_frames_in_grid(control_image, self.grid_sizeH, self.grid_sizeW)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(tqdm(self.scheduler.timesteps, desc='reverse_diffusion')):
                if torch.cuda.device_count() > 1:
                    latents, _ = self.alternate_multiGPU(latents, i)
                    control_image, _ = self.alternate_multiGPU(control_image, i)
                else:
                    latents, _ = self.alternate(latents, i)
                    control_image, _ = self.alternate(control_image, i)

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                latent_grids_in_unet = latent_model_input.split(1, dim=2)
                latent_grid = latents.split(1, dim=2)
                latents_b = []

                for j, latent_grid_in_unet in enumerate(latent_grids_in_unet):
                    latent_grid_in_unet = latent_grid_in_unet.squeeze(2)
                    latent_grid_in_unet = latent_grid_in_unet.to(self.controlnet.dtype)
                    control_image = control_image.to(self.controlnet.dtype)
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
                    noise_pred = noise_pred.unsqueeze(2).to(dtype=latents_dtype)

                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    latent = self.scheduler.step(noise_pred, t, latent_grid[j], **extra_step_kwargs).prev_sample
                    latents_b.append(latent)
                latents = torch.cat(latents_b, dim=2)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
                torch.cuda.empty_cache()

        latents = self.decode_in_f_batches(latents)
        image = grid.restore_frames_from_grid(latents, self.grid_sizeH, self.grid_sizeW)
        if getattr(self, '_n_real', None) is not None and image.shape[2] > self._n_real:
            image = image[:, :, :self._n_real, :, :]
        image = rearrange(image, "b c f h w -> b f h w c")

        has_nsfw_concept = None
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().float().numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)
        if not return_dict:
            return (image, has_nsfw_concept)
        torch.cuda.empty_cache()
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    def decode_in_f_batches(self, image_grid):
        batches = torch.split(image_grid, 1, dim=2)
        decoded_batches = []
        for batch in batches:
            decoded_batches.append(self.decode_latents(batch, self.grid_sizeH, self.grid_sizeW))
        return torch.cat(decoded_batches, dim=2)
