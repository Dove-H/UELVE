
from typing import Callable, List, Optional, Union
import os

import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from tqdm import tqdm

from diffusers.utils import logging
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.models import AutoencoderKL
from diffusers.schedulers import DDIMScheduler

import grid

from .stable_diffusion import SpatiotempStableDiffusionPipeline
from video_diffusion.prompt_attention import fatezero_register
from video_diffusion.prompt_attention.fatezero_store import InversionStore, ReplayProvider
from video_diffusion.prompt_attention.fatezero_util import (
    EmptyControl, compute_need_steps, make_controller)
from video_diffusion.elve_timing import timer

logger = logging.get_logger(__name__)


class P2pDDIMFateZeroPipeline(SpatiotempStableDiffusionPipeline):
    r"""FateZero inverting-and-editing pipeline under the 2x2 grid mechanism."""

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder,
        tokenizer,
        unet,
        scheduler: DDIMScheduler,
        controlnet=None,
        method: str = "FateZero",
        control_class: str = None,
        disk_store: bool = False,
        video_length: int = 8,
        grid_sizeH: int = 2,
        grid_sizeW: int = 2,
        accelerator=None,
        data_path: str = "",
        num_inference_steps: int = 0,
        memory_frames: int = 16,
        All_Inversion: bool = False,
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
        self.memory_frames = memory_frames
        self.All_Inversion = All_Inversion
        self.num_inference_steps = num_inference_steps
        self.empty_controller = EmptyControl()

        self._video_length_total = video_length
        self.video_length = video_length // torch.cuda.device_count()
        GPU_num = torch.cuda.device_count()
        base = os.path.basename(data_path.rstrip('/')) if data_path else 'unknown'
        if GPU_num > 1:
            self.store_dir = os.path.join('Attentions', f'{GPU_num}GPU',
                                          f'GPU{dist.get_rank()}', base, str(self.video_length))
        else:
            self.store_dir = os.path.join('Attentions', f'{GPU_num}GPU', base, str(self.video_length))

        self._inv_store = None
        self._need = None
        self._num_grids = 0
        self._num_inv_steps = 0
        self._n_real = None

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def alternate(self, latents, flag):
        org = list(range(self.video_length))
        alt = org[self.grid_sizeH:] + org[:self.grid_sizeH]
        if self.video_length == self.grid_sizeH * self.grid_sizeW:
            indicate, sqc = org, org
        elif flag % 2 == 0:
            indicate, sqc = alt, alt
        else:
            indicate, sqc = org, torch.argsort(torch.tensor(alt))
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
            data = latents[:, :, :self.grid_sizeH, :, :].to(device).contiguous()
            data1 = torch.zeros_like(data)
            latents = latents[:, :, self.grid_sizeH:, :, :]
            if rank == 0:
                with timer('comm_send'):
                    dist.send(data, dst=1)
                with timer('comm_recv'):
                    dist.recv(data1, src=1)
            else:
                with timer('comm_recv'):
                    dist.recv(data1, src=0)
                with timer('comm_send'):
                    dist.send(data, dst=0)
            latents = torch.cat([latents, data1], dim=2).to(device)
            latents = grid.combine_frames_in_grid(latents, self.grid_sizeH, self.grid_sizeW)
        else:
            indicate = org
            sqc = torch.argsort(torch.tensor(alt))
            latents = grid.restore_frames_from_grid(latents, self.grid_sizeH, self.grid_sizeW)
            data = latents[:, :, -self.grid_sizeH:, :, :].to(device).contiguous()
            data1 = torch.zeros_like(data)
            latents = latents[:, :, :-self.grid_sizeH, :, :]
            if rank == 0:
                with timer('comm_send'):
                    dist.send(data, dst=1)
                with timer('comm_recv'):
                    dist.recv(data1, src=1)
            else:
                with timer('comm_recv'):
                    dist.recv(data1, src=0)
                with timer('comm_send'):
                    dist.send(data, dst=0)
            latents = torch.cat([data1, latents], dim=2).to(device)
            latents = grid.combine_frames_in_grid(latents, self.grid_sizeH, self.grid_sizeW)
        with timer('comm_barrier'):
            dist.barrier()
        return latents, indicate

    def _want_alt(self, step: int) -> bool:
        return (self._video_length_total > self.grid_sizeH * self.grid_sizeW) and (step % 2 == 0)

    def _group_shuffle(self, latents, flag):
        if torch.cuda.device_count() > 1:
            with timer('comm_exchange_total'):
                return self.alternate_multiGPU(latents, flag)
        return self.alternate(latents, flag)

    @staticmethod
    def reassign_values(indices, grid_sizeH, grid_sizeW):
        result = []
        sublist_size = (grid_sizeH * grid_sizeW)
        sublists = [indices[i:i + sublist_size] for i in range(0, len(indices), sublist_size)]
        for sublist in sublists:
            sorted_sublist = sorted(sublist)
            result.append([sorted_sublist.index(x) for x in sublist])
        return result

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def encode_in_f_batches(self, image_grid, model):
        batches = torch.split(image_grid, 1, dim=0)
        encoded_batches = []
        for batch in batches:
            latent_dist = model(batch).latent_dist.sample()
            encoded_batches.append(latent_dist)
        return torch.cat(encoded_batches, dim=0)

    def decode_in_f_batches(self, image_grid):
        batches = torch.split(image_grid, 1, dim=2)
        decoded_batches = []
        for batch in batches:
            decoded_batches.append(self.decode_latents(batch, self.grid_sizeH, self.grid_sizeW))
        return torch.cat(decoded_batches, dim=2)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prepare_latents_ddim_inverted(self, image, text_embeddings, scheduler_config,
                                      editing_params: dict, num_inference_steps: int,
                                      generator=None, LOW_RESOURCE: bool = True):
        self.prepare_before_train_loop()
        self.scheduler_config = scheduler_config
        self.inverse_scheduler = DDIMScheduler.from_config(self.scheduler_config)
        self.inverse_scheduler.set_timesteps(num_inference_steps, device=self.device)
        self.timesteps = list(reversed(self.inverse_scheduler.timesteps))
        self.num_inference_steps = num_inference_steps

        image = rearrange(image, "(b f) c h w -> b c f h w", b=1)
        n_real = image.shape[2]
        gA = self.grid_sizeH * self.grid_sizeW
        n_pad = (-n_real) % gA
        if n_pad:
            print(f'[FateZero] 帧数 {n_real} 非 {gA} 整数倍，复制末帧补齐到 {n_real + n_pad}')
            image = torch.cat([image, image[:, :, -1:, :, :].expand(-1, -1, n_pad, -1, -1)], dim=2)
        self._n_real = n_real
        self.video_length = n_real + n_pad
        num_grids = self.video_length // gA

        init_latents = grid.combine_frames_in_grid(image, self.grid_sizeH, self.grid_sizeW)
        init_latents = rearrange(init_latents, "b c f h w -> (b f) c h w")
        init_latents = self.encode_in_f_batches(init_latents, self.vae.encode)
        init_latents = 0.18215 * init_latents
        init_latents_bcfhw = rearrange(init_latents, "(b f) c h w -> b c f h w", b=1)

        prompts = [editing_params['source_prompt'], editing_params['prompt']]
        latent_replace_steps = editing_params.get('latent_replace_steps', (0.2, 0.8))
        if isinstance(latent_replace_steps, (int, float)):
            latent_replace_steps = (0.0, float(latent_replace_steps))
        latent_replace_steps = (float(latent_replace_steps[0]), float(latent_replace_steps[1]))
        need = compute_need_steps(prompts, num_inference_steps,
                                  editing_params.get('cross_replace_steps'),
                                  editing_params.get('self_replace_steps', 0.0),
                                  editing_params.get('blend_latents', False),
                                  self.tokenizer,
                                  latent_replace_steps=latent_replace_steps)
        meta = dict(source_prompt=prompts[0], num_steps=num_inference_steps,
                    num_grids=num_grids, video_length=n_real,
                    gridH=self.grid_sizeH, gridW=self.grid_sizeW,
                    latent_replace_steps=latent_replace_steps,
                    alt_sliding=self._want_alt(0))
        self._latent_replace_steps = latent_replace_steps
        self._need = need
        self._num_grids = num_grids
        self._num_inv_steps = num_inference_steps

        keep_on_gpu = self.video_length <= self.memory_frames
        print(f'[FateZero] 帧数(含补齐)={self.video_length} memory_frames={self.memory_frames} '
              f'→ 特征{"驻留GPU" if keep_on_gpu else "异步落盘+编辑期预取"}')

        if (not self.All_Inversion) and InversionStore.cache_valid(self.store_dir, meta, need):
            print(f'[FateZero] 缓存命中，跳过反演: {self.store_dir}')
            self._inv_store = None
            init_latent = InversionStore.load_cached(self.store_dir)
            return init_latent.to(device=self._execution_device,
                                  dtype=init_latents_bcfhw.dtype)

        store = InversionStore(store_dir=self.store_dir, need=need, keep_on_gpu=keep_on_gpu)
        store.set_meta(meta)
        fatezero_register.register_attention_control(self, store)
        latent = self.ddim_clean2noisy_loop(init_latents_bcfhw, text_embeddings, store)
        fatezero_register.register_attention_control(self, self.empty_controller)
        store.finish(latent)
        self._inv_store = store
        return store.init_latent

    def process_with_unet(self, unet, latent_model_input, t, text_embeddings, b_indices, store):
        noise_preds = []
        for j, latent_grid in enumerate(latent_model_input.split(1, dim=2)):
            store.begin_grid()
            noise_pred = unet(latent_grid, t, encoder_hidden_states=text_embeddings,
                              indices=b_indices[j]).sample
            store.end_grid(j)
            noise_preds.append(noise_pred)
        return torch.cat(noise_preds, dim=2)

    @torch.no_grad()
    def ddim_clean2noisy_loop(self, latent, text_embeddings, store: InversionStore):
        weight_dtype = latent.dtype
        _, cond_embeddings = text_embeddings.chunk(2)
        latent = latent.clone().detach()
        print('[FateZero] Invert clean image to noise latents by DDIM and Unet')

        for i, t in enumerate(tqdm(self.timesteps)):
            store.begin_step(i)
            want_alt = self._want_alt(i)
            if want_alt:
                latent, indices = self._group_shuffle(latent, 0)
            else:
                indices = list(range(self.video_length))
            b_indices = self.reassign_values(indices, self.grid_sizeH, self.grid_sizeW)

            noise_pred = self.process_with_unet(self.unet, latent, t, cond_embeddings,
                                                b_indices, store)
            latent = self.next_clean2noise_step(noise_pred, t, latent)

            if want_alt:
                latent, _ = self._group_shuffle(latent, 1)
            latent = store.step_callback(latent, i)
            latent = latent.to(weight_dtype)
            torch.cuda.empty_cache()
        return latent

    def next_clean2noise_step(self, model_output, timestep, sample):
        scheduler = self.inverse_scheduler
        timestep, next_timestep = min(
            timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = scheduler.alphas_cumprod[timestep] if timestep >= 0 else scheduler.final_alpha_cumprod
        alpha_prod_t_next = scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        return alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(self, prompt, source_prompt, edit_type=None, latents=None,
                 num_inference_steps: int = 50, guidance_scale: float = 7.5,
                 num_images_per_prompt: int = 1, eta: float = 0.0, generator=None,
                 output_type: str = "pil", return_dict: bool = True,
                 strength=None, clip_length=None, image=None, save_path=None,
                 control_image=None, controlnet_conditioning_scale=1.0,
                 uncond_embeddings_list=None, callback=None, callback_steps=1,
                 **p2p_config):
        assert edit_type in ['swap', None], f'unsupported edit_type {edit_type}'
        return self.p2preplace_edit(prompt=prompt, source_prompt=source_prompt,
                                    latents=latents, num_inference_steps=num_inference_steps,
                                    guidance_scale=guidance_scale, eta=eta, generator=generator,
                                    output_type=output_type, return_dict=return_dict,
                                    save_path=save_path, **p2p_config)

    def _build_replay_provider(self, device, dtype=torch.float16):
        store = self._inv_store
        if store is not None and store.keep_on_gpu:
            load_shard = lambda s, j: store.shards.get((s, j))
            load_latent = lambda s: store.latents.get(s)
        else:
            d = self.store_dir
            load_shard = lambda s, j: torch.load(
                os.path.join(d, f'step{s:03d}_grid{j:02d}.pt'), map_location='cpu')
            load_latent = lambda s: torch.load(
                os.path.join(d, f'lat{s:03d}.pt'), map_location='cpu')
        return ReplayProvider(self._num_inv_steps, self._num_grids, self._need,
                              load_shard, load_latent, device, dtype)

    def p2preplace_edit(self, prompt, source_prompt, latents, num_inference_steps,
                        guidance_scale=7.5, eta=0.0, generator=None, output_type="pil",
                        return_dict=True, save_path=None, **cfg):
        prompts = [source_prompt, prompt]
        equal_length = len(source_prompt.split()) == len(prompt.split())
        is_replace_controller = cfg.get('is_replace_controller', True) and equal_length

        blend_th_legacy = cfg.get('blend_th', None)
        blend_self_th = cfg.get('blend_self_th', blend_th_legacy if blend_th_legacy is not None else (0.3, 0.3))
        blend_latents_th = cfg.get('blend_latents_th', blend_th_legacy if blend_th_legacy is not None else (0.3, 0.3))
        latent_replace_steps = cfg.get('latent_replace_steps',
                                       getattr(self, '_latent_replace_steps', (0.2, 0.8)))

        provider = self._build_replay_provider(self._execution_device)
        edit_controller = make_controller(
            self.tokenizer, prompts,
            NUM_DDIM_STEPS=num_inference_steps,
            is_replace_controller=is_replace_controller,
            cross_replace_steps=cfg.get('cross_replace_steps', {'default_': 0.5}),
            self_replace_steps=cfg.get('self_replace_steps', 0.0),
            blend_words=cfg.get('blend_words', None),
            equilizer_params=cfg.get('eq_params', None),
            replay_provider=provider,
            use_inversion_attention=cfg.get('use_inversion_attention', True),
            blend_self_th=blend_self_th,
            blend_latents_th=blend_latents_th,
            latent_replace_steps=latent_replace_steps,
            blend_latents=cfg.get('blend_latents', False),
            blend_self_attention=cfg.get('blend_self_attention', False),
            save_path=save_path,
            grid_sizeH=self.grid_sizeH, grid_sizeW=self.grid_sizeW,
        )
        fatezero_register.register_attention_control(self, edit_controller)

        sdimage_output = self.sd_ddim_pipeline(
            prompt=prompt, latents=latents, num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale, eta=eta, generator=generator,
            output_type=output_type, return_dict=return_dict,
            controller=edit_controller)

        fatezero_register.register_attention_control(self, self.empty_controller)

        mask_list = (edit_controller.latent_blend.mask_list
                     if edit_controller.latent_blend is not None
                     and hasattr(edit_controller.latent_blend, 'mask_list') else None)
        return {"sdimage_output": sdimage_output, "attention_output": None, "mask_list": mask_list}

    @torch.no_grad()
    def sd_ddim_pipeline(self, prompt, latents, num_inference_steps=50,
                         guidance_scale=7.5, eta=0.0, generator=None,
                         output_type="pil", return_dict=True, controller=None):
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        text_embeddings = self._encode_prompt(
            prompt, device, 1, do_classifier_free_guidance, None)

        self.scheduler = DDIMScheduler.from_config(self.scheduler_config)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        latents_dtype = latents.dtype
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(tqdm(timesteps, desc='reverse_diffusion')):
                want_alt = self._want_alt(num_inference_steps - 1 - i)
                if want_alt:
                    latents, indices = self._group_shuffle(latents, 0)
                else:
                    indices = list(range(self.video_length))
                b_indices = self.reassign_values(indices, self.grid_sizeH, self.grid_sizeW)

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                controller.begin_edit_step(i)
                latent_grids = latents.split(1, dim=2)
                latents_b = []
                for j, latent_grid_in_unet in enumerate(latent_model_input.split(1, dim=2)):
                    controller.begin_grid(j)
                    noise_pred = self.unet(
                        latent_grid_in_unet, t, encoder_hidden_states=text_embeddings,
                        indices=b_indices[j]).sample.to(dtype=latents_dtype)
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond)
                    latent = self.scheduler.step(
                        noise_pred, t, latent_grids[j], **extra_step_kwargs).prev_sample
                    latents_b.append(latent)
                latents = torch.cat(latents_b, dim=2)

                if want_alt:
                    latents, _ = self._group_shuffle(latents, 1)
                latents = controller.step_callback(latents).to(device=device, dtype=latents_dtype)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                torch.cuda.empty_cache()

        latents = self.decode_in_f_batches(latents)
        image = grid.restore_frames_from_grid(latents, self.grid_sizeH, self.grid_sizeW)
        if self._n_real is not None and self._n_real < image.shape[2]:
            image = image[:, :, :self._n_real]
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
