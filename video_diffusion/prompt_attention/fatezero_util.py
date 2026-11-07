
from typing import Optional, Union, Tuple, List, Dict
import abc
import numpy as np
from einops import rearrange

import torch

import grid
import video_diffusion.prompt_attention.ptp_utils as ptp_utils
import video_diffusion.prompt_attention.seq_aligner as seq_aligner
from video_diffusion.prompt_attention.spatial_blend import SpatialBlender
from video_diffusion.prompt_attention.fatezero_store import AttentionControl, ReplayProvider, ATTN_MAX_TOKENS

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

_KEYS = ('down_cross', 'mid_cross', 'up_cross', 'down_self', 'mid_self', 'up_self')


class EmptyControl:
    def step_callback(self, x_t, *args, **kwargs):
        return x_t

    def between_steps(self):
        return

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn


def compute_need_steps(prompts, num_steps, cross_replace_steps, self_replace_steps,
                       blend_latents, tokenizer, latent_replace_steps=(0.2, 0.8)) -> dict:
    alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer)
    cross_edit = {t for t in range(num_steps) if float(alpha[t].max()) > 0}

    if type(self_replace_steps) is float:
        self_replace_steps = (0.0, self_replace_steps)
    self_edit = set(range(int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])))

    lat_edit = set()
    if blend_latents:
        if isinstance(latent_replace_steps, (int, float)):
            latent_replace_steps = (0.0, float(latent_replace_steps))
        s_win, e_win = int(latent_replace_steps[0] * num_steps), int(latent_replace_steps[1] * num_steps)
        lat_edit = {i for i in range(num_steps) if s_win < i + 1 < e_win}

    inv = lambda i: num_steps - 1 - i
    return {
        'cross': {inv(i) for i in (cross_edit | self_edit | lat_edit)},
        'self': {inv(i) for i in self_edit},
        'latents': {inv(i) for i in lat_edit},
    }


class AttentionControlEdit(AttentionControl, abc.ABC):

    def begin_edit_step(self, edit_step: int):
        self.cur_step = edit_step
        self.cur_att_layer = 0
        assert self.use_inversion_attention, "重构版 FateZero pipeline 仅支持 use_inversion_attention=True"
        self._store_step = self.num_steps - 1 - edit_step
        self._step_shards = []

    def begin_grid(self, grid_idx: int):
        self.cur_grid_idx = grid_idx
        self.attention_position_counter_dict = {k: 0 for k in _KEYS}
        if self.replay_provider is not None and self.replay_provider.has_shard(self._store_step):
            self.cur_shard = self.replay_provider.get_shard(self._store_step, grid_idx)
            self._step_shards.append(self.cur_shard)
        else:
            self.cur_shard = None

    def step_callback(self, x_t):
        if self.latent_blend is not None and self.replay_provider is not None \
                and self.replay_provider.has_latent(self._store_step):
            x_t_device, x_t_dtype = x_t.device, x_t.dtype
            inverted_latents = self.replay_provider.get_latent(self._store_step)
            inverted_latents = inverted_latents.to(device=x_t_device, dtype=x_t_dtype)

            inv_flat = {
                key: [entry for shard in self._step_shards for entry in shard.get(key, [])]
                for key in ("down_cross", "mid_cross", "up_cross")
            }
            blend_dict = {"down_cross": [], "mid_cross": [], "up_cross": []}
            for key in blend_dict:
                for i, attention in enumerate(inv_flat[key]):
                    concate_attention = torch.cat(
                        [attention[None, ...], self.attention_store[key][i][None, ...]], dim=0)
                    blend_dict[key].append(concate_attention)
            x_t = self.latent_blend(x_t=torch.cat([inverted_latents, x_t], dim=0),
                                    attention_store=blend_dict)[1:, ...]
        elif self.latent_blend is not None:
            self.latent_blend.counter += 1
        self.between_steps()
        return x_t

    def replace_self_attention(self, attn_base, att_replace, reshaped_mask=None):
        if att_replace.shape[-2] <= ATTN_MAX_TOKENS:
            target_device = att_replace.device
            target_dtype = att_replace.dtype
            attn_base = attn_base.to(target_device, dtype=target_dtype)
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            if reshaped_mask is not None:
                return reshaped_mask * att_replace + (1 - reshaped_mask) * attn_base
            return attn_base
        return att_replace

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def update_attention_position_dict(self, current_attention_key):
        self.attention_position_counter_dict[current_attention_key] += 1

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[-2] <= ATTN_MAX_TOKENS:
            if is_cross:
                self.step_store[key].append(attn.detach())

            current_pos = self.attention_position_counter_dict[key]
            if self.cur_shard is not None and current_pos < len(self.cur_shard[key]):
                attn_base = self.cur_shard[key][current_pos]
                attn_base = attn_base.to(device=attn.device, dtype=attn.dtype)
                self.update_attention_position_dict(key)

                if is_cross:
                    alpha_words = self.cross_replace_alpha[self.cur_step]
                    if float(alpha_words.max()) > 0:
                        clip_length = attn.shape[0] // self.batch_size
                        attn = attn.reshape(self.batch_size, clip_length, *attn.shape[1:])
                        attn_replace = attn[0:]
                        attn_replace_new = self.replace_cross_attention(attn_base, attn_replace) * alpha_words \
                            + (1 - alpha_words) * attn_replace
                        attn[0:] = attn_replace_new
                        attn = attn.reshape(self.batch_size * clip_length, *attn.shape[2:])
                elif self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]:
                    clip_length = attn.shape[0] // self.batch_size
                    attn = attn.reshape(self.batch_size, clip_length, *attn.shape[1:])
                    attn_replace = attn[0:]

                    reshaped_mask = None
                    if self.attention_blend is not None:
                        h = int(np.sqrt(attn_replace.shape[-1] // (self.grid_sizeH * self.grid_sizeW)))
                        w = h
                        mask = self.attention_blend(target_h=h, target_w=w,
                                                    attention_store=self.cur_shard,
                                                    step_in_store=self._store_step)
                        _g = self.grid_sizeH * self.grid_sizeW
                        mask = mask[:, :_g].unsqueeze(1)
                        if mask.numel() > 0:
                            mask = grid.combine_frames_in_grid(mask, self.grid_sizeH, self.grid_sizeW).squeeze(1)
                            reshaped_mask = rearrange(mask, "d c h w -> c d (h w)")[..., None]
                    attn[0:] = self.replace_self_attention(attn_base, attn_replace, reshaped_mask)
                    attn = attn.reshape(self.batch_size * clip_length, *attn.shape[2:])
            else:
                self.update_attention_position_dict(key)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()
        self.attention_position_counter_dict = {k: 0 for k in _KEYS}

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 latent_blend: Optional[SpatialBlender] = None, tokenizer=None,
                 replay_provider: ReplayProvider = None,
                 use_inversion_attention: bool = True,
                 attention_blend: SpatialBlender = None,
                 grid_sizeH: int = 2, grid_sizeW: int = 2):
        super().__init__()
        self.num_steps = num_steps
        self.batch_size = len(prompts) // 2
        assert self.batch_size == 1, 'Only support single video editing'
        self.replay_provider = replay_provider
        self.latent_blend = latent_blend
        self.attention_blend = attention_blend
        self.use_inversion_attention = use_inversion_attention
        self.grid_sizeH, self.grid_sizeW = grid_sizeH, grid_sizeW

        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(
            prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])

        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.attention_position_counter_dict = {k: 0 for k in _KEYS}
        self.cur_shard = None
        self._step_shards = []
        self._store_step = 0


class AttentionReplace(AttentionControlEdit):
    def replace_cross_attention(self, attn_base, att_replace):
        target_device = att_replace.device
        target_dtype = att_replace.dtype
        attn_base = attn_base.to(target_device, dtype=target_dtype)
        if attn_base.dim() == 3:
            return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
        elif attn_base.dim() == 4:
            return torch.einsum('thpw,bwn->bthpn', attn_base, self.mapper)

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 latent_blend: Optional[SpatialBlender] = None, tokenizer=None,
                 replay_provider: ReplayProvider = None, use_inversion_attention: bool = True,
                 attention_blend: SpatialBlender = None,
                 grid_sizeH: int = 2, grid_sizeW: int = 2):
        super().__init__(prompts, num_steps, cross_replace_steps, self_replace_steps,
                         latent_blend, tokenizer=tokenizer, replay_provider=replay_provider,
                         use_inversion_attention=use_inversion_attention,
                         attention_blend=attention_blend,
                         grid_sizeH=grid_sizeH, grid_sizeW=grid_sizeW)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)


class AttentionRefine(AttentionControlEdit):
    def replace_cross_attention(self, attn_base, att_replace):
        target_device = att_replace.device
        target_dtype = att_replace.dtype
        attn_base = attn_base.to(target_device, dtype=target_dtype)
        if attn_base.dim() == 3:
            attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        elif attn_base.dim() == 4:
            attn_base_replace = attn_base[:, :, :, self.mapper].permute(3, 0, 1, 2, 4)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 latent_blend: Optional[SpatialBlender] = None, tokenizer=None,
                 replay_provider: ReplayProvider = None, use_inversion_attention: bool = True,
                 attention_blend: SpatialBlender = None,
                 grid_sizeH: int = 2, grid_sizeW: int = 2):
        super().__init__(prompts, num_steps, cross_replace_steps, self_replace_steps,
                         latent_blend, tokenizer=tokenizer, replay_provider=replay_provider,
                         use_inversion_attention=use_inversion_attention,
                         attention_blend=attention_blend,
                         grid_sizeH=grid_sizeH, grid_sizeW=grid_sizeW)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 equalizer, latent_blend: Optional[SpatialBlender] = None,
                 controller: Optional[AttentionControlEdit] = None, tokenizer=None,
                 replay_provider: ReplayProvider = None, use_inversion_attention: bool = True,
                 attention_blend: SpatialBlender = None,
                 grid_sizeH: int = 2, grid_sizeW: int = 2):
        super().__init__(prompts, num_steps, cross_replace_steps, self_replace_steps,
                         latent_blend, tokenizer=tokenizer, replay_provider=replay_provider,
                         use_inversion_attention=use_inversion_attention,
                         attention_blend=attention_blend,
                         grid_sizeH=grid_sizeH, grid_sizeW=grid_sizeW)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


def get_equalizer(text: str, word_select, values, tokenizer=None):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)
    for word, val in zip(word_select, values):
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
    return equalizer


def make_controller(tokenizer, prompts: List[str], is_replace_controller: bool,
                    cross_replace_steps: Dict[str, float], self_replace_steps: float = 0.0,
                    blend_words=None, equilizer_params=None,
                    replay_provider: ReplayProvider = None, use_inversion_attention: bool = True,
                    blend_self_th: float = (0.3, 0.3), blend_latents_th: float = (0.3, 0.3),
                    latent_replace_steps=(0.2, 0.8),
                    NUM_DDIM_STEPS = None, blend_latents=False, blend_self_attention=False,
                    save_path=None, grid_sizeH: int = 2, grid_sizeW: int = 2) -> AttentionControlEdit:
    if (blend_words is None) or (blend_words == 'None'):
        latent_blend = None
        attention_blend = None
    else:
        if blend_latents:
            if isinstance(latent_replace_steps, (int, float)):
                latent_replace_steps = (0.0, float(latent_replace_steps))
            start_blend, end_blend = float(latent_replace_steps[0]), float(latent_replace_steps[1])
            latent_blend = SpatialBlender(prompts, blend_words,
                                          start_blend=start_blend, end_blend=end_blend,
                                          tokenizer=tokenizer, th=blend_latents_th,
                                          NUM_DDIM_STEPS=NUM_DDIM_STEPS,
                                          save_path=(save_path + '/latent_blend_mask') if save_path else None,
                                          prompt_choose='both')
            print(f'Blend latent mask with threshold {blend_latents_th}, window {tuple(latent_replace_steps)}')
        else:
            latent_blend = None
        if blend_self_attention:
            attention_blend = SpatialBlender(prompts, blend_words,
                                             start_blend=0.0, end_blend=2,
                                             tokenizer=tokenizer, th=blend_self_th,
                                             NUM_DDIM_STEPS=NUM_DDIM_STEPS,
                                             save_path=(save_path + '/attention_blend_mask') if save_path else None,
                                             prompt_choose='source')
            print(f'Blend self attention mask with threshold {blend_self_th}')
        else:
            attention_blend = None

    common = dict(latent_blend=latent_blend, tokenizer=tokenizer,
                  replay_provider=replay_provider,
                  use_inversion_attention=use_inversion_attention,
                  attention_blend=attention_blend,
                  grid_sizeH=grid_sizeH, grid_sizeW=grid_sizeW)
    if is_replace_controller:
        print('use replace controller')
        controller = AttentionReplace(prompts, NUM_DDIM_STEPS,
                                       cross_replace_steps=cross_replace_steps,
                                       self_replace_steps=self_replace_steps, **common)
    else:
        print('use refine controller')
        controller = AttentionRefine(prompts, NUM_DDIM_STEPS,
                                      cross_replace_steps=cross_replace_steps,
                                      self_replace_steps=self_replace_steps, **common)
    if equilizer_params is not None:
        eq = get_equalizer(prompts[1], equilizer_params["words"], equilizer_params["values"], tokenizer)
        controller = AttentionReweight(prompts, NUM_DDIM_STEPS,
                                       cross_replace_steps=cross_replace_steps,
                                       self_replace_steps=self_replace_steps,
                                       equalizer=eq, controller=controller, **common)
    return controller
