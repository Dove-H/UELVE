"""
Code of spatial blending module for latents and self-attention
"""

from typing import List
import os
import datetime
import grid
import numpy as np
import torchvision.utils as tvu
from einops import rearrange

import torch
import torch.nn.functional as F

import video_diffusion.prompt_attention.ptp_utils as ptp_utils

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class SpatialBlender:
    """
    Return a blending mask using the cross attention produced by both source during the inversion and target prompt during editing.
    Called in make_controller
    """
    def get_mask(self, maps, alpha, use_pool, h=None, w=None, x_t=None, step_in_store: int=None):
        """
        ([1, 40, 2, 16, 16, 77]) * ([1, 1, 1, 1, 1, 77]) -> [2, 1, 16, 16]
        mask have dimension of [clip_length, dim, res, res]
        """
        if h is None and w is None and x_t is not None:
            h, w = x_t.shape[-2:] # record the shape if only latent is provided
        k = 1
        #print("maps.shape=",maps.shape)
        if maps.dim() == 5: alpha = alpha[:, None, ...]
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = F.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = F.interpolate(maps, size=(h, w))
       
        mask = mask / mask.max(-2, keepdims=True)[0].max(-1, keepdims=True)[0]
        mask = mask.gt(self.th[1-int(use_pool)])
        if self.prompt_choose == 'both':
            assert mask.shape[0] == 2, "If using both source and target prompt"
            mask = mask[:1] + mask
        if self.save_path is not None:
            now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            save_path = f'{self.save_path}/{self.prompt_choose}/'
            if step_in_store is not None:
                save_path += f'step_in_store_{step_in_store:04d}'
            save_path +=f'/mask_{now}_{self.count:02d}.png'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if mask.shape[0] == 2:
                save_mask = mask[1:]
            else:
                save_mask = mask
            tvu.save_image(rearrange(save_mask.float(), "c p h w -> p c h w"), save_path,normalize=True)
            self.count +=1
        return mask

    def __call__(self, attention_store, step_in_store: int=None, target_h=None, target_w=None, x_t=None):
        """
            input has shape  (heads) clip res words
            one meens using target self-attention, zero is using source
            Previous implementation us all zeros
            mask should be repeat.

        Args:
            x_t (_type_): [1,4,8,64,64] # (prompt, channel, clip_length, res, res)
            attention_store (_type_): _description_

        Returns:
            _type_: _description_
        """
        if target_h is None and target_w is None and x_t is not None:
            target_h = x_t.shape[-2] # record the shape if only latent is provided
            target_w = x_t.shape[-1] # record the shape if only latent is provided
        # print("target_h.shape=",target_h)
        
        self.counter += 1
        if self.fixed_mask is not None:
            # External precomputed mask (GroundingDINO+FastSAM): bypass cross-attention entirely.
            return self._blend_with_fixed_mask(x_t)
        # if (self.counter > self.start_blend) and (self.counter < self.end_blend):


        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        # breakpoint()
        assert len(attention_store["down_cross"][0].shape) in [5, 4], \
            (f"the maps in attention_store must have shape [p c h (res_h res_w) w], or [c h (res_h res_w) w] \
            not {attention_store['down_cross'][0].shape} ")
        # maps = attention_store # [2,8,1024, 77] = [frames, head, (res, res), word_embedding]
        # a list of len(5), elements has shape [16, 256, 77]
        target_device = self.alpha_layers.device
        target_dtype  = self.alpha_layers.dtype
        # maps = [rearrange(  item, "p c h (res_h res_w) w -> p h c res_h res_w w ", 
                            # h=heads, res_h=res_h, res_w=res_h).to(target_device, dtype=target_dtype)
                            # for item in maps]
        rearranged_maps = []
        for item in maps:
            #print("map.shape=",map.shape)
            if len(item.shape) == 4: item = item[None, ...]
            ( p, c, heads, r, w)= item.shape
            res_h = int(np.sqrt(r))
            assert r == res_h* res_h, "the shape of attention map must be a squire"
            rearranged_item = rearrange(  item, "p c h (res_h res_w) w -> p h c res_h res_w w ", 
                            h=heads, res_h=res_h, res_w=res_h)
            rearranged_maps.append(rearranged_item.to(target_device, dtype=target_dtype))
        maps = torch.cat(rearranged_maps, dim=1)
        
        if self.prompt_choose == 'source':
            #print("source")
            # We support self-attention blending using only source prompt
            masked_alpah_layers = self.alpha_layers[0:1]
        else:
            masked_alpah_layers = self.alpha_layers
        mask = self.get_mask(maps, masked_alpah_layers, True, target_h, target_w, step_in_store=step_in_store)
        mask2 = self.get_mask(maps, masked_alpah_layers, True, 64, 64, step_in_store=step_in_store)
        #print("mask_shape_latent=",mask.shape)
        if self.substruct_layers is not None:
            maps_sub = self.get_mask(maps, self.substruct_layers, False)
            mask = mask * maps_sub
        mask = mask.float()
        mask2 = mask2.float()
        # mask is one: use generated information
        # mask is zero: use inverted information
        self.mask_list.append(mask[0][:, None, :, :].float().cpu().detach())
        if x_t is not None:
            if x_t.dim()==5: 
                mask2 = mask2[:, None, ...]
            # x_t [2,4,2,64,64]
            if (self.counter > self.start_blend) and (self.counter < self.end_blend):
                #print("x_t.shape=",x_t.shape)
                #print("mask2.shape=",mask2.shape)
                grid_sizeW=x_t.shape[-1]//64
                grid_sizeH=x_t.shape[-2]//64
                mask2=grid.combine_frames_in_grid(mask2,grid_sizeH,grid_sizeW)
                #print("mask222.shape=",mask2.shape)
                x_t = x_t[:1] + mask2 * (x_t - x_t[:1])
            return x_t
        else:
            return mask
       
    def _blend_with_fixed_mask(self, x_t):
        """External-mask latent blending (GroundingDINO+FastSAM via mask_utils.gen_masks).

        self.fixed_mask: [F, H, W] float in [0,1] (F = frames on this rank, e.g. 512x512).
        Area-resize to per-frame latent resolution, tile per-frame masks into the same 2x2 grid
        layout as x_t, then blend: mask=1 -> generated (target) latent, mask=0 -> inverted (source).
        Bypasses cross-attention, sidestepping the attention-mask shape issues in the grid layout.
        """
        if x_t is None:
            return None
        # per-frame latent resolution inferred from x_t's grid layout (64 for 512-input)
        grid_sizeH = max(x_t.shape[-2] // 64, 1)
        grid_sizeW = max(x_t.shape[-1] // 64, 1)
        hl, wl = x_t.shape[-2] // grid_sizeH, x_t.shape[-1] // grid_sizeW
        fm = self.fixed_mask.to(device=x_t.device, dtype=x_t.dtype)
        if fm.dim() == 2:
            fm = fm.unsqueeze(0)                                  # [F, H, W]
        m = F.interpolate(fm.unsqueeze(0), size=(hl, wl), mode="area")  # [1,F,H,W] -> [1,F,hl,wl]
        m = (m > 0.5).float()
        mask_pf = m.unsqueeze(1)                                  # [1,1,F,hl,wl] = (b,c,videolength,h,w)
        mask_grid = grid.combine_frames_in_grid(mask_pf, grid_sizeH, grid_sizeW)  # [1,1,F//(HW),gh,gw]
        if x_t.dim() == 4:
            mask_grid = mask_grid.squeeze(2)                      # [1,1,gh,gw] for 4D x_t
        # per-frame latent mask for downstream visualization: [F,1,hl,wl]
        self.mask_list.append(m.squeeze(0).unsqueeze(1).float().cpu().detach())
        if (self.counter > self.start_blend) and (self.counter < self.end_blend):
            x_t = x_t[:1] + mask_grid * (x_t - x_t[:1])
        return x_t

    def __init__(self, prompts: List[str], words: [List[List[str]]], substruct_words=None,
                 start_blend=0.2, end_blend=0.8,
                 th=(0.9, 0.9), tokenizer=None, NUM_DDIM_STEPS =None,
                 save_path = None, prompt_choose='source', fixed_mask=None):
        """
        Args:
            start_blend (float, optional): For latent blending, defaults to 0.2, for attention fusion better to be 0.0
            end_blend (float, optional): For latent blending, defaults to 0.8, for attention fusion better to be 1.0
            fixed_mask (torch.Tensor, optional): External precomputed per-frame mask [F,H,W] in [0,1]
                (e.g. from GroundingDINO+FastSAM). If given, latent blending uses it directly and
                bypasses the cross-attention-derived mask.
        """
        self.count = 0
        self.MAX_NUM_WORDS = 77
        self.NUM_DDIM_STEPS = NUM_DDIM_STEPS
        if save_path is not None:
            self.save_path = save_path
            os.makedirs(self.save_path, exist_ok=True)
        else:
            self.save_path = None
        assert prompt_choose in ['source', 'both'], "choose to generate the mask by only source prompt or both the source and target"
        self.prompt_choose = prompt_choose

        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, self.MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                # debug me
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        # self.alpha_layers.shape = torch.Size([2, 1, 1, 1, 1, 77]), 1 denotes the world to be replaced
        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts),  1, 1, 1, 1, self.MAX_NUM_WORDS)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None
        
        self.alpha_layers = alpha_layers.to(device)
        #print('the index mask of edited word in the prompt')
        #print(self.alpha_layers[0][..., 0:(len(prompts[0].split(" "))+2)])
        #print(self.alpha_layers[1][..., 0:(len(prompts[1].split(" "))+2)])
        
        self.start_blend = int(start_blend * self.NUM_DDIM_STEPS)
        self.end_blend = int(end_blend * self.NUM_DDIM_STEPS)
        self.counter = 0
        self.th=th
        self.mask_list = []
        self.fixed_mask = fixed_mask.to(device) if fixed_mask is not None else None

