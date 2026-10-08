from typing import Optional, Union, Tuple, List, Dict
import abc
import numpy as np
import copy
import grid
import os
from einops import rearrange
import torch
import time
import traceback
import torch.distributed as dist
import torch.nn.functional as F
import video_diffusion.prompt_attention.ptp_utils as ptp_utils
import video_diffusion.prompt_attention.seq_aligner as seq_aligner
from video_diffusion.prompt_attention.spatial_blend import SpatialBlender
# from video_diffusion.prompt_attention.visualization import show_cross_attention, show_self_attention_comp
from video_diffusion.prompt_attention.attention_store import AttentionStore, AttentionControl
from video_diffusion.prompt_attention.attention_register import register_attention_control

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')        
class EmptyControl:
    def step_callback(self, x_t):
        return x_t
    def between_steps(self):
        return
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn

class AttentionControlEdit(AttentionStore, abc.ABC): 
    def step_callback(self, x_t,step_in_store,latents_path):
        x_t = super().step_callback(x_t,flag=1,attention_path=self.attention_path)
        x_t_device = x_t.device
        x_t_dtype = x_t.dtype

        if self.latent_blend is not None:
            file_latents_path = os.path.join(latents_path, f"{step_in_store}.pt")  
            # 加载 latent 数据
            time_start=time.time()
            inverted_latents = torch.load(file_latents_path,weights_only=True)
            time_end=time.time()
            #print("latents qu","t=",step_in_store,"time=",time_end-time_start)
            inverted_latents = inverted_latents.to(device =x_t_device, dtype=x_t_dtype)
            blend_dict = self.get_empty_cross_store()
            # each element in blend_dict have (prompt head) clip_length (res res) words, 
            # to better align with  (b c f h w) #恒为20
            # print("attention_store_all_step=",len(self.additional_attention_store.attention_store_all_step))
            #print("attention_store_all_step=",len(self.attention_store_all_step)) 0到19
            
            #step_in_store_atten_dict = self.additional_attention_store.attention_store_all_step[step_in_store]
            # Load attention data
            # file_attention_path = os.path.join(self.attention_path, f"{step_in_store}.pt")
            # attention_data = torch.load(file_attention_path)
            # step_in_store_atten_dict = attention_data['step_store']
            step_in_store_atten_dict=self.step_in_store_atten_dict
            if isinstance(step_in_store_atten_dict, str): 
                step_in_store_atten_dict = torch.load(step_in_store_atten_dict,weights_only=True)
            for key in blend_dict.keys():
                place_in_unet_cross_atten_list = step_in_store_atten_dict[key]
                for i, attention in enumerate(place_in_unet_cross_atten_list):
                    concate_attention = torch.cat([attention[None, ...], self.attention_store[key][i][None, ...]], dim=0)
                    blend_dict[key].append(copy.deepcopy(concate_attention))
            x_t = self.latent_blend(x_t = copy.deepcopy(torch.cat([inverted_latents, x_t], dim=0)), attention_store = copy.deepcopy(blend_dict))
            if step_in_store>0:
                file_attention_path = os.path.join(self.attention_path, f"{step_in_store-1}.pt")
                time_start1=time.time()
                attention_data = torch.load(file_attention_path,weights_only=True)
                time_end1=time.time()
                #print("attentions qu","t=",step_in_store-1,"time=",time_end1-time_start1)
                self.step_in_store_atten_dict = attention_data['step_store']
            return x_t[1:, ...]
        else:
            #为下一个时间步准备attention feature map
            if step_in_store>0:
                file_attention_path = os.path.join(self.attention_path, f"{step_in_store-1}.pt")
                
                attention_data = torch.load(file_attention_path,weights_only=True)
                
                
                self.step_in_store_atten_dict = attention_data['step_store']            
            return x_t
        
    def replace_self_attention(self, attn_base, att_replace, reshaped_mask=None):
        if att_replace.shape[-2] <= 32 ** 2:
            target_device = att_replace.device
            target_dtype  = att_replace.dtype
            attn_base = attn_base.to(target_device, dtype=target_dtype)
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            if reshaped_mask is not None:
                return_attention = reshaped_mask*att_replace + (1-reshaped_mask)*attn_base
                return return_attention
            else:
                return attn_base
        else:
            return att_replace
    

  

    def replace_temp_attention(self, attn_base, att_replace, reshaped_mask=None):

        if att_replace.shape[1] <= 32 ** 2:
            target_device = att_replace.device
            target_dtype  = att_replace.dtype
            attn_base = attn_base.to(target_device, dtype=target_dtype)
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            h = torch.sqrt(torch.tensor(float(attn_base.shape[1]))).int()
            frames_in_batch=attn_base.shape[-1]
            attn_base=attn_base.view(1,h,h,8,frames_in_batch,frames_in_batch)

            if reshaped_mask is not None:
                att_replace=att_replace.view(1,h,h,8,frames_in_batch,frames_in_batch)
                reshaped_mask=reshaped_mask.unsqueeze(0)
                return_attention = reshaped_mask*att_replace + (1-reshaped_mask)*attn_base
                return_attention=return_attention.view(1,h*h,8,frames_in_batch,frames_in_batch)
                return return_attention
            else:
                # 无 temp_blend 掩码时整层替换为 inversion 保存的 temporal attention（FateZero 时序锚定）
                attn_base=attn_base.unsqueeze(0)
                attn_base=attn_base.view(1,h*h,8,frames_in_batch,frames_in_batch)
                return attn_base
        else:
            return att_replace


    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def update_attention_position_dict(self, current_attention_key):
        self.attention_position_counter_dict[current_attention_key] +=1


    def forward(self, attn, is_cross: int, place_in_unet: str,grid_sizeH=2,grid_sizeW=2):
        
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet,grid_sizeH,grid_sizeW)
        
        if attn.shape[-2] <= 32 ** 2:
            key = f"{place_in_unet}_{'self' if is_cross == 1 else 'cross' if is_cross == 2 else 'temp'}"
            current_pos = self.attention_position_counter_dict[key] #恒为20
            # print("attention_store_all_step=",len(self.additional_attention_store.attention_store_all_step))
            # print("self.cur_step=",self.cur_step) 0到19
            if self.use_inversion_attention:
                step_in_store = self.num_steps- self.cur_step -1
            else:
                step_in_store = self.cur_step
            
            #print("step_in_store=",step_in_store) 
            #step_in_store_atten_dict = self.additional_attention_store.attention_store_all_step[step_in_store]
            # file_attention_path = os.path.join(self.attention_path, f"{step_in_store}.pt")
            # attention_data = torch.load(file_attention_path)
            # step_in_store_atten_dict = attention_data['step_store']
            step_in_store_atten_dict=self.step_in_store_atten_dict
            if isinstance(step_in_store_atten_dict, str): 
                step_in_store_atten_dict = torch.load(step_in_store_atten_dict)
            
            # Note that attn is append to step_store, 
            # if attn is get through clean -> noisy, we should inverse it
            attn_base = step_in_store_atten_dict[key][current_pos]          
           
            self.update_attention_position_dict(key)
            # save in format of [temp, head, resolution, text_embedding]
            if is_cross==2 or is_cross==3 or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
                clip_length = attn.shape[0] // (self.batch_size)
               
                attn = attn.reshape(self.batch_size, clip_length, *attn.shape[1:])
                # Replace att_replace with attn_base
                attn_base, attn_repalce = attn_base, attn[0:]
                if is_cross==2:
                    alpha_words = self.cross_replace_alpha[self.cur_step]
                    attn_repalce_new =  self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce 
                    attn[0:] = attn_repalce_new # b t h p n = [1, 1, 8, 1024, 77]
                if is_cross==1 and attn.shape[-2] <= 32 ** 2:
                    if self.attention_blend is not None and attn_repalce.shape[-2] <= 32 ** 2:
                        h = int(np.sqrt(attn_repalce.shape[-1]//(grid_sizeH*grid_sizeW)))
                        w=h
                        mask = self.attention_blend(target_h = h, target_w =w, attention_store= step_in_store_atten_dict, step_in_store=step_in_store)
                        # The cross-attention mask from attention_blend already carries exactly one grid's
                        # worth of frames (mask.shape[1] == grid_sizeH*grid_sizeW). It is grid-agnostic, so
                        # take the available frames for the current grid. (Previously this sliced
                        # mask[:, _j*_g:(_j+1)*_g] assuming 2*_g frames -> empty for cur_grid_idx=1 -> crash.)
                        _g = grid_sizeH * grid_sizeW
                        mask = mask[:, :_g]
                        mask=mask.unsqueeze(1)
                        if mask.numel() == 0:  # safety: degenerate mask -> skip blend this step
                            reshaped_mask = None
                        else:
                            mask=grid.combine_frames_in_grid(mask,grid_sizeH,grid_sizeW)
                            mask=mask.squeeze(1)
                            reshaped_mask = rearrange(mask, "d c h w -> c d (h w)")[..., None]
                    else:
                        reshaped_mask = None
                    attn[0:] = self.replace_self_attention(attn_base, attn_repalce, reshaped_mask)

                if is_cross==3 and attn_base.shape[0]<=1024:
                    if self.temp_blend is not None :
                            h = int(np.sqrt(attn_repalce.shape[1]))
                            w = int(np.sqrt(attn_repalce.shape[2]))
                            mask = self.temp_blend(target_h = h, target_w =w, attention_store= step_in_store_atten_dict, step_in_store=step_in_store)

                            mask = torch.max(mask, dim=1, keepdim=True)[0]
                            reshaped_mask_temp = mask.view(h, w, 1, 1, 1)
                    else:
                        reshaped_mask_temp=None
                    if not getattr(self, '_temp_replay_logged', False):
                        print(f"[temp-attn] temporal attention replay engaged: attn_base {tuple(attn_base.shape)} @ step {self.cur_step}")
                        self._temp_replay_logged = True
                    attn[0:] = self.replace_temp_attention(attn_base, attn_repalce, reshaped_mask_temp)

                attn = attn.reshape(self.batch_size * clip_length, *attn.shape[2:])
                # save in format of [temp, head, resolution, text_embedding]
                
        return attn
    def between_steps(self):
        
        super().between_steps()
        self.step_store = self.get_empty_store()
        
        self.attention_position_counter_dict = {
            'down_cross': 0,
            'mid_cross': 0,
            'up_cross': 0,
            'down_self': 0,
            'mid_self': 0,
            'up_self': 0,
            'down_temp': 0,
            'mid_temp': 0,
            'up_temp': 0,
        }
        return 
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 latent_blend: Optional[SpatialBlender], tokenizer=None, 
                 additional_attention_store: AttentionStore =None,
                 use_inversion_attention: bool=False,
                 attention_blend: SpatialBlender= None,
                 temp_blend: SpatialBlender= None,
                 save_self_attention: bool=True,
                 disk_store=False,
                 attention_path: str=None  # 新增参数
                 ):
        super(AttentionControlEdit, self).__init__(
            save_self_attention=save_self_attention,
            disk_store=disk_store,stage=1)
        self.attention_path = attention_path  # 存储
        self.temp_blend = temp_blend

        self.num_steps = num_steps
        self.additional_attention_store = additional_attention_store
        self.batch_size = len(prompts)
        self.attention_blend = attention_blend
        if self.additional_attention_store is not None:
            # the attention_store is provided outside, only pass in one promp
            self.batch_size = len(prompts) //2
            assert self.batch_size==1, 'Only support single video editing with additional attention_store'

        #加载第一个时间步的数据
        first_step_path = os.path.join(self.attention_path,  f"{self.num_steps-1}.pt")
        attention_data = torch.load(first_step_path,weights_only=True)
        self.step_in_store_atten_dict = attention_data['step_store']

    
            
            
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_steps=num_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        #self.cur_step=0
        self.latent_blend = latent_blend
        # We need to know the current position in attention
        self.prev_attention_key_name = 0
        self.use_inversion_attention = use_inversion_attention
        self.attention_position_counter_dict = {
            'down_cross': 0,
            'mid_cross': 0,
            'up_cross': 0,
            'down_self': 0,
            'mid_self': 0,
            'up_self': 0,
            'down_temp': 0,
            'mid_temp': 0,
            'up_temp': 0,
        }

class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        # torch.Size([8, 4096, 77]), torch.Size([1, 77, 77]) -> [1, 8, 4096, 77]
        # Can be extend to temp, use temp as batch size
        target_device = att_replace.device
        target_dtype  = att_replace.dtype
        attn_base = attn_base.to(target_device, dtype=target_dtype)
       
        if attn_base.dim()==3:
            return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
        elif attn_base.dim()==4:
            return torch.einsum('thpw,bwn->bthpn', attn_base, self.mapper)
      
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 latent_blend: Optional[SpatialBlender] = None, tokenizer=None,
                 additional_attention_store=None,
                 use_inversion_attention = False,
                 attention_blend: SpatialBlender=None,
                 temp_blend: SpatialBlender = None,
                 save_self_attention: bool = True,
                 disk_store=False,
                 attention_path: str=None  ):
        super(AttentionReplace, self).__init__(
            prompts, num_steps, cross_replace_steps, self_replace_steps, latent_blend, tokenizer=tokenizer,
            additional_attention_store=additional_attention_store, use_inversion_attention = use_inversion_attention,
            attention_blend=attention_blend,temp_blend=temp_blend,
            save_self_attention = save_self_attention,
            disk_store=disk_store,
            attention_path=attention_path
            )
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)

class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        
        target_device = att_replace.device
        target_dtype  = att_replace.dtype
        attn_base = attn_base.to(target_device, dtype=target_dtype)
        if attn_base.dim()==3:
            attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        elif attn_base.dim()==4:
            attn_base_replace = attn_base[:, :, :, self.mapper].permute(3, 0, 1, 2, 4)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 latent_blend: Optional[SpatialBlender] = None, tokenizer=None,
                 additional_attention_store=None,
                 use_inversion_attention = False,
                 attention_blend: SpatialBlender=None,
                 temp_blend: SpatialBlender = None,
                 save_self_attention : bool=True,
                 disk_store = False,
                 attention_path: str=None  # 新增参数
                 ):
        super(AttentionRefine, self).__init__(
            prompts, num_steps, cross_replace_steps, self_replace_steps, latent_blend, tokenizer=tokenizer,
            additional_attention_store=additional_attention_store, use_inversion_attention = use_inversion_attention,
            attention_blend=attention_blend,temp_blend=temp_blend,
            save_self_attention = save_self_attention,
            disk_store = disk_store,
            attention_path=attention_path  # 传递参数
            )
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):
    """First replace the weight, than increase the attention at a area

    Args:
        AttentionControlEdit (_type_): _description_
    """

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                latent_blend: Optional[SpatialBlender] = None, controller: Optional[AttentionControlEdit] = None, tokenizer=None,
                additional_attention_store=None,
                use_inversion_attention = False,
                attention_blend: SpatialBlender=None,
                temp_blend: SpatialBlender = None,
                save_self_attention:bool = True,
                disk_store = False,
                attention_path: str=None  
                ):
        super(AttentionReweight, self).__init__(
            prompts, num_steps, cross_replace_steps, self_replace_steps, latent_blend, tokenizer=tokenizer,
            additional_attention_store=additional_attention_store,
            use_inversion_attention = use_inversion_attention,
            attention_blend=attention_blend,temp_blend=temp_blend,
            save_self_attention=save_self_attention,
            disk_store = disk_store,
            attention_path=attention_path
            )
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller

def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]], tokenizer=None):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)
    
    for word, val in zip(word_select, values):
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
    return equalizer



def make_controller(tokenizer, prompts: List[str], is_replace_controller: bool,
                    cross_replace_steps: Dict[str, float], self_replace_steps: float=0.0, 
                    blend_words=None, equilizer_params=None, 
                    additional_attention_store=None, use_inversion_attention = False, 
                    blend_self_th: float=(0.3, 0.3),
                    blend_temp_th: float=(0.3, 0.3),
                    blend_latents_th: float=(0.3, 0.3),
                    NUM_DDIM_STEPS=None,
                    blend_latents = False,
                    blend_self_attention=False,
                    blend_temp_attention=False,
                    save_path = None,
                    save_self_attention = True,
                    disk_store = False,
                    attention_path: str=None  # 新增参数
                    ) -> AttentionControlEdit:
    if (blend_words is None) or (blend_words == 'None'):
        latent_blend = None
        attention_blend =None
    else:
        if blend_latents:
            latent_blend = SpatialBlender( prompts, blend_words,
                                       start_blend = 0.2, end_blend=0.8,
                                       tokenizer=tokenizer, th=blend_latents_th, NUM_DDIM_STEPS=NUM_DDIM_STEPS,
                            save_path=save_path+f'/latent_blend_mask',
                            prompt_choose='both')
            print(f'Blend latent mask with threshold {blend_latents_th}')
        else:
            latent_blend = None
        if blend_self_attention:
            attention_blend = SpatialBlender( prompts, blend_words,
                                                    start_blend = 0.0, end_blend=2,
                                                  tokenizer=tokenizer, th=blend_self_th, NUM_DDIM_STEPS=NUM_DDIM_STEPS,
                           save_path=save_path+f'/attention_blend_mask',
                           prompt_choose='source')
            print(f'Blend self attention mask with threshold {blend_self_th}')
        else:
            attention_blend = None
        
        if blend_temp_attention:
            temp_blend = SpatialBlender( prompts, blend_words,
                                                    start_blend = 0.0, end_blend=2,
                                                  tokenizer=tokenizer, th=blend_temp_th, NUM_DDIM_STEPS=NUM_DDIM_STEPS,
                           save_path=save_path+f'/temp_blend_mask',
                           prompt_choose='source')
            print(f'Blend temp attention mask with threshold {blend_temp_th}')
        else:
            temp_blend = None


    if is_replace_controller:
        print('use replace controller')
        controller = AttentionReplace(prompts, NUM_DDIM_STEPS, 
                                      cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, 
                                      latent_blend=latent_blend, tokenizer=tokenizer,
                                      additional_attention_store=additional_attention_store,
                                      use_inversion_attention = use_inversion_attention,
                                      attention_blend=attention_blend,temp_blend=temp_blend,
                                      save_self_attention = save_self_attention,
                                      disk_store=disk_store,
                                      attention_path=attention_path
                                      )
    else:
        print('use refine controller')
        controller = AttentionRefine(prompts, NUM_DDIM_STEPS,
                                     cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps,
                                     latent_blend=latent_blend, tokenizer=tokenizer,
                                     additional_attention_store=additional_attention_store,
                                     use_inversion_attention = use_inversion_attention,
                                     attention_blend=attention_blend,temp_blend=temp_blend,
                                     save_self_attention = save_self_attention,
                                     disk_store=disk_store,
                                     attention_path=attention_path
                                     )
    if equilizer_params is not None:
        eq = get_equalizer(prompts[1], equilizer_params["words"], equilizer_params["values"], tokenizer=tokenizer)
        controller = AttentionReweight(prompts, NUM_DDIM_STEPS, 
                                       cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, 
                                       equalizer=eq, latent_blend=latent_blend, controller=controller, 
                                        tokenizer=tokenizer,
                                        additional_attention_store=additional_attention_store,
                                        use_inversion_attention = use_inversion_attention,
                                        attention_blend=attention_blend,temp_blend=temp_blend,
                                        save_self_attention = save_self_attention,
                                        disk_store=disk_store,
                                        attention_path=attention_path
                                       )
    return controller

