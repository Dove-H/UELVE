"""
Code of attention storer AttentionStore, which is a base class for attention editor in attention_util.py

"""

import abc
import os
import copy
import torch
import time
from video_diffusion.common.util import get_time_string

class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        self.cur_att_layer = 0
        self.cur_step += 1
        self.between_steps()
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        """I guess the diffusion of google has some unconditional attention layer
        No unconditional attention layer in Stable diffusion

        Returns:
            _type_: _description_
        """
        # return self.num_att_layers if config_dict['LOW_RESOURCE'] else 0
        return 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: int, place_in_unet: str,grid_sizeH,grid_sizeW):
        raise NotImplementedError

    def __call__(self, attn, is_cross: int, place_in_unet: str,grid_sizeH=2,grid_sizeW=2):
        
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if self.LOW_RESOURCE:
                # For inversion without null text file 
                attn = self.forward(attn, is_cross, place_in_unet,grid_sizeH,grid_sizeW)
            else:
                # For classifier-free guidance scale!=1
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet,grid_sizeH,grid_sizeW)
        self.cur_att_layer += 1

        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self, 
                 ):
        self.LOW_RESOURCE = False # assume the edit have cfg
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class AttentionStore(AttentionControl):
    def step_callback(self, x_t,flag,attention_path):
        self.flag=flag
        self.attention_path=attention_path
        x_t = super().step_callback(x_t)
        
        # if flag==0:
        #    self.latents_store.append(x_t.cpu().detach())
        return x_t
    
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": [],}

    @staticmethod
    def get_empty_cross_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                }

    def forward(self, attn, is_cross: int, place_in_unet: str,grid_sizeH=2,grid_sizeW=2):
        key = f"{place_in_unet}_{'self' if is_cross == 1 else 'cross' }"
        if attn.shape[-2] <= 32 ** 2: 
            if self.stage==0:
                if is_cross==2 or self.save_self_attention:
                    if attn.shape[-2] == 32**2:
                        append_tensor = attn.cpu().detach()
                    else:
                        append_tensor = attn
                    self.step_store[key].append(copy.deepcopy(append_tensor))
            elif self.stage==1:
                if is_cross==2:  # avoid memory overhead
                        if attn.shape[-2] == 32**2:
                            append_tensor = attn.cpu().detach()
                        else:
                            append_tensor = attn
                        self.step_store[key].append(copy.deepcopy(append_tensor))
        return attn


    def between_steps(self):
        if len(self.attention_store) == 0:
                self.attention_store = self.step_store
        if self.flag==1: 
            for key in self.attention_store:
            
                for i in range(len(self.attention_store[key])):
                
                    self.attention_store[key][i] += self.step_store[key][i]
        elif self.flag==0:
            # Save to disk instead of memory
            os.makedirs(self.attention_path, exist_ok=True)
            time_start=time.time()
            torch.save({
                'step_store': copy.deepcopy(self.step_store),
                'cur_step': self.cur_step-1
            }, os.path.join(self.attention_path, f'{self.cur_step-1}.pt'))
            time_end=time.time()
            #print("attentions cun","t=",self.cur_step-1 ,"time=",time_end-time_start)
            #self.attention_store_all_step.append(copy.deepcopy(self.step_store))
            
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        "divide the attention map value in attention store by denoising steps"
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store_all_step = []
        self.attention_store = {}

    def __init__(self, save_self_attention:bool=True, disk_store=False,stage:int=-1):
        super(AttentionStore, self).__init__()
        #print("stage:",stage)
        self.stage=stage
        self.disk_store = disk_store
        if self.disk_store:
            time_string = get_time_string()
            path = f'./trash/attention_cache_{time_string}'
            os.makedirs(path, exist_ok=True)
            self.store_dir = path
        else:
            self.store_dir =None
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.save_self_attention = save_self_attention
        self.latents_store = []
        self.attention_store_all_step = []
        self.flag=0
        self.attention_path=None
