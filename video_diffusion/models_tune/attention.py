# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py
# https://github.com/showlab/Tune-A-Video/blob/main/tuneavideo/models/attention.py

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import CrossAttention, FeedForward, AdaLayerNorm

from einops import rearrange, repeat
import grid

@dataclass
class Transformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class Transformer3DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # Define input layers
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        # Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        if use_linear_projection:
            self.proj_out = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        return_dict: bool = True,
        grid_sizeH=2,
        grid_sizeW=2,
        indices=[]
    ):
        # 输入验证
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        
        # 获取视频长度并重组张量
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        
        # 如果需要，扩展encoder_hidden_states以匹配时间维度
        if encoder_hidden_states is not None:
            encoder_hidden_states = repeat(
                encoder_hidden_states, 
                'b n c -> (b f) n c', 
                f=video_length * grid_sizeH * grid_sizeW
            )
        
        # 保存残差连接
        batch, channel, height, width = hidden_states.shape
        residual = hidden_states
        
        # 归一化和投影
        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
            hidden_states = self.proj_in(hidden_states)
        
        # 通过所有transformer块
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                video_length=video_length,
                batch_bf=batch,
                channel=inner_dim,
                height_g=height,
                weight_g=width,
                grid_sizeH=grid_sizeH,
                grid_sizeW=grid_sizeW,
                indices=indices
            )
        
        # 输出处理
        if not self.use_linear_projection:
            hidden_states = (
                hidden_states.reshape(batch, height, width, inner_dim)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = (
                hidden_states.reshape(batch, height, width, inner_dim)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
        
        # 残差连接
        output = hidden_states + residual
        
        # 重组回原始形状
        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)
        
        # 根据return_dict决定返回格式
        if not return_dict:
            return (output,)
        
        return Transformer3DModelOutput(sample=output)









class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = num_embeds_ada_norm is not None

        #self-attention
        self.attn1 = self_Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )  # is a self-attention
        self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

        # Cross-Attn
        if cross_attention_dim is not None:
            self.attn2 = CrossAttention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        else:
            self.attn2 = None

        if cross_attention_dim is not None:
            self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        else:
            self.norm2 = None
        
        # Feed-forward
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.norm3 = nn.LayerNorm(dim)

        # Temp-Attn
        self.attn_temp = CrossAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )
        nn.init.zeros_(self.attn_temp.to_out[0].weight.data)
        self.norm_temp = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

    def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool):
        if not is_xformers_available():
            print("Here is how to install it")
            raise ModuleNotFoundError(
                "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                " xformers",
                name="xformers",
            )
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only"
                " available for GPU "
            )
        else:
            try:
                # Make sure we can run the memory efficient attention
                _ = xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e
            self.attn1._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            if self.attn2 is not None:
                self.attn2._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            #self.attn_temp._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, attention_mask=None, video_length=None,batch_bf=None,channel=None,height_g=None,weight_g=None,grid_sizeH=2,grid_sizeW=2,indices=[]):
        #print("indices",indices)
        # SparseCausal-Attention
        #print("_______________latents.shape=",hidden_states.shape)
        norm_hidden_states = (
            self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states)
        )

        if self.only_cross_attention:
            hidden_states = (
                self.attn1(norm_hidden_states, encoder_hidden_states, attention_mask=attention_mask) + hidden_states
            )
        else:
            hidden_states = self.attn1(norm_hidden_states, attention_mask=attention_mask) + hidden_states
        
        
        
        
        # 拆网格   
        #print("0",hidden_states.shape)
        #print("hidden_states999.shape=",hidden_states.shape)
        hidden_states = hidden_states.reshape(batch_bf, height_g, weight_g, channel).permute(0, 3, 1, 2).contiguous()
        #print("1",hidden_states.shape)
        hidden_states = rearrange(hidden_states, "(b f) c h w -> b c f h w", f=video_length) 
        #print("video_length=",video_length)
        #print("2",hidden_states.shape)
        #print("####",grid_size)
        
        hidden_states = grid.restore_frames_from_grid(hidden_states,grid_sizeH,grid_sizeW)
        if indices !=[]:
            #print("indices8090=",indices)
            #print("hidden_states8090.shape=",hidden_states.shape)
            inverse_indices = torch.argsort(torch.tensor(indices))
            hidden_states = hidden_states[:, :, inverse_indices, :, :]
        
        #print("3",hidden_states.shape)
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w", f=video_length*grid_sizeH*grid_sizeW)
        #print("4",hidden_states.shape)
        new_batch=batch_bf*grid_sizeH*grid_sizeW
        new_height=int(height_g/grid_sizeH)
        new_weight=int(weight_g/grid_sizeW)
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(new_batch, new_height*new_weight, channel)
        #print("5",hidden_states.shape)  



        if self.attn2 is not None:
            # Cross-Attention
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )
            # print("*****context.shape=",encoder_hidden_states.shape)
            # print("*****latent_cur.shape=",norm_hidden_states.shape)
            hidden_states = (
                self.attn2(
                    norm_hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
                )
                + hidden_states
            )


        
        
        # Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
        #return hidden_states


        
        
        # temp-Attention
        d = hidden_states.shape[1]
        hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
        norm_hidden_states = (
            self.norm_temp(hidden_states, timestep) if self.use_ada_layer_norm else self.norm_temp(hidden_states)
        )
        hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
        hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)
        
        #合网格
        new_batch=batch_bf*grid_sizeH*grid_sizeW
        new_height=int(height_g/grid_sizeH)
        new_weight=int(weight_g/grid_sizeW)
        #print("0",hidden_states.shape)
        hidden_states = hidden_states.reshape(new_batch, new_height, new_weight, channel).permute(0, 3, 1, 2).contiguous()
        #print("1",hidden_states.shape)
        hidden_states = rearrange(hidden_states, "(b f) c h w -> b c f h w", f=video_length*grid_sizeH*grid_sizeW) 
        #print("2",hidden_states.shape)
        
        if indices !=[]:
            hidden_states = hidden_states[:, :, indices, :, :]
        hidden_states = grid.combine_frames_in_grid(hidden_states,grid_sizeH,grid_sizeW)
        
        #print("3",hidden_states.shape)
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w", f=video_length)
        #print("4",hidden_states.shape)
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_bf, height_g*weight_g, channel)
        #print("5",hidden_states.shape)  

        return hidden_states


# class FrameAttention(CrossAttention):
#     def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None):
#         batch_size, sequence_length, _ = hidden_states.shape

#         encoder_hidden_states = encoder_hidden_states

#         if self.group_norm is not None:
#             hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

#         query = self.to_q(hidden_states)
#         dim = query.shape[-1]
#         query = self.reshape_heads_to_batch_dim(query)

#         if self.added_kv_proj_dim is not None:
#             raise NotImplementedError

#         encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
#         key = self.to_k(encoder_hidden_states)
#         value = self.to_v(encoder_hidden_states)

#         former_frame_index = torch.arange(video_length) - 1
#         former_frame_index[0] = 0

#         key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
#         key = key[:, [0] * video_length]
#         key = rearrange(key, "b f d c -> (b f) d c")

#         value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
#         value = value[:, [0] * video_length]
#         value = rearrange(value, "b f d c -> (b f) d c")

#         key = self.reshape_heads_to_batch_dim(key)
#         value = self.reshape_heads_to_batch_dim(value)

#         if attention_mask is not None:
#             if attention_mask.shape[-1] != query.shape[1]:
#                 target_length = query.shape[1]
#                 attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
#                 attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

#         # attention, what we cannot get enough of
#         if self._use_memory_efficient_attention_xformers:
#             hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
#             # Some versions of xformers return output in fp32, cast it back to the dtype of the input
#             hidden_states = hidden_states.to(query.dtype)
#         else:
#             if self._slice_size is None or query.shape[0] // self._slice_size == 1:
#                 hidden_states = self._attention(query, key, value, attention_mask)
#             else:
#                 hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

#         # linear proj
#         hidden_states = self.to_out[0](hidden_states)

#         # dropout
#         hidden_states = self.to_out[1](hidden_states)
#         return hidden_states

class self_Attention(CrossAttention):
    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape

        encoder_hidden_states = encoder_hidden_states

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        dim = query.shape[-1]
        query = self.reshape_heads_to_batch_dim(query)

        if self.added_kv_proj_dim is not None:
            key = self.to_k(hidden_states)
            value = self.to_v(hidden_states)
            encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)

            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)
            encoder_hidden_states_key_proj = self.reshape_heads_to_batch_dim(encoder_hidden_states_key_proj)
            encoder_hidden_states_value_proj = self.reshape_heads_to_batch_dim(encoder_hidden_states_value_proj)

            key = torch.concat([encoder_hidden_states_key_proj, key], dim=1)
            value = torch.concat([encoder_hidden_states_value_proj, value], dim=1)
        else:
            encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        # attention, what we cannot get enough of
        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, attention_mask)
            else:
                hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states
    