"""
FateZero 模式（重构版）controller 注册。

与原版 FateZero 的 register_attention_control 语义对齐：
- attn1 (SparseCausalAttention) → is_cross=False（'{place}_self' 键）
- attn2 (CrossAttention)         → is_cross=True（'{place}_cross' 键）
- attn_temp 完全不注册（原版 register_recr 即跳过；其 to_out 零初始化在
  zero-shot 下为恒等映射，tune 后自由前向，两种模式都不需要重放）

与旧版 ELVE register 的差别：不再整段复制 forward（旧文件自带的 FIXME），
而是给每个 attention 模块实例绑一个只包含"计算 attention probs → 交给
controller → bmm 输出"的紧凑闭包。超过 32² token 的大层走 xformers，
与 ≤1024 token 的记录/重放路径在反演、编辑两侧对称跳过。
"""

import torch
from einops import rearrange


def register_attention_control(model, controller):
    """把 controller 挂到 UNet 的 attn1 / attn2 上（attn_temp 跳过）。"""

    def make_controller_attention(module, is_cross: bool, place_in_unet: str):
        """紧凑的受控 attention：softmax 概率交给 controller 再与 V 相乘。"""
        to_out = module.to_out[0] if isinstance(module.to_out, torch.nn.ModuleList) else module.to_out

        def _controller_attention(query, key, value, attention_mask=None):
            if module.upcast_attention:
                query = query.float()
                key = key.float()
            attention_scores = torch.baddbmm(
                torch.empty(query.shape[0], query.shape[1], key.shape[1],
                            dtype=query.dtype, device=query.device),
                query, key.transpose(-1, -2), beta=0, alpha=module.scale,
            )
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            if module.upcast_softmax:
                attention_scores = attention_scores.float()
            attention_probs = attention_scores.softmax(dim=-1).to(value.dtype)

            # CORE: 记录（反演）/ 编辑（编辑期），[b, heads, s, t]
            attention_probs = controller(
                rearrange(attention_probs, "(b h) s t -> b h s t", h=module.heads),
                is_cross, place_in_unet)
            attention_probs = rearrange(attention_probs, "b h s t -> (b h) s t", h=module.heads)

            hidden_states = torch.bmm(attention_probs, value)
            return module.reshape_batch_dim_to_heads(hidden_states)

        return _controller_attention

    def patch_forward(module, is_cross: bool, place_in_unet: str):
        """返回替代 module.forward 的函数（保留原投影逻辑，替换 attention 计算）。"""
        _controller_attention = make_controller_attention(module, is_cross, place_in_unet)
        to_out = module.to_out[0] if isinstance(module.to_out, torch.nn.ModuleList) else module.to_out

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None,
                    clip_length: int = None, **_ignored):
            # SparseCausalAttention_index 等额外 kwargs 在原 forward 中本就无效果（gather 已注释），忽略
            if module.group_norm is not None:
                hidden_states = module.group_norm(
                    hidden_states.transpose(1, 2)).transpose(1, 2)

            query = module.to_q(hidden_states)
            if is_cross:
                encoder_hidden_states = (encoder_hidden_states
                                         if encoder_hidden_states is not None else hidden_states)
                key = module.to_k(encoder_hidden_states)
                value = module.to_v(encoder_hidden_states)
            else:
                key = module.to_k(hidden_states)
                value = module.to_v(hidden_states)

            query = module.reshape_heads_to_batch_dim(query)
            key = module.reshape_heads_to_batch_dim(key)
            value = module.reshape_heads_to_batch_dim(value)

            if getattr(module, "_use_memory_efficient_attention_xformers", False) and query.shape[-2] > 32 ** 2:
                # 大层（>1024 token）走 xformers，不经过 controller，两侧对称跳过
                hidden_states = module._memory_efficient_attention_xformers(
                    query, key, value, attention_mask)
                hidden_states = hidden_states.to(query.dtype)
            else:
                hidden_states = _controller_attention(query, key, value, attention_mask)

            hidden_states = to_out(hidden_states)
            return module.to_out[1](hidden_states)

        return forward

    def register_recr(net_, count, place_in_unet):
        name, module = net_
        cls = module.__class__.__name__
        if name == 'attn1' and cls == 'SparseCausalAttention':
            module.forward = patch_forward(module, False, place_in_unet)
            return count + 1
        if name == 'attn2' and cls == 'CrossAttention':
            module.forward = patch_forward(module, True, place_in_unet)
            return count + 1
        # attn_temp：不注册（原版语义）
        if hasattr(module, 'children'):
            for sub in module.named_children():
                count = register_recr(sub, count, place_in_unet)
        return count

    att_count = 0
    for net in model.unet.named_children():
        if "down" in net[0]:
            att_count += register_recr(net, 0, "down")
        elif "up" in net[0]:
            att_count += register_recr(net, 0, "up")
        elif "mid" in net[0]:
            att_count += register_recr(net, 0, "mid")
    print(f'[fatezero_register] registered {att_count} attention layers '
          f'(attn1=self, attn2=cross, attn_temp skipped)')
    controller.num_att_layers = att_count
