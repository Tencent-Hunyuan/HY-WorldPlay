# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

from typing import Any, Tuple, Optional, Union, Dict

import torch
import torch.nn as nn
from einops import rearrange
from loguru import logger

from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from .modules.activation_layers import get_activation_layer
from .modules.norm_layers import get_norm_layer
from .modules.embed_layers import TimestepEmbedder, PatchEmbed, TextProjection, VisionProjection
from .modules.attention import parallel_attention, sequence_parallel_attention_txt, sequence_parallel_attention_vision
from .modules.posemb_layers import apply_rotary_emb, get_nd_rotary_pos_embed
from .modules.mlp_layers import MLP, MLPEmbedder, FinalLayer, LinearWarpforSingle
from .modules.modulate_layers import ModulateDiT, modulate, apply_gate
from .modules.token_refiner import SingleTokenRefiner

from hyvideo.utils.communications import all_gather
from hyvideo.utils.infer_utils import torch_compile_wrapper
from hyvideo.models.text_encoders.byT5 import ByT5Mapper
from hyvideo.commons.parallel_states import get_parallel_state

from hyvideo.prope.camera_rope import prope_qkv

def is_blocks(n: str, m) -> bool:
    print('is_blocks', n, flush=True)
    is_valid = (
        "double_blocks" in n and str.isdigit(n.split(".")[-1])
    ) or (
        "single_blocks" in n and str.isdigit(n.split(".")[-1])
    )
    return is_valid


class MMDoubleStreamBlock(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu_tanh",
        attn_mode: str = None,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.heads_num = heads_num
        self.attn_mode = attn_mode

        self.hidden_size = hidden_size
        self.qkv_bias = qkv_bias
        self.factory_kwargs = factory_kwargs

        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.img_mod = ModulateDiT(
            hidden_size, factor=6, act_layer=get_activation_layer("silu"), **factory_kwargs
        )
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.img_attn_q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.img_attn_k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.img_attn_v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.img_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.img_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.img_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.img_mlp = MLP(
            hidden_size, mlp_hidden_dim, act_layer=get_activation_layer(mlp_act_type), bias=True, **factory_kwargs
        )

        self.txt_mod = ModulateDiT(
            hidden_size, factor=6, act_layer=get_activation_layer("silu"), **factory_kwargs
        )
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.txt_attn_q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.txt_attn_k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.txt_attn_v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        self.txt_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.txt_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.txt_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.txt_mlp = MLP(
            hidden_size, mlp_hidden_dim, act_layer=get_activation_layer(mlp_act_type), bias=True, **factory_kwargs
        )

        self.hybrid_seq_parallel_attn = None

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    # modulating module for txt embedding
    def modulate_txt(self,
                     vec_txt,
                     txt):
        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.txt_mod(vec_txt).chunk(6, dim=-1)

        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale)
        txt_q = self.txt_attn_q(txt_modulated)
        txt_k = self.txt_attn_k(txt_modulated)
        txt_v = self.txt_attn_v(txt_modulated)
        txt_q = rearrange(txt_q, "B L (H D) -> B L H D", H=self.heads_num)
        txt_k = rearrange(txt_k, "B L (H D) -> B L H D", H=self.heads_num)
        txt_v = rearrange(txt_v, "B L (H D) -> B L H D", H=self.heads_num)
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)
        return txt_q, txt_k, txt_v, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate

    # modulating module for img embedding
    def modulate_img(self,
                     vec,
                     img):
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.img_mod(vec).chunk(6, dim=-1)

        img_modulated = self.img_norm1(img)
        img_modulated = modulate(img_modulated, shift=img_mod1_shift, scale=img_mod1_scale)

        img_q = self.img_attn_q(img_modulated)
        img_k = self.img_attn_k(img_modulated)
        img_v = self.img_attn_v(img_modulated)
        img_q = rearrange(img_q, "B L (H D) -> B L H D", H=self.heads_num)
        img_k = rearrange(img_k, "B L (H D) -> B L H D", H=self.heads_num)
        img_v = rearrange(img_v, "B L (H D) -> B L H D", H=self.heads_num)
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)
        return img_q, img_k, img_v, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate

    @torch_compile_wrapper()
    def forward_txt(
            self,
            txt: torch.Tensor,
            vec_txt: torch.Tensor,
            text_mask=None,
            attn_param=None,
            is_flash=False,
            block_idx=None,
            kv_cache: Optional[dict] = None,
            cache_txt: bool = False,
    ) -> Tuple[torch.Tensor]:
        (txt_q, txt_k, txt_v, txt_mod1_gate,
         txt_mod2_shift, txt_mod2_scale, txt_mod2_gate) = self.modulate_txt(vec_txt, txt)

        attn_mode = 'torch_causal'  # for ar model, the default mode is torch_causal
        txt_attn = sequence_parallel_attention_txt(
            (txt_q),
            (txt_k),
            (txt_v),
            img_q_len=txt_q.shape[1],
            img_kv_len=txt_k.shape[1],
            text_mask=text_mask,
            attn_mode=attn_mode,
            attn_param=attn_param,
            block_idx=block_idx,
            kv_cache=kv_cache,
            cache_txt=cache_txt,
        )

        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(modulate(self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale)),
            gate=txt_mod2_gate,
        )
        return txt

    @torch_compile_wrapper()
    def forward_vision(
            self,
            img: torch.Tensor,
            vec: torch.Tensor,
            freqs_cis: tuple = None,
            attn_param=None,
            block_idx=None,
            viewmats: Optional[torch.Tensor] = None,
            Ks: Optional[torch.Tensor] = None,
            kv_cache: Optional[dict] = None,
            cache_vision: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_q, img_k, img_v, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate = self.modulate_img(vec, img)

        # add camera pose through prope
        # delete rope components (original attn included)
        img_q_prope, img_k_prope, img_v_prope, apply_fn_o = prope_qkv(
            img_q.permute(0, 2, 1, 3),
            img_k.permute(0, 2, 1, 3),
            img_v.permute(0, 2, 1, 3),
            viewmats=viewmats,
            Ks=Ks,
        )  # [batch, num_heads, seqlen, head_dim]
        img_q_prope = img_q_prope.permute(0, 2, 1, 3)  # [batch, seqlen, num_heads, head_dim]
        img_k_prope = img_k_prope.permute(0, 2, 1, 3)  # [batch, seqlen, num_heads, head_dim]
        img_v_prope = img_v_prope.permute(0, 2, 1, 3)  # [batch, seqlen, num_heads, head_dim]

        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert (
                    img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk

        img_attn, img_attn_prope = sequence_parallel_attention_vision(
            (img_q, img_q_prope),
            (img_k, img_k_prope),
            (img_v, img_v_prope),
            block_idx=block_idx,
            kv_cache=kv_cache,
            cache_vision=cache_vision,
        )

        img_attn_prope = rearrange(img_attn_prope, "B L (H D) -> B H L D", H=self.heads_num)
        img_attn_prope = apply_fn_o(img_attn_prope)  # [batch, num_heads, seqlen, head_dim]
        img_attn_prope = rearrange(img_attn_prope, "B H L D -> B L (H D)")

        img = img + apply_gate(self.img_attn_proj(img_attn) + self.img_attn_prope_proj(img_attn_prope),
                               gate=img_mod1_gate)
        img = img + apply_gate(
            self.img_mlp(
                modulate(self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale)
            ),
            gate=img_mod2_gate,
        )

        return img

    @torch_compile_wrapper()
    def forward_bi(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec_txt: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis: tuple = None,
        text_mask=None,
        attn_param=None,
        is_flash=False,
        block_idx=None,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_q, img_k, img_v, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate = self.modulate_img(vec, img)
        (txt_q, txt_k, txt_v, txt_mod1_gate,
         txt_mod2_shift, txt_mod2_scale, txt_mod2_gate) = self.modulate_txt(vec_txt, txt)

        # add camera pose through prope
        img_q_prope, img_k_prope, img_v_prope, apply_fn_o = prope_qkv(
            img_q.permute(0, 2, 1, 3),
            img_k.permute(0, 2, 1, 3),
            img_v.permute(0, 2, 1, 3),
            viewmats=viewmats,
            Ks=Ks,
        )  # [batch, num_heads, seqlen, head_dim]
        img_q_prope = img_q_prope.permute(0, 2, 1, 3) # [batch, seqlen, num_heads, head_dim]
        img_k_prope = img_k_prope.permute(0, 2, 1, 3) # [batch, seqlen, num_heads, head_dim]
        img_v_prope = img_v_prope.permute(0, 2, 1, 3) # [batch, seqlen, num_heads, head_dim]

        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk

        attn_mode = 'flash' if is_flash else self.attn_mode
        attn = parallel_attention(
            (img_q, txt_q),
            (img_k, txt_k),
            (img_v, txt_v),
            img_q_len=img_q.shape[1],
            img_kv_len=img_k.shape[1],
            text_mask=text_mask,
            attn_mode=attn_mode,
            attn_param=attn_param,
            block_idx=block_idx,
        )
        img_attn, txt_attn = attn[:, :img_q.shape[1]].contiguous(), attn[:, img_q.shape[1]:].contiguous()

        attn_prope = parallel_attention(
            (img_q_prope, txt_q),
            (img_k_prope, txt_k),
            (img_v_prope, txt_v),
            img_q_len=img_q.shape[1],
            img_kv_len=img_k.shape[1],
            text_mask=text_mask,
            attn_mode=attn_mode,
            attn_param=attn_param,
            block_idx=block_idx,
        )
        img_attn_prope = attn_prope[:, :img_q_prope.shape[1]].contiguous()
        img_attn_prope = rearrange(img_attn_prope, "B L (H D) -> B H L D", H=self.heads_num)
        img_attn_prope = apply_fn_o(img_attn_prope) # [batch, num_heads, seqlen, head_dim]
        img_attn_prope = rearrange(img_attn_prope, "B H L D -> B L (H D)")

        img = img + apply_gate(self.img_attn_proj(img_attn) +
                               self.img_attn_prope_proj(img_attn_prope), gate=img_mod1_gate)
        img = img + apply_gate(
            self.img_mlp(
                modulate(self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale)
            ),
            gate=img_mod2_gate,
        )

        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(modulate(self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale)),
            gate=txt_mod2_gate,
        )
        return img, txt

    @torch_compile_wrapper()
    def forward(
            self,
            img: torch.Tensor,
            txt: torch.Tensor,
            vec_txt: torch.Tensor,
            vec: torch.Tensor,
            freqs_cis: tuple = None,
            text_mask=None,
            attn_param=None,
            is_flash=False,
            block_idx=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_q, img_k, img_v, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate = self.modulate_img(vec, img)
        (txt_q, txt_k, txt_v, txt_mod1_gate,
         txt_mod2_shift, txt_mod2_scale, txt_mod2_gate) = self.modulate_txt(vec_txt, txt)

        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert (
                    img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk

        attn_mode = 'flash' if is_flash else self.attn_mode
        attn = parallel_attention(
            (img_q, txt_q),
            (img_k, txt_k),
            (img_v, txt_v),
            img_q_len=img_q.shape[1],
            img_kv_len=img_k.shape[1],
            text_mask=text_mask,
            attn_mode=attn_mode,
            attn_param=attn_param,
            block_idx=block_idx,
        )

        img_attn, txt_attn = attn[:, :img_q.shape[1]].contiguous(), attn[:, img_q.shape[1]:].contiguous()

        img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
        img = img + apply_gate(
            self.img_mlp(
                modulate(self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale)
            ),
            gate=img_mod2_gate,
        )

        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(modulate(self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale)),
            gate=txt_mod2_gate,
        )

        return img, txt

class MMSingleStreamBlock(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        attn_mode: str = None,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qk_scale: float = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.attn_mode = attn_mode

        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.linear1_q = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.linear1_k = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.linear1_v = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.linear1_mlp = nn.Linear(hidden_size, mlp_hidden_dim, **factory_kwargs)
        self.linear2 = LinearWarpforSingle(hidden_size + mlp_hidden_dim, hidden_size, bias=True, **factory_kwargs)
        self.mlp_act = get_activation_layer(mlp_act_type)()

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )

        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.modulation = ModulateDiT(hidden_size, factor=3, act_layer=get_activation_layer("silu"), **factory_kwargs)
        self.hybrid_seq_parallel_attn = None

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        x: torch.Tensor,
        vec_txt: torch.Tensor,
        vec: torch.Tensor,
        txt_len: int,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        text_mask=None,
        attn_param=None,
        is_flash=False,
    ) -> torch.Tensor:
        """Forward pass for the single stream block."""
        mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
        txt_mod_shift, txt_mod_scale, txt_mod_gate = self.modulation(vec_txt).chunk(3, dim=-1)

        x_pre_norm = self.pre_norm(x)
        img_token, txt_token = x_pre_norm[:, :-txt_len, :], x_pre_norm[:, -txt_len:, :]

        img_mod = modulate(img_token, shift=mod_shift, scale=mod_scale)
        txt_mod = modulate(txt_token, shift=txt_mod_shift, scale=txt_mod_scale)
        x_mod = torch.cat([img_mod, txt_mod], dim=1)

        q = self.linear1_q(x_mod)
        k = self.linear1_k(x_mod)
        v = self.linear1_v(x_mod)

        q = rearrange(q, "B L (H D) -> B L H D", H=self.heads_num)
        k = rearrange(k, "B L (H D) -> B L H D", H=self.heads_num)
        v = rearrange(v, "B L (H D) -> B L H D", H=self.heads_num)
        
        mlp = self.linear1_mlp(x_mod)

        # Apply QK-Norm if needed.
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
        img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
        img_v, txt_v = v[:, :-txt_len, :, :], v[:, -txt_len:, :, :]
        img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
        assert (
            img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
        ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
        img_q, img_k = img_qq, img_kk

        if is_flash:
            attn_mode = 'flash'
        else:
            attn_mode = self.attn_mode
        attn = parallel_attention(
            (img_q, txt_q),
            (img_k, txt_k),
            (img_v, txt_v),
            img_q_len=img_q.shape[1],
            img_kv_len=img_k.shape[1],
            text_mask=text_mask,
            attn_mode=attn_mode,
            attn_param=attn_param,
        )
        output = self.linear2(attn, self.mlp_act(mlp))
        img_output, txt_output = output[:, :-txt_len, :], output[:, -txt_len:, :]

        img_output = apply_gate(img_output, gate=mod_gate)
        txt_output = apply_gate(txt_output, gate=txt_mod_gate)
        gate_output = torch.cat([img_output, txt_output], dim=1)

        return x + gate_output

class HunyuanVideo_1_5_DiffusionTransformer(ModelMixin, ConfigMixin):
    """
    HunyuanVideo Transformer backbone.

    Args:
        patch_size (list): The size of the patch.
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        hidden_size (int): The hidden size of the transformer backbone.
        heads_num (int): The number of attention heads.
        mlp_width_ratio (float): Width ratio for the transformer MLPs.
        mlp_act_type (str): Activation type for the transformer MLPs.
        mm_double_blocks_depth (int): Number of double-stream transformer blocks.
        mm_single_blocks_depth (int): Number of single-stream transformer blocks.
        rope_dim_list (list): Rotary embedding dim for t, h, w.
        qkv_bias (bool): Use bias in qkv projection.
        qk_norm (bool): Whether to use qk norm.
        qk_norm_type (str): Type of qk norm.
        guidance_embed (bool): Use guidance embedding for distillation.
        text_projection (str): Text input projection. Default is "single_refiner".
        use_attention_mask (bool): If to use attention mask.
        text_states_dim (int): Text encoder output dim.
        text_states_dim_2 (int): Secondary text encoder output dim.
        text_pool_type (str): Type for text pooling.
        rope_theta (int): Rotary embedding theta parameter.
        attn_mode (str): Attention mode identifier.
        attn_param (dict): Attention parameter dictionary.
        glyph_byT5_v2 (bool): Use ByT5 glyph module.
        vision_projection (str): Vision condition embedding mode.
        vision_states_dim (int): Vision encoder states input dim.
        is_reshape_temporal_channels (bool): For video VAE adaptation.
        use_cond_type_embedding (bool): Use condition type embedding.
    """

    @register_to_config
    def __init__(
        self,
        patch_size: list = [1, 2, 2],
        in_channels: int = 4,
        concat_condition: bool = True,
        out_channels: int = None,
        hidden_size: int = 3072,
        heads_num: int = 24,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        mm_double_blocks_depth: int = 20,
        mm_single_blocks_depth: int = 40,
        rope_dim_list: list = [16, 56, 56],
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        guidance_embed: bool = False,
        use_meanflow: bool = False,
        text_projection: str = "single_refiner",
        use_attention_mask: bool = True,
        text_states_dim: int = 4096,
        text_states_dim_2: int = 768,
        text_pool_type: str = None,
        rope_theta: int = 256,
        attn_mode: str = "flash",
        attn_param: dict = None,
        glyph_byT5_v2: bool = False,
        vision_projection: str = "none",
        vision_states_dim: int = 1280,
        is_reshape_temporal_channels: bool = False,
        use_cond_type_embedding: bool = False,
        ideal_resolution: str = None,
        ideal_task: str = None,
    ):
        super().__init__()
        factory_kwargs = {}

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embed = guidance_embed
        self.rope_dim_list = rope_dim_list
        self.rope_theta = rope_theta
        # Text projection. Default to linear projection.
        # Alternative: TokenRefiner. See more details (LI-DiT): http://arxiv.org/abs/2406.11831
        self.use_attention_mask = use_attention_mask
        self.text_projection = text_projection
        self.attn_mode = attn_mode
        self.text_pool_type = text_pool_type
        self.text_states_dim = text_states_dim
        self.text_states_dim_2 = text_states_dim_2
        self.vision_states_dim = vision_states_dim

        self.glyph_byT5_v2 = glyph_byT5_v2
        if self.glyph_byT5_v2:
            self.byt5_in = ByT5Mapper(
                in_dim=1472,
                out_dim=2048,
                hidden_dim=2048,
                out_dim1=hidden_size,
                use_residual=False
            )

        if hidden_size % heads_num != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by heads_num {heads_num}"
            )
        pe_dim = hidden_size // heads_num
        if sum(rope_dim_list) != pe_dim:
            raise ValueError(
                f"Got {rope_dim_list} but expected positional dim {pe_dim}"
            )
        self.hidden_size = hidden_size
        self.heads_num = heads_num

        self.img_in = PatchEmbed(
            self.patch_size, self.in_channels, self.hidden_size,
            is_reshape_temporal_channels=is_reshape_temporal_channels, 
            concat_condition=concat_condition, **factory_kwargs
        )

        # Vision projection
        if vision_projection == "linear":
            self.vision_in = VisionProjection(
                input_dim=self.vision_states_dim, output_dim=self.hidden_size
            )
        else:
            self.vision_in = None

        # Text projection
        if self.text_projection == "linear":
            self.txt_in = TextProjection(
                text_states_dim,
                self.hidden_size,
                get_activation_layer("silu"),
                **factory_kwargs,
            )
        elif self.text_projection == "single_refiner":
            self.txt_in = SingleTokenRefiner(
                text_states_dim,
                hidden_size,
                heads_num,
                depth=2,
                **factory_kwargs,
            )
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        # time modulation
        self.time_in = TimestepEmbedder(
            self.hidden_size, get_activation_layer("silu"), **factory_kwargs
        )
        self.vector_in = (
            MLPEmbedder(
                self.config.text_states_dim_2, self.hidden_size, **factory_kwargs
            ) if self.text_pool_type is not None else None
        )
        self.guidance_in = (
            TimestepEmbedder(
                self.hidden_size, get_activation_layer("silu"), **factory_kwargs
            )
            if guidance_embed
            else None
        )

        self.time_r_in = (
            TimestepEmbedder(self.hidden_size, get_activation_layer("silu"), **factory_kwargs)
            if use_meanflow
            else None
        )


        self.double_blocks = nn.ModuleList(
            [
                MMDoubleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    attn_mode=attn_mode,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    **factory_kwargs,
                )
                for _ in range(mm_double_blocks_depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                MMSingleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    attn_mode=attn_mode,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    **factory_kwargs,
                )
                for _ in range(mm_single_blocks_depth)
            ]
        )

        self.final_layer = FinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
            get_activation_layer("silu"),
            **factory_kwargs,
        )

        # STA
        if attn_param is None:
            self.attn_param = { 
                # STA
                'win_size': [[3, 3, 3]],
                'win_type': "fixed",
                'win_ratio': 10,
                'tile_size': [6, 8, 8],
                # SSTA
                'ssta_topk': 64,
                'ssta_threshold': 0.0,
                'ssta_lambda': 0.7,
                'ssta_sampling_type': 'importance',
                'ssta_adaptive_pool': None,
                # flex-block-attn:
                "attn_sparse_type": 'ssta',
                "attn_pad_type": 'zero',
                "attn_use_text_mask": 1,
                "attn_mask_share_within_head": 0,
            }
        else:
            self.attn_param = attn_param

        if attn_mode == 'flex-block-attn':
            self.register_to_config(attn_param=self.attn_param)

        if use_cond_type_embedding:
            self.cond_type_embedding = nn.Embedding(3, self.hidden_size) 
            self.cond_type_embedding.weight.data.fill_(0)
            assert self.glyph_byT5_v2, "text type embedding is only used when glyph_byT5_v2 is True"
            assert vision_projection is not None, "text type embedding is only used when vision_projection is not None"
            # 0: text_encoder feature
            # 1: byt5 feature
            # 2: vision_encoder feature
        else:
            self.cond_type_embedding = None

        self.gradient_checkpointing = False

    def load_hunyuan_state_dict(self, model_path):
        load_key = "module"
        bare_model = "unknown"

        if model_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(model_path, device="cpu")
        else:
            state_dict = torch.load(
                model_path, map_location="cpu", weights_only=True
            )

        if bare_model == "unknown" and ("ema" in state_dict or "module" in state_dict):
            bare_model = False
        if bare_model is False:
            if load_key in state_dict:
                state_dict = state_dict[load_key]
            else:
                raise KeyError(
                    f"Missing key: `{load_key}` in the checkpoint: {model_path}. The keys in the checkpoint "
                    f"are: {list(state_dict.keys())}."
                )

        result = self.load_state_dict(state_dict, strict=False)

        if result.missing_keys:
            logger.info("[load.py] Missing keys when loading state_dict:")
            for key in result.missing_keys:
                logger.info(f"[load.py] Missing key: {key}")
        if result.unexpected_keys:
            logger.info("[load.py] Unexpected keys when loading state_dict:")
            for key in result.unexpected_keys:
                logger.info(f"[load.py] Unexpected key: {key}")
        if result.missing_keys or result.unexpected_keys:
            raise ValueError(f"Missing: {result.missing_keys}, Unexpected: {result.unexpected_keys}")

        return result

    def enable_deterministic(self):
        for block in self.double_blocks:
            block.enable_deterministic()
        for block in self.single_blocks:
            block.enable_deterministic()

    def disable_deterministic(self):
        for block in self.double_blocks:
            block.disable_deterministic()
        for block in self.single_blocks:
            block.disable_deterministic()

    def get_rotary_pos_embed(self, rope_sizes):
        target_ndim = 3
        head_dim = self.hidden_size // self.heads_num
        rope_dim_list = self.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert (
            sum(rope_dim_list) == head_dim
        ), "sum(rope_dim_list) should equal to head_dim of attention layer"
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list,
            rope_sizes,
            theta=self.rope_theta,
            use_real=True,
            theta_rescale_factor=1,
        )
        return freqs_cos, freqs_sin

    def reorder_txt_token(self, byt5_txt, txt, byt5_text_mask, text_mask, zero_feat=False, is_reorder=True):
        if is_reorder:
            reorder_txt = []
            reorder_mask = []
            for i in range(text_mask.shape[0]):
                byt5_text_mask_i = byt5_text_mask[i].bool()
                text_mask_i = text_mask[i].bool()

                byt5_txt_i = byt5_txt[i]
                txt_i = txt[i]
                if zero_feat:
                    # When using block mask with approximate computation, set pad to zero to reduce error
                    pad_byt5 = torch.zeros_like(byt5_txt_i[~byt5_text_mask_i])
                    pad_text = torch.zeros_like(txt_i[~text_mask_i])
                    reorder_txt_i = torch.cat([byt5_txt_i[byt5_text_mask_i],
                                               txt_i[text_mask_i], pad_byt5, pad_text], dim=0)
                else:
                    reorder_txt_i = torch.cat([byt5_txt_i[byt5_text_mask_i], txt_i[text_mask_i],
                                               byt5_txt_i[~byt5_text_mask_i], txt_i[~text_mask_i]], dim=0)
                reorder_mask_i = torch.cat([byt5_text_mask_i[byt5_text_mask_i], text_mask_i[text_mask_i],
                                            byt5_text_mask_i[~byt5_text_mask_i], text_mask_i[~text_mask_i]], dim=0)

                reorder_txt.append(reorder_txt_i)
                reorder_mask.append(reorder_mask_i)

            reorder_txt = torch.stack(reorder_txt)
            reorder_mask = torch.stack(reorder_mask).to(dtype=torch.int64)
        else:
            reorder_txt = torch.concat([byt5_txt, txt], dim=1)
            reorder_mask = torch.concat([byt5_text_mask, text_mask], dim=1).to(dtype=torch.int64)

        return reorder_txt, reorder_mask

    def add_action_parameters(self):
        # Initialize the embedding module for action conditioning (using Silu activation).
        self.action_in = TimestepEmbedder(
            self.hidden_size, get_activation_layer("silu")
        )

        # Zero-initialize the weights of the final layer (mlp[2]) of the embedder.
        # This ensures the initial output is zero, preventing the action conditioning
        # from affecting the model's behavior at the start of training.
        nn.init.zeros_(self.action_in.mlp[2].weight)

        # Zero-initialize the bias term if it exists.
        if self.action_in.mlp[2].bias is not None:
            nn.init.zeros_(self.action_in.mlp[2].bias)

        # Iterate through each transformer double-block to inject action-specific projections.
        for block in self.double_blocks:
            # Create a new linear projection layer for image attention properties.
            block.img_attn_prope_proj = nn.Linear(
                block.hidden_size,
                block.hidden_size,
                bias=block.qkv_bias,
                **block.factory_kwargs
            )

            # Zero-initialize the weights of this new projection layer.
            # Again, this ensures the new layer acts as an identity function initially.
            nn.init.zeros_(block.img_attn_prope_proj.weight)

            # Zero-initialize the bias term for the projection layer.
            if block.img_attn_prope_proj.bias is not None:
                nn.init.zeros_(block.img_attn_prope_proj.bias)

    def get_text_and_mask(self,
                    encoder_attention_mask,
                    text_states,
                    timestep_txt,
                    extra_kwargs,
                    vision_states,
                    mask_type):
        text_mask = encoder_attention_mask
        txt = text_states
        bs = txt.shape[0]

        # Prepare modulation vectors
        vec_txt = self.time_in(timestep_txt)

        # Embed text tokens
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, timestep_txt, text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )
        if self.cond_type_embedding is not None:
            cond_emb = self.cond_type_embedding(
                torch.zeros_like(txt[:, :, 0], device=text_mask.device, dtype=torch.long)
            )
            txt = txt + cond_emb

        if self.glyph_byT5_v2:
            byt5_text_states = extra_kwargs["byt5_text_states"]
            byt5_text_mask = extra_kwargs["byt5_text_mask"]
            byt5_txt = self.byt5_in(byt5_text_states)
            if self.cond_type_embedding is not None:
                cond_emb = self.cond_type_embedding(
                    torch.ones_like(byt5_txt[:, :, 0], device=byt5_txt.device, dtype=torch.long)
                )
                byt5_txt = byt5_txt + cond_emb
            txt, text_mask = self.reorder_txt_token(
                byt5_txt, txt, byt5_text_mask, text_mask, zero_feat=True
            )

        if self.vision_in is not None and vision_states is not None:
            extra_encoder_hidden_states = self.vision_in(vision_states)
            # If t2v, set extra_attention_mask to 0 to avoid attention to semantic tokens
            if mask_type == "t2v" and torch.all(vision_states == 0):
                extra_attention_mask = torch.zeros(
                    (bs, extra_encoder_hidden_states.shape[1]),
                    dtype=text_mask.dtype,
                    device=text_mask.device,
                )
                # Set vision tokens to zero to mitigate potential block mask error in SSTA
                extra_encoder_hidden_states = extra_encoder_hidden_states * 0.0
            else:
                extra_attention_mask = torch.ones(
                    (bs, extra_encoder_hidden_states.shape[1]),
                    dtype=text_mask.dtype,
                    device=text_mask.device,
                )
            # Ensure valid tokens precede padding tokens
            if self.cond_type_embedding is not None:
                cond_emb = self.cond_type_embedding(
                    2 * torch.ones_like(
                        extra_encoder_hidden_states[:, :, 0],
                        dtype=torch.long,
                        device=extra_encoder_hidden_states.device,
                    )
                )
                extra_encoder_hidden_states = extra_encoder_hidden_states + cond_emb

            txt, text_mask = self.reorder_txt_token(
                extra_encoder_hidden_states, txt, extra_attention_mask, text_mask
            )
        return txt, text_mask, vec_txt

    # txt embedding kv cache
    def forward_txt(
        self,
        timestep_txt: torch.Tensor,
        text_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        vision_states: torch.Tensor = None,
        mask_type="t2v",
        extra_kwargs=None,
        kv_cache: Optional[dict] = None,
        cache_txt: Optional[bool] = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        txt, text_mask, vec_txt = self.get_text_and_mask(
            encoder_attention_mask,
            text_states,
            timestep_txt,
            extra_kwargs,
            vision_states,
            mask_type
        )

        # mask the txt tokens in advance for efficiency
        txt = txt[text_mask.bool().to(txt.device)].unsqueeze(0)

        # Pass through double-stream blocks
        for index, block in enumerate(self.double_blocks):
            txt = block.forward_txt(
                txt=txt,
                vec_txt=vec_txt,
                text_mask=None,      # we have masked txt tokens already, set None here
                attn_param=None,
                is_flash=False,
                block_idx=index,
                kv_cache=kv_cache,
                cache_txt=cache_txt,
            )

    # using kv cache to calculate vision embedding
    def forward_vision(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        timestep_r=None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        return_dict: bool = False,
        mask_type="t2v",
        extra_kwargs=None,
        action: Optional[torch.Tensor] = None,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_vision: bool = False,
        rope_temporal_size=4,
        start_rope_start_idx=0,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:

        img = x = hidden_states
        t = timestep
        bs, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )
        self.attn_param['thw'] = [tt, th, tw]
        rope_temporal_size = rope_temporal_size // self.patch_size[0]
        if freqs_cos is None and freqs_sin is None:
            freqs_cos, freqs_sin = self.get_rotary_pos_embed((rope_temporal_size, th, tw))
            per_latent_size = th * tw
            start_index = start_rope_start_idx * per_latent_size
            end_index = (start_rope_start_idx + tt) * per_latent_size
            freqs_cos = freqs_cos[start_index: end_index,...]
            freqs_sin = freqs_sin[start_index: end_index,...]

        img = self.img_in(img)

        parallel_dims = get_parallel_state()
        sp_enabled = parallel_dims.sp_enabled
        if sp_enabled:
            sp_size = parallel_dims.sp
            sp_rank = parallel_dims.sp_rank
            if img.shape[1] % sp_size != 0:
                n_token = img.shape[1]
                assert n_token > (n_token // sp_size + 1) * (sp_size - 1), f'Too short context length for SP {sp_size}'
            img = torch.chunk(img, sp_size, dim=1)[sp_rank]
            freqs_cos = torch.chunk(freqs_cos, sp_size, dim=0)[sp_rank]
            freqs_sin = torch.chunk(freqs_sin, sp_size, dim=0)[sp_rank]

            viewmats = torch.chunk(viewmats, sp_size, dim=1)[sp_rank]
            Ks = torch.chunk(Ks, sp_size, dim=1)[sp_rank]
            action = action.reshape(img.shape[0], -1)
            t = t.reshape(img.shape[0], -1)
            action = torch.chunk(action, sp_size, dim=1)[sp_rank]
            t = torch.chunk(t, sp_size, dim=1)[sp_rank]
            action = action.reshape(-1)
            t = t.reshape(-1)

        # Prepare modulation vectors
        vec = self.time_in(t)

        vec = vec + self.action_in(action) # inject discrete action

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

        # Pass through double-stream blocks
        for index, block in enumerate(self.double_blocks):
            self.attn_param["layer-name"] = f"double_block_{index+1}"

            img = block.forward_vision(
                img=img,
                vec=vec,
                freqs_cis=freqs_cis,
                attn_param=self.attn_param,
                block_idx=index,
                viewmats=viewmats,
                Ks=Ks,
                kv_cache=kv_cache,
                cache_vision=cache_vision,
            )

        # Final Layer
        img = self.final_layer(img, vec)
        if sp_enabled:
            img = all_gather(img, dim=1)
        img = self.unpatchify(img, tt, th, tw)
        assert return_dict is False, "return_dict is not supported."
        features_list = None
        return (img, features_list)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        timestep_txt: torch.Tensor,
        text_states: torch.Tensor,
        text_states_2: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        timestep_r=None,
        vision_states: torch.Tensor = None,
        output_features=False,
        output_features_stride=8,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        return_dict: bool = False,
        guidance=None,
        mask_type="t2v",
        extra_kwargs=None,
        action: Optional[torch.Tensor] = None,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if guidance is None:
            guidance = torch.tensor(
                [6016.0], device=hidden_states.device, dtype=torch.bfloat16
            )

        img = x = hidden_states
        text_mask = encoder_attention_mask
        t = timestep
        txt = text_states
        bs, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )
        self.attn_param['thw'] = [tt, th, tw]
        if freqs_cos is None and freqs_sin is None:
            freqs_cos, freqs_sin = self.get_rotary_pos_embed((tt, th, tw))

        img = self.img_in(img)

        parallel_dims = get_parallel_state()
        sp_enabled = parallel_dims.sp_enabled
        if sp_enabled:
            sp_size = parallel_dims.sp
            sp_rank = parallel_dims.sp_rank
            if img.shape[1] % sp_size != 0:
                n_token = img.shape[1]
                assert n_token > (n_token // sp_size + 1) * (sp_size - 1), f'Too short context length for SP {sp_size}'
            img = torch.chunk(img, sp_size, dim=1)[sp_rank]
            freqs_cos = torch.chunk(freqs_cos, sp_size, dim=0)[sp_rank]
            freqs_sin = torch.chunk(freqs_sin, sp_size, dim=0)[sp_rank]

            if action is not None:
                viewmats = torch.chunk(viewmats, sp_size, dim=1)[sp_rank]
                Ks = torch.chunk(Ks, sp_size, dim=1)[sp_rank]
                action = action.reshape(img.shape[0], -1)
                action = torch.chunk(action, sp_size, dim=1)[sp_rank]
                action = action.reshape(-1)
                t = t.reshape(img.shape[0], -1)
                t = torch.chunk(t, sp_size, dim=1)[sp_rank]
                t = t.reshape(-1)

        vec = self.time_in(t)

        if text_states_2 is not None:
            vec_2 = self.vector_in(text_states_2)
            vec = vec + vec_2

        if self.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )
            vec = vec + self.guidance_in(guidance)

        if timestep_r is not None:
            vec = vec + self.time_r_in(timestep_r)

        if action is not None:
            vec = vec + self.action_in(action) # 添加离散的action

        txt, text_mask, vec_txt = self.get_text_and_mask(
            encoder_attention_mask,
            text_states,
            timestep_txt,
            extra_kwargs,
            vision_states,
            mask_type
        )

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

        # Pass through double-stream blocks
        for index, block in enumerate(self.double_blocks):
            force_full_attn = (
                self.attn_mode in ["flex-block-attn"]
                and self.attn_param["win_type"] == "hybrid"
                and self.attn_param["win_ratio"] > 0
                and (
                    (index + 1) % self.attn_param["win_ratio"] == 0
                    or (index + 1) == len(self.double_blocks)
                )
            )
            self.attn_param["layer-name"] = f"double_block_{index+1}"
            if viewmats is not None:
                img, txt = block.forward_bi(
                    img=img,
                    txt=txt,
                    vec_txt=vec_txt,
                    vec=vec,
                    freqs_cis=freqs_cis,
                    text_mask=text_mask,      # we have masked txt tokens already, set None here
                    attn_param=self.attn_param,
                    is_flash=force_full_attn,
                    block_idx=index,
                    viewmats=viewmats,
                    Ks=Ks,
                )
            else:
                img, txt = block(
                    img=img,
                    txt=txt,
                    vec_txt=vec_txt,
                    vec=vec,
                    freqs_cis=freqs_cis,
                    text_mask=text_mask,  # we have masked txt tokens already, set None here
                    attn_param=self.attn_param,
                    is_flash=force_full_attn,
                    block_idx=index,
                )

        # Final Layer
        img = self.final_layer(img, vec)
        if sp_enabled:
            img = all_gather(img, dim=1)
        img = self.unpatchify(img, tt, th, tw)
        assert return_dict is False, "return_dict is not supported."
        features_list = None
        return (img, features_list)

    def unpatchify(self, x, t, h, w):
        """
        Unpatchify a tensorized input back to frame format.

        Args:
            x (Tensor): Input tensor of shape (N, T, patch_size**2 * C)
            t (int): Number of time steps
            h (int): Height in patch units
            w (int): Width in patch units

        Returns:
            Tensor: Output tensor of shape (N, C, t * pt, h * ph, w * pw)
        """
        c = self.unpatchify_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))
        return imgs

    def set_attn_mode(self, attn_mode: str):
        # Set the attention mode for the current object
        self.attn_mode = attn_mode

        # Iterate through each block in the double_blocks list
        for block in self.double_blocks:
            # Set the attention mode for each block in double_blocks
            block.attn_mode = attn_mode

        # Iterate through each block in the single_blocks list
        for block in self.single_blocks:
            # Set the attention mode for each block in single_blocks
            block.attn_mode = attn_mode

