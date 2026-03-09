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

import torch
from einops import rearrange
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    BlockMask,
)

# compile first
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune"
)


def prepare_blockwise_causal_attn_mask(
    device: torch.device | str,
    num_frames: int = 21,
    frame_seqlen: int = 880,
    causal_mask=None,
) -> BlockMask:
    """We will divide the token sequence into the following format [1 latent frame] [1 latent frame]
    ...

    [1 latent frame] We use flexattention to construct the attention mask
    """

    total_length = num_frames * frame_seqlen

    def attention_mask(b, h, q_idx, kv_idx):
        return causal_mask[q_idx, kv_idx]


# add new function for computing causal attention with flex attention
def flex_attn_no_pad(
    qkv,
    key_padding_mask,
    causal=False,
    dropout_p=0.0,
    softmax_scale=None,
    deterministic=False,
):
    from flash_attn import flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import pad_input, unpad_input

    batch_size = qkv.shape[0]  # qkv shape: [B, total_length, 3, num_head, D]
    seqlen = qkv.shape[1]
    nheads = qkv.shape[-2]
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    x_unpad, indices, cu_seqlens, max_s, used_seqlens_in_batch = unpad_input(
        x, key_padding_mask
    )  # only select the unmasked tokens, the selected token index, cu_seqlens: (0, unmasked token length), max_s: max unmasked length in the batch

    # ------------------------- for chunk-wise causal attention mask -------------------------
    # get the video sequence
    latent_seq_length = (
        1560  # set for hunyuanvideo 1.5, which is for 480 * 832 resolution
    )
    chunk_seq_length = 1560 * 4
    text_seq_length = (
        max_s % chunk_seq_length
    )  # including txt, byt5, sigclip vision token
    latent_num = (max_s - text_seq_length) // latent_seq_length
    chunk_num = (max_s - text_seq_length) // chunk_seq_length
    causal_mask = torch.zeros((max_s, max_s), device=qkv.device)

    causal_mask[:text_seq_length, :text_seq_length] = (
        1  # no attention for the rest
    )
    for i in range(chunk_num):
        start_i = text_seq_length + i * chunk_seq_length
        end_i = min(start_i + chunk_seq_length, max_s)
        for j in range(i + 1):
            start_j = text_seq_length + j * chunk_seq_length
            end_j = min(start_j + chunk_seq_length, max_s)
            # full attention within chunk i for j == i, causal for j < i
            causal_mask[start_i:end_i, start_j:end_j] = 1
    causal_mask = causal_mask.to(torch.bool)  # Force bool dtype

    block_mask = prepare_blockwise_causal_attn_mask(
        device=qkv.device,
        num_frames=latent_num,
        frame_seqlen=latent_seq_length,
        causal_mask=causal_mask,
    )

    q, k, v = qkv.chunk(3, dim=2)
    q = q.squeeze(2).transpose(1, 2)
    k = k.squeeze(2).transpose(1, 2)
    v = v.squeeze(2).transpose(1, 2)
    output_unpad = flex_attention(
        q, k, v, block_mask=block_mask
    )  # output_unpad: [B, num_head, L, D]

    # bidirectional
    # x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)
    # output_unpad = flash_attn_varlen_qkvpacked_func(
    #     x_unpad,
    #     cu_seqlens,
    #     max_s,
    #     dropout_p,
    #     softmax_scale=softmax_scale,
    #     causal=causal,
    #     deterministic=deterministic,
    # )

    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"),
            indices,
            batch_size,
            seqlen,
        ),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    return output


# add new function for computing causal attention with flex attention
def flash_attn_no_pad(
    qkv,
    key_padding_mask,
    causal=False,
    dropout_p=0.0,
    softmax_scale=None,
    deterministic=False,
):
    from flash_attn import flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import pad_input, unpad_input

    batch_size = qkv.shape[0]
    seqlen = qkv.shape[1]
    nheads = qkv.shape[-2]
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    x_unpad, indices, cu_seqlens, max_s, used_seqlens_in_batch = unpad_input(
        x, key_padding_mask
    )

    x_unpad = rearrange(
        x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads
    )
    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad,
        cu_seqlens,
        max_s,
        dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic,
    )
    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"),
            indices,
            batch_size,
            seqlen,
        ),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    return output


def flash_attn_no_pad_v3(
    qkv,
    key_padding_mask,
    causal=False,
    dropout_p=0.0,
    softmax_scale=None,
    deterministic=False,
):
    from flash_attn import flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import pad_input, unpad_input
    from flash_attn_interface import (
        flash_attn_varlen_func as flash_attn_varlen_func_v3,
    )

    if flash_attn_varlen_func_v3 is None:
        raise ImportError("FlashAttention V3 backend not available")

    batch_size, seqlen, _, nheads, head_dim = qkv.shape
    query, key, value = qkv.unbind(dim=2)

    query_unpad, indices, cu_seqlens_q, max_seqlen_q, _ = unpad_input(
        rearrange(query, "b s h d -> b s (h d)"), key_padding_mask
    )
    key_unpad, _, cu_seqlens_k, _, _ = unpad_input(
        rearrange(key, "b s h d -> b s (h d)"), key_padding_mask
    )
    value_unpad, _, _, _, _ = unpad_input(
        rearrange(value, "b s h d -> b s (h d)"), key_padding_mask
    )

    query_unpad = rearrange(query_unpad, "nnz (h d) -> nnz h d", h=nheads)
    key_unpad = rearrange(key_unpad, "nnz (h d) -> nnz h d", h=nheads)
    value_unpad = rearrange(value_unpad, "nnz (h d) -> nnz h d", h=nheads)

    output_unpad = flash_attn_varlen_func_v3(
        query_unpad,
        key_unpad,
        value_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_q,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic,
    )

    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"),
            indices,
            batch_size,
            seqlen,
        ),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    return output
