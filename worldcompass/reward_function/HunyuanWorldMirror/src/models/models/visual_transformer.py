import logging
import random
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from reward_function.HunyuanWorldMirror.src.models.layers import (
    PatchEmbed,
    PatchEmbed_Mlp,
)
from reward_function.HunyuanWorldMirror.src.models.layers.block import Block
from reward_function.HunyuanWorldMirror.src.models.layers.rope import (
    RotaryPositionEmbedding2D,
    PositionGetter,
)
from reward_function.HunyuanWorldMirror.src.models.layers.vision_transformer import (
    vit_small,
    vit_base,
    vit_large,
    vit_giant2,
)

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class VisualGeometryTransformer(nn.Module):
    """
    The VisualGeometryTransformer applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.

    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
        enable_condition (bool): Whether to enable conditioning inputs.
        sampling_strategy (str): Sampling strategy for patches.
        fixed_patch_embed (bool): Whether to fix patch embedding weights.
        condition_strategy (list[str]): Strategy for each conditioning input.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        enable_cond=False,
        sampling_strategy="uniform",
        fixed_patch_embed=False,
        condition_strategy=["token", "pow3r", "token"],
        intermediate_idxs: List[int] = [4, 11, 17, 23],
    ):
        super().__init__()
        # Store config parameters
        self.enable_cond = enable_cond
        self.sampling_strategy = sampling_strategy
        self.cond_methods = condition_strategy
        self.intermediate_idxs = intermediate_idxs
        self.depth = depth
        self.patch_size = patch_size

        # Initialize patch embedding module
        self.patch_embed = self._init_patch_embedding_module(
            patch_embed,
            img_size,
            patch_size,
            num_register_tokens,
            embed_dim=embed_dim,
            is_fixed=fixed_patch_embed,
        )

        # Initialize conditioning embeddings if enabled
        if self.enable_cond:
            self._init_cond_embeddings(
                embed_dim, img_size, patch_size, num_register_tokens
            )

        # Initialize rotary position embedding
        self._init_rotary_position_embedding(rope_freq)

        # Initialize transformer blocks
        self._init_transformer_blocks(
            block_fn,
            embed_dim,
            num_heads,
            mlp_ratio,
            qkv_bias,
            proj_bias,
            ffn_bias,
            init_values,
            qk_norm,
        )

        # Initialize learnable tokens
        self._init_learnable_tokens(embed_dim, num_register_tokens)

        # Calculate patch start index based on conditioning
        if self.enable_cond:
            self.patch_start_idx = (
                1 + num_register_tokens + 1 + 1
            )  # camera + register + pose + rays
        else:
            self.patch_start_idx = 1 + num_register_tokens  # camera + register

        # Register normalization constants
        for name, value in (
            ("_resnet_mean", _RESNET_MEAN),
            ("_resnet_std", _RESNET_STD),
        ):
            self.register_buffer(
                name,
                torch.FloatTensor(value).reshape(1, 1, 3, 1, 1),
                persistent=False,
            )

        self.use_reentrant = False

    def _init_patch_embedding_module(
        self,
        patch_embed_type,
        img_size,
        patch_size,
        num_reg_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
        is_fixed=False,
        in_chans=3,
    ):
        """Create the patch embedding module.

        If 'conv', we use a simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """
        if "conv" in patch_embed_type:
            if "mlp" in patch_embed_type:
                patch_embed_module = PatchEmbed_Mlp(
                    img_size=img_size,
                    patch_size=patch_size,
                    in_chans=in_chans,
                    embed_dim=embed_dim,
                )
            else:
                patch_embed_module = PatchEmbed(
                    img_size=img_size,
                    patch_size=patch_size,
                    in_chans=in_chans,
                    embed_dim=embed_dim,
                )
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            patch_embed_module = vit_models[patch_embed_type](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_reg_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(patch_embed_module, "mask_token"):
                patch_embed_module.mask_token.requires_grad_(False)

        if is_fixed:
            for param in patch_embed_module.parameters():
                param.requires_grad_(False)

        return patch_embed_module

    def _init_cond_embeddings(
        self, embed_dim, img_size, patch_size, num_reg_tokens
    ):
        """Initialize conditioning embeddings for camera, depth, and rays."""
        assert self.cond_methods is not None
        assert self.cond_methods[0] == "token"

        # Camera pose embedding
        if self.cond_methods[0] == "token":
            self.pose_embed = nn.Sequential(
                nn.Linear(7, embed_dim, bias=True),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=True),
            )
        else:
            raise NotImplementedError

        # Depth map embedding
        if self.cond_methods[1] == "pow3r":
            self.depth_embed = self._init_patch_embedding_module(
                "conv+mlp",
                img_size,
                patch_size,
                num_reg_tokens,
                embed_dim=embed_dim,
                in_chans=1,
            )
        else:
            raise NotImplementedError

        # Ray direction embedding
        if self.cond_methods[2] == "token":
            self.ray_embed = nn.Sequential(
                nn.Linear(4, embed_dim, bias=True),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=True),
            )
        else:
            raise NotImplementedError

    def _init_rotary_position_embedding(self, rope_freq):
        self.rope = (
            RotaryPositionEmbedding2D(
                frequency=rope_freq,
            )
            if rope_freq > 0
            else None
        )
        self.pos_getter = PositionGetter() if self.rope is not None else None

    def _init_transformer_blocks(
        self,
        block_fn,
        embed_dim,
        num_heads,
        mlp_ratio,
        qkv_bias,
        proj_bias,
        ffn_bias,
        init_values,
        qk_norm,
    ):
        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(self.depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(self.depth)
            ]
        )

    def _init_learnable_tokens(self, embed_dim, num_reg_tokens):
        """Initialize learnable tokens."""
        self.cam_token = nn.Parameter(torch.zeros(1, 2, 1, embed_dim))
        self.reg_token = nn.Parameter(
            torch.zeros(1, 2, num_reg_tokens, embed_dim)
        )
        nn.init.normal_(self.cam_token, std=1e-6)
        nn.init.normal_(self.reg_token, std=1e-6)

    def forward(
        self,
        images: torch.Tensor,
        priors: List | None = None,
        cond_flags: List[int] = [0, 0, 0],
        ctx_frames: int = None,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images: Input images with shape [B, S, 3, H, W], in range [0, 1]
            priors: Optional tuple of (depth, rays, poses) for conditioning
            cond_flags: List indicating which conditions to use [pose, depth, rays]
            ctx_frames: Number of context frames to use

        Returns:
            (list[torch.Tensor], int): List of attention block outputs and patch_start_idx
        """
        depth_maps, ray_dirs, poses = (
            priors if priors is not None else (None, None, None)
        )

        # Slice to context frames if specified
        if ctx_frames is not None:
            for var_name in ["images", "depth_maps", "ray_dirs", "poses"]:
                var = locals()[var_name]
                if var is not None:
                    locals()[var_name] = var[:, :ctx_frames].clone()

        # Process image tokens
        b, seq_len, ch, h, w = images.shape
        if ch != 3:
            raise ValueError(f"Expected 3 input channels, got {ch}")

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            images = (images - self._resnet_mean) / self._resnet_std
            images = images.reshape(b * seq_len, ch, h, w)
            patch_tokens = self.patch_embed(images)
            if isinstance(patch_tokens, dict):
                patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, patch_count, embed_dim = patch_tokens.shape

        # Prepare special tokens
        cam_tokens = expand_and_flatten_special_tokens(
            self.cam_token, b, seq_len
        )
        reg_tokens = expand_and_flatten_special_tokens(
            self.reg_token, b, seq_len
        )

        # Process all tokens (optional conditioning)
        if self.enable_cond:
            pose_tokens, depth_tokens, ray_tokens = self._process_conditioning(
                depth_maps,
                ray_dirs,
                poses,
                b,
                seq_len,
                patch_count,
                embed_dim,
                images,
                cond_flags,
            )
            # Add condition tokens to patch tokens
            patch_tokens = patch_tokens + depth_tokens
            all_tokens = torch.cat(
                [cam_tokens, reg_tokens, pose_tokens, ray_tokens, patch_tokens],
                dim=1,
            )
        else:
            all_tokens = torch.cat(
                [cam_tokens, reg_tokens, patch_tokens], dim=1
            )

        _, patch_count, embed_dim = all_tokens.shape

        # Position embedding
        pos_emb = None
        if self.rope is not None:
            pos_emb = self.pos_getter(
                b * seq_len,
                h // self.patch_size,
                w // self.patch_size,
                device=images.device,
            )
            if self.patch_start_idx > 0:
                pos_emb = pos_emb + 1
                special_pos = torch.zeros(
                    b * seq_len,
                    self.patch_start_idx,
                    2,
                    device=images.device,
                    dtype=pos_emb.dtype,
                )
                pos_emb = torch.cat([special_pos, pos_emb], dim=1)

        # Forward through attention blocks
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = []
            global_tokens = None
            for idx in range(self.depth):
                local_tokens = self._process_attention_blocks(
                    tokens=(
                        all_tokens if global_tokens is None else global_tokens
                    ),
                    b=b,
                    seq_len=seq_len,
                    patch_count=patch_count,
                    embed_dim=embed_dim,
                    block_idx=idx,
                    blocks=self.frame_blocks,
                    block_type="frame",
                    pos=pos_emb,
                )
                global_tokens = self._process_attention_blocks(
                    tokens=local_tokens,
                    b=b,
                    seq_len=seq_len,
                    patch_count=patch_count,
                    embed_dim=embed_dim,
                    block_idx=idx,
                    blocks=self.global_blocks,
                    block_type="global",
                    pos=pos_emb,
                )

                # Combine frame and global intermediates
                if idx in self.intermediate_idxs:
                    combined_out = torch.cat(
                        [local_tokens, global_tokens], dim=-1
                    )
                    outputs.append(combined_out)

        return outputs, self.patch_start_idx

    def _process_conditioning(
        self,
        depth_maps,
        ray_dirs,
        poses,
        b,
        seq_len,
        patch_count,
        embed_dim,
        images,
        cond_flags,
    ):
        """Process conditioning inputs."""
        h, w = images.shape[-2:]

        # Process camera pose embedding
        use_poses = cond_flags[0] == 1 and poses is not None
        if use_poses:
            poses = poses.reshape(b * seq_len, -1)
            pose_tokens = self.pose_embed(poses).unsqueeze(1)
        else:
            pose_tokens = torch.zeros(
                (b * seq_len, 1, embed_dim),
                device=images.device,
                dtype=images.dtype,
            )

        # Process depth map embedding
        use_depth = cond_flags[1] == 1 and depth_maps is not None
        if use_depth:
            depth_maps = depth_maps.reshape(b * seq_len, 1, h, w)
            depth_tokens = self.depth_embed(depth_maps).reshape(
                b * seq_len, patch_count, embed_dim
            )
        else:
            depth_tokens = torch.zeros(
                (b * seq_len, patch_count, embed_dim),
                device=images.device,
                dtype=images.dtype,
            )

        # Process ray direction embedding
        use_rays = cond_flags[2] == 1 and ray_dirs is not None
        if use_rays:
            ray_dirs = ray_dirs.reshape(b * seq_len, -1)
            ray_tokens = self.ray_embed(ray_dirs).unsqueeze(1)
        else:
            ray_tokens = torch.zeros(
                (b * seq_len, 1, embed_dim),
                device=images.device,
                dtype=images.dtype,
            )

        return pose_tokens, depth_tokens, ray_tokens

    def _process_attention_blocks(
        self,
        tokens,
        b,
        seq_len,
        patch_count,
        embed_dim,
        block_idx,
        blocks,
        block_type,
        pos=None,
    ):
        """Process attention blocks with tokens in shape (B*S, P, C)."""
        token_shape = (b, seq_len, patch_count, embed_dim)
        if block_type == "frame":  # local
            target_shape = (b * seq_len, patch_count, embed_dim)
            pos_target_shape = (
                (b * seq_len, patch_count, 2) if pos is not None else None
            )
        else:  # global
            target_shape = (b, seq_len * patch_count, embed_dim)
            pos_target_shape = (
                (b, seq_len * patch_count, 2) if pos is not None else None
            )

        if tokens.shape != target_shape:
            tokens = tokens.reshape(*target_shape)

        if pos is not None and pos.shape != pos_target_shape:
            pos = pos.reshape(*pos_target_shape)

        if self.training:
            tokens = checkpoint(
                blocks[block_idx],
                tokens,
                pos=pos,
                use_reentrant=self.use_reentrant,
            )
        else:
            tokens = blocks[block_idx](tokens, pos=pos)

        return tokens.reshape(*token_shape)


def expand_and_flatten_special_tokens(token_tensor, b, seq_len):
    """Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing. Uses first
    position for frame 0, second position for remaining frames.

    Args:
        token_tensor: Input tensor with shape (1, 2, X, C)
        b: Batch size
        seq_len: Sequence length

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """
    # First frame uses position 0, remaining frames use position 1
    first_frame_tokens = token_tensor[:, 0:1, ...].expand(
        b, 1, *token_tensor.shape[2:]
    )
    remaining_frame_tokens = token_tensor[:, 1:, ...].expand(
        b, seq_len - 1, *token_tensor.shape[2:]
    )

    # Concatenate and flatten
    combined_tokens = torch.cat(
        [first_frame_tokens, remaining_frame_tokens], dim=1
    )
    return combined_tokens.reshape(b * seq_len, *combined_tokens.shape[2:])
