"""
Training data preprocessing script for GameFactory dataset.

This script preprocesses GameFactory/Minecraft dataset into the format required for training.
Reference: hyvideo/pipelines/worldplay_video_pipeline.py

Training dataset requirements (CameraJsonWMemDataset):
- Pose JSON: dense keys "0", "1", ..., "N-1" (N = number of video frames)
- Action JSON: each frame contains move_action and view_action
- Latent: [C_latent, T_latent, H_latent, W_latent]
- Temporal alignment: pose_keys[4*(i-1)+4] indexes the video frame corresponding to latent i
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from decord import VideoReader, cpu
from loguru import logger
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def load_vae_model(model_path: str, device: str = "cuda", dtype=None):
    """Load HunyuanVideo VAE model."""
    from hyvideo.commons import get_gpu_memory
    from hyvideo.models.autoencoders import hunyuanvideo_15_vae_w_cache

    logger.info(f"Loading VAE from {model_path}")
    vae_path = os.path.join(model_path, "vae")

    if dtype is None:
        memory_limitation = get_gpu_memory()
        GB = 1024 * 1024 * 1024
        if memory_limitation < 23 * GB:
            dtype = torch.float16
        else:
            dtype = torch.float32

    vae = hunyuanvideo_15_vae_w_cache.AutoencoderKLConv3D.from_pretrained(
        vae_path,
        torch_dtype=dtype,
    ).to(device)
    vae.eval()
    vae.requires_grad_(False)
    logger.info(f"VAE loaded successfully with dtype={dtype}")
    return vae


def load_text_encoder(model_path: str, device: str = "cuda"):
    """Load text encoder."""

    from hyvideo.models.text_encoders import PROMPT_TEMPLATE, TextEncoder

    logger.info("Loading text encoders...")

    # Load LLAMA text encoder (refer to _load_text_encoders)
    text_encoder_path = os.path.join(model_path, "text_encoder", "llm")
    if not os.path.exists(text_encoder_path):
        raise FileNotFoundError(
            f"{text_encoder_path} not found. Please check your model path."
        )

    text_encoder = TextEncoder(
        text_encoder_type="llm",
        tokenizer_type="llm",
        text_encoder_path=text_encoder_path,
        max_length=1000,
        text_encoder_precision="fp16",
        prompt_template=PROMPT_TEMPLATE["li-dit-encode-image-json"],
        prompt_template_video=PROMPT_TEMPLATE["li-dit-encode-video-json"],
        hidden_state_skip_layer=2,
        apply_final_norm=False,
        reproduce=False,
        logger=logger,
        device=device,
    )

    logger.info("Text encoder loaded successfully")

    return {"text_encoder": text_encoder}


def load_vision_encoder(model_path: str, device: str = "cuda"):
    """Load vision encoder (for i2v)."""

    from hyvideo.models.vision_encoder import VisionEncoder

    logger.info("Loading vision encoder...")

    vision_encoder_path = os.path.join(model_path, "vision_encoder", "siglip")
    if not os.path.exists(vision_encoder_path):
        raise FileNotFoundError(
            f"{vision_encoder_path} not found. Please check your model path."
        )

    vision_encoder = VisionEncoder(
        vision_encoder_type="siglip",
        vision_encoder_precision="fp16",
        vision_encoder_path=vision_encoder_path,
        processor_type=None,
        processor_path=None,
        output_key=None,
        logger=logger,
        device=device,
    )

    logger.info("Vision encoder loaded successfully")

    return {"vision_encoder": vision_encoder}


def load_byt5_encoder(
    model_path: str, device: str = "cuda", byt5_max_length: int = 256
):
    """Load byT5 encoder (refer to _load_byt5 in worldplay_video_pipeline.py)."""

    from hyvideo.models.text_encoders.byT5 import load_glyph_byT5_v2
    from hyvideo.models.text_encoders.byT5.format_prompt import MultilingualPromptFormat

    logger.info("Loading byT5 encoder...")

    glyph_root = os.path.join(model_path, "text_encoder", "Glyph-SDXL-v2")
    if not os.path.exists(glyph_root):
        logger.warning(
            f"Glyph checkpoint not found from '{glyph_root}'. Skipping byT5 loading."
        )
        return None

    byT5_google_path = os.path.join(model_path, "text_encoder", "byt5-small")
    if not os.path.exists(byT5_google_path):
        logger.warning(
            f"ByT5 google path not found from: {byT5_google_path}. Using 'google/byt5-small' from HuggingFace."
        )
        byT5_google_path = "google/byt5-small"

    multilingual_prompt_format_color_path = os.path.join(
        glyph_root, "assets/color_idx.json"
    )
    multilingual_prompt_format_font_path = os.path.join(
        glyph_root, "assets/multilingual_10-lang_idx.json"
    )

    byt5_args = dict(
        byT5_google_path=byT5_google_path,
        byT5_ckpt_path=os.path.join(glyph_root, "checkpoints/byt5_model.pt"),
        multilingual_prompt_format_color_path=multilingual_prompt_format_color_path,
        multilingual_prompt_format_font_path=multilingual_prompt_format_font_path,
        byt5_max_length=byt5_max_length,
    )

    byt5_kwargs = load_glyph_byT5_v2(byt5_args, device=device)
    prompt_format = MultilingualPromptFormat(
        font_path=multilingual_prompt_format_font_path,
        color_path=multilingual_prompt_format_color_path,
    )

    logger.info("byT5 encoder loaded successfully")

    return {
        "byt5_model": byt5_kwargs["byt5_model"],
        "byt5_tokenizer": byt5_kwargs["byt5_tokenizer"],
        "byt5_max_length": byt5_kwargs["byt5_max_length"],
        "prompt_format": prompt_format,
    }


def load_video_segment(
    video_path: str, start_frame: int, end_frame: int
) -> torch.Tensor:
    """
    Load video segment.

    Returns:
        frames: [T, H, W, C] uint8 tensor
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
    total_frames = len(vr)

    # Ensure frame indices are valid
    start_frame = max(0, start_frame)
    end_frame = min(total_frames - 1, end_frame)

    # Read specified frame range
    frame_indices = list(range(start_frame, end_frame + 1))
    frames = vr.get_batch(frame_indices).asnumpy()

    vr.seek(0)
    del vr

    return torch.from_numpy(frames.astype(np.uint8))


def resample_video_frames(
    video_frames: torch.Tensor,
    target_num_frames: Optional[int] = None,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Resample video frame sequence to target number of frames (index interpolation:
    compute source indices then convert to int to sample frames).

    - When target_num_frames != T: uniformly compute source index i*(T-1)/(target_num_frames-1),
      convert to int and take corresponding frame
    - When target_num_frames is None or equals T: return as-is

    Args:
        video_frames:      [T, H, W, C] uint8 tensor
        target_num_frames: target frame count; None means no resampling (return as-is)

    Returns:
        resampled_frames: [T', H, W, C] uint8 tensor
        source_indices:   original frame index (0-based int) for each output frame
    """
    T = video_frames.shape[0]
    if target_num_frames is None or target_num_frames == T:
        return video_frames, list(range(T))
    if target_num_frames <= 0:
        raise ValueError(
            f"target_num_frames must be a positive integer, got {target_num_frames}"
        )
    if target_num_frames == 1:
        return video_frames[:1], [0]
    # Uniformly distributed source indices (float), convert to int then sample frames
    source_indices: List[int] = [
        int(round(i * (T - 1) / (target_num_frames - 1)))
        for i in range(target_num_frames)
    ]
    # Ensure indices are within bounds
    source_indices = [min(i, T - 1) for i in source_indices]
    frames_out = video_frames[source_indices]
    return frames_out, source_indices


def encode_video_to_latent(
    vae: nn.Module,
    video_frames: torch.Tensor,
    target_height: int,
    target_width: int,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Encode video to latent representation.

    Reference VAE encoder logic:
    - Frame 0 is processed alone -> latent frame 0
    - Every 4 frames as a group -> latent frame 1, 2, ...
    - Required frame count: 1 + 4 * (L - 1) where L is number of latent frames

    Important: VAE requires spatial dimensions to be divisible by ffactor_spatial(16),
    and intermediate layers must be divisible by 2. Video must be resize+crop to
    target resolution first.

    Args:
        vae: VAE model
        video_frames: [T, H, W, C] uint8 tensor
        target_height: target height (e.g., 480), must be divisible by 16
        target_width: target width (e.g., 832), must be divisible by 16
        device: device

    Returns:
        latent: [1, C_latent, T_latent, H_latent, W_latent] float32 tensor
    """
    H, W = video_frames.shape[1], video_frames.shape[2]

    # Resize + CenterCrop to target resolution (consistent with inference pipeline)
    if H != target_height or W != target_width:
        scale_factor = max(target_width / W, target_height / H)
        resize_h = int(round(H * scale_factor))
        resize_w = int(round(W * scale_factor))

        # [T, H, W, C] → [T, C, H, W] for interpolate
        frames = video_frames.permute(0, 3, 1, 2).float()  # [T, C, H, W]
        frames = torch.nn.functional.interpolate(
            frames, size=(resize_h, resize_w), mode="bilinear", align_corners=False
        )

        # Center crop
        crop_top = (resize_h - target_height) // 2
        crop_left = (resize_w - target_width) // 2
        frames = frames[
            :,
            :,
            crop_top : crop_top + target_height,
            crop_left : crop_left + target_width,
        ]

        # Normalize to [-1, 1]
        video = frames / 127.5 - 1.0  # [T, C, H, W]
    else:
        video = video_frames.permute(0, 3, 1, 2).float() / 127.5 - 1.0  # [T, C, H, W]

    # Add batch dimension and reshape to [B, C, T, H, W]
    video = video.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, C, T, H, W]

    # VAE encoder handles frame count internally (no extra padding needed)
    # iter_ = 1 + (num_frame - 1) // 4

    vae_dtype = next(vae.parameters()).dtype
    video = video.to(device, dtype=vae_dtype)

    with torch.no_grad():
        latent = vae.encode(video).latent_dist.sample()
        # latent: [B, C_latent, T_latent, H_latent, W_latent]
        latent = latent * vae.config.scaling_factor

    return latent.cpu().float()


def encode_first_frame_to_latent(
    vae: nn.Module,
    first_frame: torch.Tensor,
    target_height: int,
    target_width: int,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Encode first frame to VAE latent (for image_cond).

    Reference inference flow get_image_condition_latents():
    1. Resize + CenterCrop to target resolution
    2. Normalize [0.5]
    3. VAE encode
    4. Multiply by scaling_factor

    Args:
        first_frame: [H, W, C] uint8 tensor
        target_height: target height (e.g., 480)
        target_width: target width (e.g., 832)

    Returns:
        image_cond: [1, C_latent, 1, H_latent, W_latent] float32 tensor
    """
    # Convert to PIL Image
    frame_np = first_frame.numpy()
    pil_image = Image.fromarray(frame_np)

    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resize_width = int(round(original_width * scale_factor))
    resize_height = int(round(original_height * scale_factor))

    # Transform consistent with inference pipeline
    ref_image_transform = transforms.Compose(
        [
            transforms.Resize(
                (resize_height, resize_width),
                interpolation=transforms.InterpolationMode.LANCZOS,
            ),
            transforms.CenterCrop((target_height, target_width)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    ref_images_pixel_values = ref_image_transform(pil_image)
    # [C, H, W] → [1, C, 1, H, W]
    ref_images_pixel_values = (
        ref_images_pixel_values.unsqueeze(0).unsqueeze(2).to(device)
    )

    vae_dtype = next(vae.parameters()).dtype
    ref_images_pixel_values = ref_images_pixel_values.to(dtype=vae_dtype)

    with torch.no_grad():
        # Use mode() instead of sample(), consistent with inference
        cond_latents = vae.encode(ref_images_pixel_values).latent_dist.mode()
        cond_latents = cond_latents * vae.config.scaling_factor

    return cond_latents.cpu().float()  # [1, C_latent, 1, H_latent, W_latent]


def encode_prompt(
    prompt: str,
    text_encoders: Dict,
    device: str = "cuda",
    max_length: int = 1000,
) -> Dict[str, torch.Tensor]:
    """
    Encode text prompt (refer to encode_prompt in worldplay_video_pipeline.py).

    Returns:
        dict with keys:
            - prompt_embeds: [1, seq_len, dim]
            - prompt_mask: [1, seq_len]
    """
    text_encoder = text_encoders["text_encoder"]

    with torch.no_grad():
        # Use TextEncoder API
        text_inputs = text_encoder.text2tokens(
            prompt, data_type="video", max_length=max_length
        )

        prompt_outputs = text_encoder.encode(
            text_inputs, data_type="video", device=device
        )

        prompt_embeds = prompt_outputs.hidden_state
        prompt_mask = prompt_outputs.attention_mask

        if prompt_mask is not None:
            prompt_mask = prompt_mask.to(device)

    return {
        "prompt_embeds": prompt_embeds.cpu(),
        "prompt_mask": prompt_mask.cpu(),
    }


def encode_first_frame(
    first_frame: torch.Tensor,
    vision_encoder_dict: Dict,
    target_height: int,
    target_width: int,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Encode first frame visual features (for i2v vision_states).

    Reference inference pipeline _prepare_vision_states():
    1. resize_and_center_crop to target resolution
    2. VisionEncoder.encode_images(numpy_array)

    Args:
        first_frame: [H, W, C] uint8 tensor
        target_height: target height
        target_width: target width

    Returns:
        dict with keys:
            - vision_states: [1, seq_len, dim]
    """
    vision_encoder = vision_encoder_dict["vision_encoder"]

    # Resize and center crop to target resolution (consistent with inference)
    frame_np = first_frame.numpy()  # [H, W, C] uint8
    pil_image = Image.fromarray(frame_np)
    original_width, original_height = pil_image.size

    scale_factor = max(target_width / original_width, target_height / original_height)
    resize_width = int(round(original_width * scale_factor))
    resize_height = int(round(original_height * scale_factor))

    # Resize
    pil_image = pil_image.resize((resize_width, resize_height), Image.LANCZOS)
    # Center crop
    left = (resize_width - target_width) // 2
    top = (resize_height - target_height) // 2
    pil_image = pil_image.crop((left, top, left + target_width, top + target_height))

    input_image_np = np.array(pil_image)

    with torch.no_grad():
        # Use VisionEncoder encode_images method
        vision_outputs = vision_encoder.encode_images(input_image_np)
        vision_states = vision_outputs.last_hidden_state  # [1, seq_len, dim]

    return {
        "vision_states": vision_states.cpu(),
    }


def encode_byt5_prompt(
    prompt: str,
    byt5_dict: Dict,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Encode byT5 text prompt (refer to _prepare_byt5_embeddings in worldplay_video_pipeline.py).

    Returns:
        dict with keys:
            - byt5_text_states: [1, seq_len, dim]
            - byt5_text_mask: [1, seq_len]
    """
    if byt5_dict is None:
        # If byT5 is not loaded, return zero tensors
        logger.warning("byT5 not loaded, using zero tensors")
        return {
            "byt5_text_states": torch.zeros(1, 256, 1472),
            "byt5_text_mask": torch.zeros(1, 256, dtype=torch.int64),
        }

    byt5_model = byt5_dict["byt5_model"]
    byt5_tokenizer = byt5_dict["byt5_tokenizer"]
    byt5_max_length = byt5_dict["byt5_max_length"]

    # Extract text inside quotes (if any)
    import re

    pattern = r'"(.*?)"|"(.*?)"'
    matches = re.findall(pattern, prompt)
    glyph_texts = [match[0] or match[1] for match in matches]

    if len(glyph_texts) == 0:
        # No quoted text, return zero tensors
        return {
            "byt5_text_states": torch.zeros(1, byt5_max_length, 1472).to(device),
            "byt5_text_mask": torch.zeros(1, byt5_max_length, dtype=torch.int64).to(
                device
            ),
        }

    # Format text
    prompt_format = byt5_dict["prompt_format"]
    text_styles = [
        {"color": None, "font-family": None} for _ in range(len(glyph_texts))
    ]
    formatted_text = prompt_format.format_prompt(glyph_texts, text_styles)

    # Tokenize
    byt5_text_inputs = byt5_tokenizer(
        formatted_text,
        padding="max_length",
        max_length=byt5_max_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )

    text_ids = byt5_text_inputs.input_ids.to(device)
    text_mask = byt5_text_inputs.attention_mask.to(device)

    with torch.no_grad():
        byt5_outputs = byt5_model(text_ids, attention_mask=text_mask.float())
        byt5_embeddings = byt5_outputs[0]

    return {
        "byt5_text_states": byt5_embeddings.cpu(),
        "byt5_text_mask": text_mask.cpu(),
    }


def _pose_from_action_data(action_data: Dict) -> np.ndarray:
    """Compute W2C matrix (4x4 float32) from single-frame GameFactory action_data."""
    pos = action_data.get("pos", [0.0, 0.0, 0.0])
    pitch = action_data.get("pre_pitch", 0.0)
    yaw = action_data.get("pre_yaw", 0.0)

    pitch_rad, yaw_rad = np.deg2rad(pitch), np.deg2rad(yaw)
    cos_pitch, sin_pitch = np.cos(pitch_rad), np.sin(pitch_rad)
    cos_yaw, sin_yaw = np.cos(yaw_rad), np.sin(yaw_rad)

    R = np.array(
        [
            [cos_yaw, -sin_yaw * cos_pitch, sin_yaw * sin_pitch],
            [sin_yaw, cos_yaw * cos_pitch, -cos_yaw * sin_pitch],
            [0.0, sin_pitch, cos_pitch],
        ],
        dtype=np.float32,
    )
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R
    c2w[:3, 3] = pos
    return np.linalg.inv(c2w)


def _action_from_action_data(action_data: Dict) -> Tuple[str, str]:
    """Extract (move_action, view_action) strings from single-frame GameFactory action_data."""
    ws = action_data.get("ws", 1)  # 0=S, 1=none, 2=W
    ad = action_data.get("ad", 0)  # 0=none, 1=A, 2=D

    move_action = ""
    if ws == 2:
        move_action += "W"
    elif ws == 0:
        move_action += "S"
    if ad == 2:
        move_action += "D"
    elif ad == 1:
        move_action += "A"

    pitch_delta = action_data.get("pitch_delta", 0.0)
    yaw_delta = action_data.get("yaw_delta", 0.0)

    view_action = ""
    if abs(yaw_delta) > 0.1:
        if yaw_delta > 0:
            view_action = "LL"  # turn left
        else:
            view_action = "LR"  # turn right
    elif abs(pitch_delta) > 0.1:
        if pitch_delta > 0:
            view_action = "LD"  # look down
        else:
            view_action = "LU"  # look up

    return move_action, view_action


def convert_gamefactory_actions_to_pose_and_actions(
    metadata: Dict,
    start_frame: int,
    end_frame: int,
    target_height: int = 480,
    target_width: int = 832,
    source_frame_indices: Optional[List[float]] = None,
) -> Tuple[Dict, Dict]:
    """
    Convert GameFactory metadata to pose and action format required for training.

    Important: CameraJsonWMemDataset uses pose_keys[4*(i-1)+4] for positional indexing,
    so pose_json and action_json must contain entries for **every output video frame**,
    with keys "0", "1", "2", ..., "num_output_frames-1".

    Args:
        metadata:             GameFactory metadata dict
        start_frame:          start frame of original video (for indexing metadata)
        end_frame:            end frame of original video (for indexing metadata)
        target_height:        target height
        target_width:         target width
        source_frame_indices: frame index (0-based int) in original segment for each
                              output frame (from resample_video_frames()). None means
                              one-to-one mapping (default).

    Returns:
        pose_dict:   {"0": {"w2c": ..., "intrinsic": ...}, ...}
        action_dict: {"0": {"move_action": ..., "view_action": ...}, ...}
    """
    actions = metadata.get("actions", {})
    original_total_frames = end_frame - start_frame + 1

    # Intrinsic matrix (FOV = 60 deg, unnormalized; training code normalizes on load)
    focal_length = target_width / (2.0 * np.tan(np.deg2rad(60.0) / 2.0))
    intrinsic = [
        [focal_length, 0.0, target_width / 2.0],
        [0.0, focal_length, target_height / 2.0],
        [0.0, 0.0, 1.0],
    ]

    # (output frame index, frame index int within original segment)
    if source_frame_indices is None:
        iter_pairs = [(i, i) for i in range(original_total_frames)]
    else:
        iter_pairs = list(enumerate(source_frame_indices))

    pose_dict: Dict = {}
    action_dict: Dict = {}

    for out_idx, frame_offset in iter_pairs:
        # Integer index, look up metadata directly (no interpolation)
        frame_offset = int(frame_offset)
        frame_offset = min(max(0, frame_offset), original_total_frames - 1)
        frame_key = str(start_frame + frame_offset)
        action_data = actions.get(frame_key)

        if action_data is not None:
            w2c = _pose_from_action_data(action_data)
            move_action, view_action = _action_from_action_data(action_data)
        else:
            w2c = np.eye(4, dtype=np.float32)
            move_action = ""
            view_action = ""

        pose_dict[str(out_idx)] = {
            "w2c": w2c.tolist(),
            "intrinsic": intrinsic,
        }
        action_dict[str(out_idx)] = {
            "move_action": move_action,
            "view_action": view_action,
        }

    return pose_dict, action_dict


def load_annotation_csv(csv_path: str) -> List[Dict]:
    """
    Load annotation.csv file.

    Returns:
        List of dicts with keys: video_name, start_frame, end_frame, prompt
    """
    annotations = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_name = row["original video name"].strip()
            start_frame = int(row["start frame index"])
            end_frame = int(row["end frame index"])
            prompt = row["prompt"].strip()

            annotations.append(
                {
                    "video_name": video_name,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "prompt": prompt,
                }
            )
    return annotations


def preprocess_single_segment(
    video_path: str,
    metadata_path: str,
    start_frame: int,
    end_frame: int,
    prompt: str,
    vae: nn.Module,
    text_encoders: Dict,
    vision_encoders: Dict,
    byt5_encoders: Dict,
    output_dir: str,
    segment_id: str,
    target_height: int = 480,
    target_width: int = 832,
    device: str = "cuda",
    target_num_frames: Optional[int] = None,
) -> Dict[str, str]:
    """
    Preprocess a single video segment.

    Args (new):
        target_num_frames: resample segment to this frame count (None = unchanged).
            Index interpolation: uniform source indices -> int -> sample.

    Returns:
        dict with paths to saved files
    """
    # Create output directory
    segment_output_dir = os.path.join(output_dir, segment_id)
    os.makedirs(segment_output_dir, exist_ok=True)

    # 1. Load video segment
    logger.info(f"Loading video segment: {video_path} [{start_frame}:{end_frame}]")
    video_frames = load_video_segment(video_path, start_frame, end_frame)
    original_num_frames = video_frames.shape[0]

    # Optional resampling (index interpolation: uniform source indices -> int -> sample)
    video_frames, source_frame_indices = resample_video_frames(
        video_frames, target_num_frames=target_num_frames
    )
    if target_num_frames is not None:
        logger.info(
            f"Resampled: {original_num_frames} -> {video_frames.shape[0]} frames "
            f"(target_num_frames={target_num_frames})"
        )

    num_video_frames = video_frames.shape[0]
    # VAE temporal structure: frame 0 → latent 0, frames 1..4 → latent 1, etc.
    num_latent_frames = 1 + (num_video_frames - 1) // 4
    logger.info(
        f"Video frames: {num_video_frames} -> Latent frames: {num_latent_frames}"
    )

    # 2. Encode video to latent
    logger.info("Encoding video to latent...")
    # latent: [1, C, T, H, W] - keep batch dim consistent with training code
    latent = encode_video_to_latent(
        vae,
        video_frames,
        target_height=target_height,
        target_width=target_width,
        device=device,
    )

    # 3. Encode text prompt
    logger.info(f"Encoding prompt: {prompt[:50]}...")
    prompt_embeds_dict = encode_prompt(prompt, text_encoders, device=device)

    # 3.5 Encode byT5 prompt
    logger.info("Encoding byT5 prompt...")
    byt5_embeds_dict = encode_byt5_prompt(prompt, byt5_encoders, device=device)

    # 4. Encode first frame image_cond (VAE latent) and vision_states (for i2v)
    logger.info("Encoding first frame for i2v (image_cond + vision_states)")
    image_cond = encode_first_frame_to_latent(
        vae, video_frames[0], target_height, target_width, device=device
    )
    vision_states_dict = encode_first_frame(
        video_frames[0], vision_encoders, target_height, target_width, device=device
    )

    # 5. Process pose and action data
    # Important: every video frame must have an entry (dense keys "0", "1", ..., "N-1")
    logger.info("Processing pose and action data from metadata")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    pose_dict, action_dict = convert_gamefactory_actions_to_pose_and_actions(
        metadata,
        start_frame,
        end_frame,
        target_height=target_height,
        target_width=target_width,
        source_frame_indices=source_frame_indices,
    )

    # 6. Save data
    pose_save_path = os.path.join(segment_output_dir, f"{segment_id}_pose.json")
    with open(pose_save_path, "w") as f:
        json.dump(pose_dict, f, indent=2)

    action_save_path = os.path.join(segment_output_dir, f"{segment_id}_action.json")
    with open(action_save_path, "w") as f:
        json.dump(action_dict, f, indent=2)

    latent_save_path = os.path.join(segment_output_dir, f"{segment_id}_latent.pt")
    logger.info(f"Saving latent to: {latent_save_path}")

    save_dict = {
        "latent": latent,  # [1, C_latent, T_latent, H_latent, W_latent]
        "prompt_embeds": prompt_embeds_dict["prompt_embeds"],  # [1, seq_len, dim]
        "prompt_mask": prompt_embeds_dict["prompt_mask"],  # [1, seq_len]
        "byt5_text_states": byt5_embeds_dict["byt5_text_states"],  # [1, byt5_len, 1472]
        "byt5_text_mask": byt5_embeds_dict["byt5_text_mask"],  # [1, byt5_len]
        "image_cond": image_cond,  # [1, C_latent, 1, H_latent, W_latent]
        "vision_states": vision_states_dict["vision_states"],  # [1, seq_len, dim]
    }

    torch.save(save_dict, latent_save_path)

    return {
        "latent_path": latent_save_path,
        "pose_path": pose_save_path,
        "action_path": action_save_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess GameFactory/Minecraft dataset"
    )

    # Input paths
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Dataset root directory (contains annotation.csv, metadata/, video/)",
    )

    # Output paths
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="dataset_index.json",
        help="Output index filename",
    )

    # Model path
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to HunyuanVideo model"
    )

    # Target resolution
    parser.add_argument(
        "--target_height", type=int, default=480, help="Target height (default: 480)"
    )
    parser.add_argument(
        "--target_width", type=int, default=832, help="Target width (default: 832)"
    )

    # Other options
    parser.add_argument("--device", type=str, default="cuda", help="Compute device")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to process (for testing)",
    )

    parser.add_argument(
        "--target_num_frames",
        type=int,
        default=None,
        help="Resample each clip to this frame count (index interpolation: uniform indices -> int). None = unchanged.",
    )

    args = parser.parse_args()

    # Validate dataset structure
    annotation_csv = os.path.join(args.data_root, "annotation.csv")
    metadata_dir = os.path.join(args.data_root, "metadata")
    video_dir = os.path.join(args.data_root, "video")

    if not os.path.exists(annotation_csv):
        logger.error(f"annotation.csv not found at {annotation_csv}")
        sys.exit(1)

    if not os.path.exists(metadata_dir):
        logger.error(f"metadata directory not found at {metadata_dir}")
        sys.exit(1)

    if not os.path.exists(video_dir):
        logger.error(f"video directory not found at {video_dir}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    logger.info("=" * 50)
    logger.info("Loading models...")
    logger.info("=" * 50)

    vae = load_vae_model(args.model_path, device=args.device)
    text_encoders = load_text_encoder(args.model_path, device=args.device)
    vision_encoders = load_vision_encoder(args.model_path, device=args.device)
    byt5_encoders = load_byt5_encoder(args.model_path, device=args.device)

    # Load annotation.csv
    logger.info("=" * 50)
    logger.info(f"Loading annotations from {annotation_csv}")
    logger.info("=" * 50)

    annotations = load_annotation_csv(annotation_csv)

    if args.num_samples:
        annotations = annotations[: args.num_samples]

    logger.info(f"Found {len(annotations)} segments to process")

    # Preprocess each segment
    logger.info("=" * 50)
    logger.info("Processing segments...")
    logger.info("=" * 50)

    dataset_index = []
    output_json_path = os.path.join(args.output_dir, args.output_json)

    for ann_idx, annotation in enumerate(tqdm(annotations, desc="Processing segments")):
        try:
            video_name = annotation["video_name"]
            start_frame = annotation["start_frame"]
            end_frame = annotation["end_frame"]
            prompt = annotation["prompt"]

            # Generate unique segment ID
            segment_id = f"{Path(video_name).stem}_{start_frame}_{end_frame}"

            # Build actual filename (refer to game_factory_dataset.py)
            video_idx = int(video_name.split("_")[-2])
            stem = os.path.splitext(video_name)[0]
            stem = stem + f"_{video_idx}"

            # Video path
            video_path = os.path.join(video_dir, stem + ".mp4")
            if not os.path.exists(video_path):
                logger.warning(f"Video not found: {video_path}, skipping")
                continue

            # Metadata path
            metadata_path = os.path.join(metadata_dir, f"{stem}.json")
            if not os.path.exists(metadata_path):
                logger.warning(f"Metadata not found: {metadata_path}, skipping")
                continue

            # Preprocess segment
            result = preprocess_single_segment(
                video_path=video_path,
                metadata_path=metadata_path,
                start_frame=start_frame,
                end_frame=end_frame,
                prompt=prompt,
                vae=vae,
                text_encoders=text_encoders,
                vision_encoders=vision_encoders,
                byt5_encoders=byt5_encoders,
                output_dir=args.output_dir,
                segment_id=segment_id,
                target_height=args.target_height,
                target_width=args.target_width,
                device=args.device,
                target_num_frames=args.target_num_frames,
            )

            # Add to index
            dataset_index.append(
                {
                    "segment_id": segment_id,
                    "video_name": video_name,
                    "video_path": video_path,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "latent_path": result["latent_path"],
                    "pose_path": result["pose_path"],
                    "action_path": result["action_path"],
                    "prompt": prompt,
                }
            )
            with open(output_json_path, "w") as f:
                json.dump(dataset_index, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to process segment {ann_idx}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Save index file
    logger.info("=" * 50)
    logger.info(f"Saving dataset index to {output_json_path}")
    logger.info("=" * 50)

    with open(output_json_path, "w") as f:
        json.dump(dataset_index, f, indent=2)

    logger.info(f"Successfully processed {len(dataset_index)} segments")
    logger.info(f"Dataset index saved to: {output_json_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
