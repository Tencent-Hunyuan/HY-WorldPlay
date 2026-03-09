"""Prepare image and text latents for training.

This script extracts image and text features using HunyuanVideo encoders
and saves them as .pt files for training.

Usage:
    # Single GPU
    python dataset/prepare_image_text_latent_simple.py \
        --input_json /path/to/input.json \
        --output_dir /path/to/output \
        --hunyuan_checkpoint_path /path/to/hunyuanvideo_1_5

    # Multi-GPU
    torchrun --nproc_per_node=8 dataset/prepare_image_text_latent_simple.py \
        --input_json /path/to/input.json \
        --output_dir /path/to/output \
        --hunyuan_checkpoint_path /path/to/hunyuanvideo_1_5

Input JSON format:
    [
        {"image_path": "/path/to/image1.jpg", "caption": "A sunset"},
        {"image_path": "/path/to/image2.png", "caption": "A cat"}
    ]

Output:
    - {output_dir}/latents/{item_id}.pt
    - {output_dir}/latents.json (for --json_path in training)
"""

import os
import sys

sys.path.append(os.path.abspath("."))

import argparse
import json
import re
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from fastvideo.models.hyvideo.models.autoencoders import (
    hunyuanvideo_15_vae_w_cache,
)
from fastvideo.models.hyvideo.models.vision_encoder import VisionEncoder
from fastvideo.models.hyvideo.models.text_encoders import (
    PROMPT_TEMPLATE,
    TextEncoder,
)
from fastvideo.models.hyvideo.models.text_encoders.byT5 import (
    load_glyph_byT5_v2,
)
from fastvideo.models.hyvideo.models.text_encoders.byT5.format_prompt import (
    MultilingualPromptFormat,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Input JSON file with image_path and caption fields",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for latent files",
    )
    parser.add_argument(
        "--hunyuan_checkpoint_path",
        type=str,
        required=True,
        help="Path to HunyuanVideo checkpoint (contains vae/, text_encoder/, vision_encoder/)",
    )
    parser.add_argument("--target_height", type=int, default=480)
    parser.add_argument("--target_width", type=int, default=832)
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip if output file already exists",
    )
    return parser.parse_args()


class LatentExtractor:
    """Extract latents from images and text using HunyuanVideo encoders.

    This class loads and manages all encoder models:
    - VAE: Encodes images to latent space
    - Vision Encoder (SigLIP): Extracts visual features for conditioning
    - Text Encoder (LLM): Encodes text prompts
    - byT5: Encodes glyph/text features
    """

    def __init__(self, checkpoint_path, device, target_size=(480, 832)):
        self.device = device
        self.target_size = target_size  # (height, width)
        self._load_models(checkpoint_path)

    def _load_models(self, checkpoint_path):
        """Load all encoder models from checkpoint directory."""
        print(f"Loading models from {checkpoint_path}...")

        # ===== VAE: Variational Autoencoder for image-to-latent encoding =====
        self.vae = (
            hunyuanvideo_15_vae_w_cache.AutoencoderKLConv3D.from_pretrained(
                os.path.join(checkpoint_path, "vae"), torch_dtype=torch.float32
            )
            .to(self.device)
            .eval()
        )

        # ===== Vision Encoder: SigLIP for extracting visual features =====
        self.vision_encoder = VisionEncoder(
            vision_encoder_type="siglip",
            vision_encoder_precision="fp16",
            vision_encoder_path=os.path.join(
                checkpoint_path, "vision_encoder/siglip"
            ),
            processor_type=None,
            processor_path=None,
            output_key=None,
            logger=None,
            device=self.device,
        )

        # ===== Text Encoder: LLM-based text encoder for prompt embeddings =====
        self.text_encoder = TextEncoder(
            text_encoder_type="llm",
            tokenizer_type="llm",
            text_encoder_path=os.path.join(checkpoint_path, "text_encoder/llm"),
            max_length=1000,
            text_encoder_precision="fp16",
            prompt_template=PROMPT_TEMPLATE["li-dit-encode-image-json"],
            prompt_template_video=PROMPT_TEMPLATE["li-dit-encode-video-json"],
            hidden_state_skip_layer=2,
            apply_final_norm=False,
            reproduce=False,
            logger=None,
            device=self.device,
        )
        self.text_len = self.text_encoder.max_length

        # ===== byT5: Byte-level T5 for glyph/text encoding =====
        load_from = os.path.join(checkpoint_path, "text_encoder")
        glyph_root = os.path.join(load_from, "Glyph-SDXL-v2")
        byt5_args = dict(
            byT5_google_path=os.path.join(load_from, "byt5-small"),
            byT5_ckpt_path=os.path.join(
                glyph_root, "checkpoints/byt5_model.pt"
            ),
            multilingual_prompt_format_color_path=os.path.join(
                glyph_root, "assets/color_idx.json"
            ),
            multilingual_prompt_format_font_path=os.path.join(
                glyph_root, "assets/multilingual_10-lang_idx.json"
            ),
            byt5_max_length=256,
        )
        byt5_kwargs = load_glyph_byT5_v2(
            byt5_args,
            device=(
                f"cuda:{self.device}"
                if isinstance(self.device, int)
                else str(self.device)
            ),
        )
        self.prompt_format = MultilingualPromptFormat(
            font_path=byt5_args["multilingual_prompt_format_font_path"],
            color_path=byt5_args["multilingual_prompt_format_color_path"],
        )
        self.byt5_model = byt5_kwargs["byt5_model"]
        self.byt5_tokenizer = byt5_kwargs["byt5_tokenizer"]
        self.byt5_max_length = byt5_kwargs["byt5_max_length"]
        print("All models loaded.")

    def _process_byt5_prompt(self, prompt_text):
        """Process text prompt through byT5 encoder. Extracts quoted text (glyph text) from prompt
        and encodes it.

        Args:
            prompt_text: Input text prompt

        Returns:
            byt5_embeddings: [1, max_length, 1472] tensor
            byt5_mask: [1, max_length] attention mask
        """
        byt5_embeddings = torch.zeros(
            (1, self.byt5_max_length, 1472), device=self.device
        )
        byt5_mask = torch.zeros(
            (1, self.byt5_max_length), device=self.device, dtype=torch.int64
        )

        # Extract quoted text (glyph text) from prompt using regex
        pattern = r'\"(.*?)\"|"(.*?)"'
        matches = re.findall(pattern, prompt_text)
        glyph_texts = [m[0] or m[1] for m in matches]
        glyph_texts = (
            list(dict.fromkeys(glyph_texts))
            if len(glyph_texts) > 1
            else glyph_texts
        )

        if glyph_texts:
            # Format and encode glyph text
            text_styles = [
                {"color": None, "font-family": None} for _ in glyph_texts
            ]
            formatted_text = self.prompt_format.format_prompt(
                glyph_texts, text_styles
            )
            inputs = self.byt5_tokenizer(
                formatted_text,
                padding="max_length",
                max_length=self.byt5_max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_ids = inputs.input_ids.to(self.device)
            text_mask = inputs.attention_mask.to(self.device)
            byt5_embeddings = self.byt5_model(
                text_ids, attention_mask=text_mask.float()
            )[0]
            byt5_mask = text_mask

        return byt5_embeddings, byt5_mask

    @torch.no_grad()
    def extract(self, image_path, caption):
        """Extract all latents from an image and its caption.

        Args:
            image_path: Path to input image
            caption: Text caption/prompt for the image

        Returns:
            dict containing:
                - latent: VAE-encoded image latent [1, C, T, H, W]
                - image_cond: Condition latent (first frame) [1, C, 1, H, W]
                - prompt_embeds: Text embeddings from LLM encoder
                - prompt_mask: Attention mask for text
                - vision_states: Visual features from SigLIP
                - byt5_text_states: byT5 embeddings
                - byt5_text_mask: byT5 attention mask
        """
        # ===== Step 1: Load and preprocess image =====
        # Load image and convert to tensor [C, H, W], normalize to [-1, 1]
        image = Image.open(image_path).convert("RGB")
        image_tensor = (
            torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
        )
        image_tensor = (
            image_tensor / 255.0 - 0.5
        ) * 2.0  # Normalize to [-1, 1]

        # Resize to target resolution
        image_tensor = F.interpolate(
            image_tensor.unsqueeze(0),
            size=self.target_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        image_tensor = image_tensor.to(self.device)

        # ===== Step 2: VAE encoding =====
        # Encode image as condition latent (single frame)
        # Input shape: [B, C, T, H, W] = [1, 3, 1, H, W]
        cond_latents = self.vae.encode(
            image_tensor.unsqueeze(0).unsqueeze(2)
        ).latent_dist.mode()
        cond_latents.mul_(self.vae.config.scaling_factor)

        # Encode as video latent (for training)
        # Input shape: [B, C, T, H, W] = [1, 3, 1, H, W]
        latents = self.vae.encode(
            image_tensor.unsqueeze(1).unsqueeze(0)
        ).latent_dist.mode()
        latents.mul_(self.vae.config.scaling_factor)

        # ===== Step 3: Vision encoder (SigLIP) =====
        # Convert back to [0, 255] uint8 for vision encoder
        ref_image = (image_tensor + 1) * 127.5
        vision_states = self.vision_encoder.encode_images(
            ref_image.cpu().numpy().astype(np.uint8)
        )
        vision_states = vision_states.last_hidden_state.to(
            device=self.device, dtype=torch.bfloat16
        )

        # ===== Step 4: Text encoder (LLM) =====
        text_inputs = self.text_encoder.text2tokens(
            caption, data_type="video", max_length=self.text_len
        )
        prompt_outputs = self.text_encoder.encode(
            text_inputs, data_type="video", device=self.device
        )
        prompt_embeds = prompt_outputs.hidden_state.to(
            dtype=self.text_encoder.dtype, device=self.device
        )
        attention_mask = (
            prompt_outputs.attention_mask.to(self.device)
            if prompt_outputs.attention_mask is not None
            else None
        )

        # ===== Step 5: byT5 encoder =====
        byt5_embeddings, byt5_masks = self._process_byt5_prompt(caption)

        return {
            "latent": latents.to(torch.bfloat16),
            "prompt_embeds": prompt_embeds,
            "image_cond": cond_latents.to(torch.bfloat16),
            "vision_states": vision_states,
            "prompt_mask": attention_mask,
            "byt5_text_states": byt5_embeddings,
            "byt5_text_mask": byt5_masks,
        }


def main():
    args = parse_args()

    # ===== Distributed setup =====
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    # Create output directories
    latents_dir = os.path.join(args.output_dir, "latents")
    os.makedirs(latents_dir, exist_ok=True)

    # Load input data
    with open(args.input_json, "r") as f:
        data_list = json.load(f)

    # ===== Distribute data across GPUs =====
    if world_size > 1:
        per_gpu = (len(data_list) + world_size - 1) // world_size
        data_list = data_list[local_rank * per_gpu : (local_rank + 1) * per_gpu]

    print(f"GPU {local_rank}: Processing {len(data_list)} items")

    # Initialize extractor
    extractor = LatentExtractor(
        args.hunyuan_checkpoint_path,
        device,
        (args.target_height, args.target_width),
    )

    # ===== Process each item =====
    output_items = []
    for idx, item in enumerate(tqdm(data_list, desc=f"GPU {local_rank}")):
        image_path = item.get("image_path")
        caption = item.get("caption", "")

        if not image_path or not os.path.exists(image_path):
            continue

        item_id = f"{local_rank}_{idx:06d}"
        output_path = os.path.join(latents_dir, f"{item_id}.pt")

        # Skip if already processed
        if args.skip_existing and os.path.exists(output_path):
            output_items.append(
                {
                    "latent_path": output_path,
                    "image_path": image_path,
                    "caption": caption,
                }
            )
            continue

        try:
            latents = extractor.extract(image_path, caption)
            torch.save(latents, output_path)
            output_items.append(
                {
                    "latent_path": output_path,
                    "image_path": image_path,
                    "caption": caption,
                }
            )
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    # ===== Save output JSON =====
    output_json = os.path.join(args.output_dir, f"latents_gpu{local_rank}.json")
    with open(output_json, "w") as f:
        json.dump(output_items, f, indent=2, ensure_ascii=False)

    # ===== Merge results on rank 0 =====
    if local_rank == 0 and world_size > 1:
        import time

        time.sleep(3)  # Wait for other GPUs to finish writing
        all_items = []
        for gpu_id in range(world_size):
            gpu_json = os.path.join(
                args.output_dir, f"latents_gpu{gpu_id}.json"
            )
            if os.path.exists(gpu_json):
                with open(gpu_json, "r") as f:
                    all_items.extend(json.load(f))
        merged_json = os.path.join(args.output_dir, "latents.json")
        with open(merged_json, "w") as f:
            json.dump(all_items, f, indent=2, ensure_ascii=False)
        print(f"Merged {len(all_items)} items to {merged_json}")
    elif world_size == 1:
        os.rename(output_json, os.path.join(args.output_dir, "latents.json"))
        print(
            f"Saved {len(output_items)} items to {args.output_dir}/latents.json"
        )


if __name__ == "__main__":
    main()
