import argparse
import os
import sys

sys.path.append(os.path.abspath("."))

import importlib.util

import torch
from loguru import logger

_spec = importlib.util.spec_from_file_location(
    "preprocess_gamefactory_dataset",
    os.path.join(
        os.path.dirname(__file__), "hy_preprocess", "preprocess_gamefactory_dataset.py"
    ),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

encode_byt5_prompt = _mod.encode_byt5_prompt
encode_prompt = _mod.encode_prompt
load_byt5_encoder = _mod.load_byt5_encoder
load_text_encoder = _mod.load_text_encoder


def main():
    parser = argparse.ArgumentParser(description="Generate negative prompt embedding file")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to HunyuanVideo-1.5 model"
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device")
    parser.add_argument(
        "--neg_prompt",
        type=str,
        default="",
        help="Negative text prompt (default: empty string)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load text encoder
    logger.info("Loading text encoder...")
    text_encoders = load_text_encoder(args.model_path, device=args.device)

    # Encode negative prompt
    logger.info(f"Encoding negative prompt: '{args.neg_prompt}'")
    neg_prompt_dict = encode_prompt(args.neg_prompt, text_encoders, device=args.device)

    neg_prompt_save = {
        "negative_prompt_embeds": neg_prompt_dict["prompt_embeds"],  # [1, seq_len, dim]
        "negative_prompt_mask": neg_prompt_dict["prompt_mask"],  # [1, seq_len]
    }
    neg_prompt_path = os.path.join(args.output_dir, "hunyuan_neg_prompt.pt")
    torch.save(neg_prompt_save, neg_prompt_path)
    logger.info(f"Saved: {neg_prompt_path}")
    logger.info(
        f"  negative_prompt_embeds: {neg_prompt_save['negative_prompt_embeds'].shape}"
    )
    logger.info(
        f"  negative_prompt_mask:   {neg_prompt_save['negative_prompt_mask'].shape}"
    )

    # Free text encoder GPU memory
    del text_encoders
    torch.cuda.empty_cache()

    # Load byT5 encoder
    logger.info("Loading byT5 encoder...")
    byt5_encoders = load_byt5_encoder(args.model_path, device=args.device)

    logger.info(f"Encoding byT5 negative prompt: '{args.neg_prompt}'")
    neg_byt5_dict = encode_byt5_prompt(
        args.neg_prompt, byt5_encoders, device=args.device
    )

    neg_byt5_save = {
        "byt5_text_states": neg_byt5_dict["byt5_text_states"],  # [1, byt5_len, 1472]
        "byt5_text_mask": neg_byt5_dict["byt5_text_mask"],  # [1, byt5_len]
    }
    neg_byt5_path = os.path.join(args.output_dir, "hunyuan_neg_byt5_prompt.pt")
    torch.save(neg_byt5_save, neg_byt5_path)
    logger.info(f"Saved: {neg_byt5_path}")
    logger.info(f"  byt5_text_states: {neg_byt5_save['byt5_text_states'].shape}")
    logger.info(f"  byt5_text_mask:   {neg_byt5_save['byt5_text_mask'].shape}")

    logger.info(f"  neg_prompt_pt  → {neg_prompt_path}")
    logger.info(f"  neg_byt5_pt    → {neg_byt5_path}")


if __name__ == "__main__":
    main()
