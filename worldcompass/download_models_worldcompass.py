#!/usr/bin/env python3
"""Download script for HY-WorldPlay models. Downloads all required models from HuggingFace and
ModelScope.

Usage:
    python download_models.py --hf_token <your_token>

The HF token is required for downloading the vision encoder from FLUX.1-Redux-dev.
Request access at: https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev
"""

import argparse
import os
import shutil
import sys


def check_dependencies():
    """Check and install required dependencies."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub...")
        os.system("pip install -U 'huggingface_hub[cli]'")

    try:
        import modelscope
    except ImportError:
        print("Installing modelscope...")
        os.system("pip install modelscope")


def download_hy_worldplay(cache_dir=None):
    """Download only ar_model/diffusion_pytorch_model.safetensors from tencent/HY-WorldPlay."""
    from huggingface_hub import snapshot_download

    print("\n" + "=" * 60)
    print(
        "[1/8] Downloading ar_model/diffusion_pytorch_model.safetensors from tencent/HY-WorldPlay..."
    )
    print("=" * 60)

    # Only download the ar_model/diffusion_pytorch_model.safetensors file
    worldplay_path = snapshot_download(
        "tencent/HY-WorldPlay",
        allow_patterns=["ar_model/diffusion_pytorch_model.safetensors"],
        cache_dir=cache_dir,
    )
    model_path = os.path.join(
        worldplay_path, "ar_model", "diffusion_pytorch_model.safetensors"
    )
    # os.makedirs(target_dir, exist_ok=True)
    # target_path = os.path.join(target_dir, "diffusion_pytorch_model.safetensors")
    # if os.path.abspath(model_path) != os.path.abspath(target_path):
    #     shutil.move(model_path, target_path)
    #     model_path = target_path

    print(f"Downloaded file: {model_path}")

    return model_path


def download_hunyuan_video(cache_dir=None):
    """Download HunyuanVideo-1.5 base models (vae, scheduler, transformer)."""
    from huggingface_hub import snapshot_download

    print("\n" + "=" * 60)
    print(
        "[2/8] Downloading tencent/HunyuanVideo-1.5 (vae, scheduler, transformer)..."
    )
    print("=" * 60)

    hunyuan_path = snapshot_download(
        "tencent/HunyuanVideo-1.5",
        allow_patterns=["vae/*", "scheduler/*", "transformer/480p_i2v/*"],
        cache_dir=cache_dir,
    )
    print(f"Downloaded to: {hunyuan_path}")
    return hunyuan_path


def download_llm_text_encoder(hunyuan_path, cache_dir=None):
    """Download Qwen2.5-VL-7B-Instruct as the LLM text encoder."""
    from huggingface_hub import snapshot_download

    print("\n" + "=" * 60)
    print("[3/8] Downloading LLM text encoder (Qwen2.5-VL-7B-Instruct)...")
    print("=" * 60)

    text_encoder_base = os.path.join(hunyuan_path, "text_encoder")
    os.makedirs(text_encoder_base, exist_ok=True)

    llm_target = os.path.join(text_encoder_base, "llm")

    if (
        os.path.exists(llm_target)
        and os.path.isdir(llm_target)
        and len(os.listdir(llm_target)) > 5
    ):
        print(f"LLM text encoder already exists at: {llm_target}")
        return

    # Clean up old/broken downloads
    if os.path.islink(llm_target):
        os.unlink(llm_target)
    elif os.path.exists(llm_target):
        shutil.rmtree(llm_target)

    print("Downloading Qwen/Qwen2.5-VL-7B-Instruct (~15GB)...")
    qwen_cache = snapshot_download(
        "Qwen/Qwen2.5-VL-7B-Instruct", cache_dir=cache_dir
    )

    # Copy files (resolve symlinks)
    os.makedirs(llm_target, exist_ok=True)
    for item in os.listdir(qwen_cache):
        src = os.path.realpath(os.path.join(qwen_cache, item))
        dst = os.path.join(llm_target, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    print(f"Copied to: {llm_target}")


def download_byt5_encoders(hunyuan_path, cache_dir=None):
    """Download ByT5 text encoders (byt5-small and Glyph-SDXL-v2)."""
    from huggingface_hub import snapshot_download
    from modelscope import snapshot_download as ms_snapshot_download

    print("\n" + "=" * 60)
    print("[4/8] Downloading ByT5 text encoders...")
    print("=" * 60)

    text_encoder_base = os.path.join(hunyuan_path, "text_encoder")
    os.makedirs(text_encoder_base, exist_ok=True)

    # 1. Download google/byt5-small
    byt5_target = os.path.join(text_encoder_base, "byt5-small")
    if (
        os.path.exists(byt5_target)
        and os.path.isdir(byt5_target)
        and len(os.listdir(byt5_target)) > 3
    ):
        print(f"byt5-small already exists at: {byt5_target}")
    else:
        if os.path.islink(byt5_target):
            os.unlink(byt5_target)
        elif os.path.exists(byt5_target):
            shutil.rmtree(byt5_target)

        print("Downloading google/byt5-small...")
        byt5_cache = snapshot_download("google/byt5-small", cache_dir=cache_dir)

        os.makedirs(byt5_target, exist_ok=True)
        for item in os.listdir(byt5_cache):
            src = os.path.realpath(os.path.join(byt5_cache, item))
            dst = os.path.join(byt5_target, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        print(f"Copied to: {byt5_target}")

    # 2. Download Glyph-SDXL-v2 from ModelScope
    glyph_target = os.path.join(text_encoder_base, "Glyph-SDXL-v2")
    if os.path.exists(glyph_target) and os.path.exists(
        os.path.join(glyph_target, "checkpoints", "byt5_model.pt")
    ):
        print(f"Glyph-SDXL-v2 already exists at: {glyph_target}")
    else:
        if os.path.exists(glyph_target):
            shutil.rmtree(glyph_target)

        print("Downloading AI-ModelScope/Glyph-SDXL-v2 from ModelScope...")
        # Use custom cache_dir if provided, otherwise use /tmp/glyph_cache
        glyph_cache_dir = cache_dir if cache_dir else "/tmp/glyph_cache"
        glyph_cache = ms_snapshot_download(
            "AI-ModelScope/Glyph-SDXL-v2", cache_dir=glyph_cache_dir
        )

        os.makedirs(glyph_target, exist_ok=True)
        for item in os.listdir(glyph_cache):
            src = os.path.join(glyph_cache, item)
            dst = os.path.join(glyph_target, item)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        print(f"Copied to: {glyph_target}")


def download_vision_encoder(hunyuan_path, hf_token, cache_dir=None):
    """Download SigLIP vision encoder from FLUX.1-Redux-dev."""
    from huggingface_hub import snapshot_download

    print("\n" + "=" * 60)
    print("[5/8] Downloading Vision Encoder (SigLIP from FLUX.1-Redux-dev)...")
    print("=" * 60)

    if not hf_token:
        print("WARNING: No HF token provided!")
        print(
            "The vision encoder requires access to: https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev"
        )
        print("Skipping vision encoder download.")
        print("\nYou can download it manually later.")
        return

    vision_encoder_base = os.path.join(hunyuan_path, "vision_encoder")
    os.makedirs(vision_encoder_base, exist_ok=True)

    siglip_target = os.path.join(vision_encoder_base, "siglip")

    if (
        os.path.exists(siglip_target)
        and os.path.isdir(siglip_target)
        and len(os.listdir(siglip_target)) > 3
    ):
        print(f"siglip already exists at: {siglip_target}")
        return

    # Clean up old/broken downloads
    if os.path.islink(siglip_target):
        os.unlink(siglip_target)
    elif os.path.exists(siglip_target):
        shutil.rmtree(siglip_target)

    print("Downloading black-forest-labs/FLUX.1-Redux-dev...")
    try:
        flux_cache = snapshot_download(
            "black-forest-labs/FLUX.1-Redux-dev",
            token=hf_token,
            cache_dir=cache_dir,
        )

        # Copy files (resolve symlinks)
        os.makedirs(siglip_target, exist_ok=True)
        for item in os.listdir(flux_cache):
            src = os.path.realpath(os.path.join(flux_cache, item))
            dst = os.path.join(siglip_target, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        print(f"Copied to: {siglip_target}")
    except Exception as e:
        print(f"ERROR: Failed to download vision encoder: {e}")
        print(
            "Make sure you have requested access to FLUX.1-Redux-dev and your token is valid."
        )


def download_worldmirror(cache_dir=None):
    """Download HunyuanWorld-Mirror for camera pose estimation."""
    print("\n" + "=" * 60)
    print("[6/8] Downloading WorldMirror (tencent/HunyuanWorld-Mirror)...")
    print("=" * 60)

    try:
        from huggingface_hub import snapshot_download

        print("Downloading tencent/HunyuanWorld-Mirror...")
        worldmirror_path = snapshot_download(
            "tencent/HunyuanWorld-Mirror", cache_dir=cache_dir
        )
        print(f"Downloaded to: {worldmirror_path}")

        # Try to load with WorldMirror if available
        try:
            from reward_function.HunyuanWorldMirror import WorldMirror

            worldmirror_model = WorldMirror.from_pretrained(worldmirror_path)
            print("✓ WorldMirror model loaded successfully")
        except ImportError:
            print(
                "✓ WorldMirror model downloaded (module will load on first use)"
            )

        return worldmirror_path
    except ImportError:
        print("WARNING: WorldMirror module not found.")
        print(
            "Skipping WorldMirror download. The model will be downloaded on first use."
        )
        return None
    except Exception as e:
        print(f"WARNING: Failed to download WorldMirror: {e}")
        print("The model will be downloaded on first use.")
        return None


def download_depth_anything_3(cache_dir=None):
    """Download DepthAnything3 model for camera pose estimation (optional)."""
    print("\n" + "=" * 60)
    print("[7/8] Downloading DepthAnything3 (depth-anything/DA3-GIANT-1.1)...")
    print("=" * 60)

    from huggingface_hub import snapshot_download

    print("Downloading depth-anything/DA3-GIANT-1.1...")
    da3_path = snapshot_download(
        "depth-anything/DA3-GIANT-1.1", cache_dir=cache_dir
    )
    print(f"Downloaded to: {da3_path}")

    return None


def print_paths(cache_dir=None):
    """Print the model paths for run.sh configuration."""
    from huggingface_hub import snapshot_download

    print("\n" + "=" * 60)
    print("[8/8] Verifying downloads...")
    print("=" * 60)

    hunyuan_path = snapshot_download(
        "tencent/HunyuanVideo-1.5", local_files_only=True, cache_dir=cache_dir
    )
    worldplay_path = snapshot_download(
        "tencent/HY-WorldPlay", local_files_only=True, cache_dir=cache_dir
    )

    # Try to get WorldMirror path
    worldmirror_path = snapshot_download(
        "tencent/HunyuanWorld-Mirror",
        local_files_only=True,
        cache_dir=cache_dir,
    )
    da3_path = snapshot_download(
        "depth-anything/DA3-GIANT-1.1",
        local_files_only=True,
        cache_dir=cache_dir,
    )

    print("\n" + "=" * 60)
    print("ALL DOWNLOADS COMPLETE!")
    print("=" * 60)
    print(
        "\nADD these paths to your prepare_dataset/prepare_image_text_latent_simple.py:\n"
    )
    print(f"--hunyuan_checkpoint_path {hunyuan_path}")

    print("\nModify these paths in your scripts/full_hy_nft.sh:\n")
    print(f"CACHE_DIR={cache_dir}")
    print(f"HUNYUAN_CHECKPOINT={hunyuan_path}")
    print(f"WORLDPLAY_CHECKPOINT={worldplay_path}")

    print(
        "\nYou can now run: bash prepare_dataset/extract_latents.sh to prepare your dataset"
    )
    print("\nAnd then run: bash scripts/full_hy_nft.sh to start training")


def main():
    parser = argparse.ArgumentParser(
        description="Download all required models for HY-WorldPlay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download to default cache directory (~/.cache/huggingface/hub/)
    python download_models.py --hf_token hf_xxxxxxxxxxxxx
    
    # Download to custom directory
    python download_models.py --hf_token hf_xxxxxxxxxxxxx --cache_dir /mnt/data/models

Note:
    The HuggingFace token is required for downloading the vision encoder
    from black-forest-labs/FLUX.1-Redux-dev. You need to:
    1. Request access at: https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev
    2. Wait for approval (usually instant)
    3. Create a token at: https://huggingface.co/settings/tokens (select "Read" permission)
        """,
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token for downloading gated models (required for vision encoder)",
    )
    parser.add_argument(
        "--skip_vision_encoder",
        action="store_true",
        help="Skip downloading the vision encoder (if you don't have FLUX access yet)",
    )
    parser.add_argument(
        "--skip_worldmirror",
        action="store_true",
        help="Skip downloading WorldMirror model (optional, used for camera pose estimation)",
    )
    parser.add_argument(
        "--skip_depth_anything_3",
        action="store_true",
        help="Skip downloading DepthAnything3 model (optional, alternative to WorldMirror)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory to cache downloaded models (default: ~/.cache/huggingface/hub/)",
    )

    args = parser.parse_args()

    # Validate and prepare cache_dir
    cache_dir = None
    if args.cache_dir:
        cache_dir = os.path.abspath(os.path.expanduser(args.cache_dir))
        os.makedirs(cache_dir, exist_ok=True)
        print(f"\n{'=' * 60}")
        print(f"Using custom cache directory: {cache_dir}")
        print(f"{'=' * 60}\n")

    print("=" * 60)
    print("HY-WorldPlay Model Download Script")
    print("=" * 60)

    # Check dependencies
    check_dependencies()

    # Download models
    worldplay_path = download_hy_worldplay(cache_dir=cache_dir)
    hunyuan_path = download_hunyuan_video(cache_dir=cache_dir)
    download_llm_text_encoder(hunyuan_path, cache_dir=cache_dir)
    download_byt5_encoders(hunyuan_path, cache_dir=cache_dir)

    if not args.skip_vision_encoder:
        download_vision_encoder(
            hunyuan_path, args.hf_token, cache_dir=cache_dir
        )
    else:
        print(
            "\n[5/8] Skipping vision encoder download (--skip_vision_encoder flag)"
        )

    download_worldmirror(cache_dir=cache_dir)
    download_depth_anything_3(cache_dir=cache_dir)

    # Print final paths
    print_paths(cache_dir=cache_dir)


if __name__ == "__main__":
    main()
