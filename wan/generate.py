# Video generation script with distributed inference support
import torch
import sys, os
import time
import argparse
import json
from typing import List, Dict

sys.path.append(os.path.abspath("."))
import numpy as np

from wan.inference.helper import (
    MyVAE,
    CHUNK_SIZE,
)
# Lazy import pose utilities to avoid hyvideo's parallel state init at module load
# (hyvideo/generate.py calls initialize_parallel_state() at import time which
# conflicts with WAN's own distributed init when running in single-GPU mode)
from wan.inference.pipeline_wan_w_mem_relative_rope import WanPipeline
from wan.models.dits.arwan_w_action_w_mem_relative_rope import WanTransformer3DModel
from wan.distributed import (
    get_world_group,
    maybe_init_distributed_environment_and_model_parallel,
)

from wan.models.par_vae.tools import DistController
from wan.models.par_vae.context_parallel.wrapper_vae import DistWrapper
from tqdm import tqdm

from diffusers.utils import export_to_video
from safetensors.torch import load_file




def sanitize_filename(text: str, max_length: int = 50) -> str:
    """
    Convert text to safe filename format.

    Args:
        text: Input text (e.g., prompt)
        max_length: Maximum length of output filename

    Returns:
        Sanitized filename string
    """
    import re

    # Remove or replace invalid filename characters
    # Strip special characters that are unsafe for filenames
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", text)
    # Replace spaces and commas with underscores
    text = re.sub(r"[\s,]+", "_", text)
    # Remove multiple consecutive underscores
    text = re.sub(r"_+", "_", text)
    # Strip leading/trailing underscores
    text = text.strip("_")

    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length].rstrip("_")

    return text


class WanRunner:
    def __init__(
        self,
        model_id,
        ckpt_path,
        ar_model_path,
    ):
        # Get distributed environment info from torchrun
        # Extract rank information from environment variables
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        self.model_id = model_id
        self.ckpt_path = ckpt_path
        self.ar_model_path = ar_model_path

        # Set device
        # Assign CUDA device based on local rank for multi-GPU setup
        device = torch.device(
            f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        )
        self.device = device
        torch.cuda.set_device(self.local_rank)

        # Initialize distributed environment - needed even for single GPU
        # Setup distributed training for multi-process inference
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        # Initialize model parallel with project's custom function
        # Enable sequence parallelism (sp_size=1 for single GPU)
        maybe_init_distributed_environment_and_model_parallel(
            1,
            sp_size=self.world_size,
            distributed_init_method="env://",
        )

        # Initialize models
        self._init_models()

    def _init_models(self):
        # Load VAE for encoding/decoding video frames
        # Use low_cpu_mem_usage to avoid loading entire model to RAM first
        self.vae = (
            MyVAE.from_pretrained(
                self.model_id, subfolder="vae", torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True, device_map=self.device
            )
            .eval()
            .requires_grad_(False)
        )

        # Load main diffusion pipeline - load directly to GPU
        self.pipe = WanPipeline.from_pretrained(
            self.model_id, vae=self.vae, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        # Move pipeline to GPU immediately after loading each component
        self.pipe.to(self.device)

        # Reuse the same VAE for distributed wrapper instead of loading another copy
        dist_vae = self.vae
        vae_chunk_dim = 4  # chunk width dim for vae input
        # Setup VAE for context-parallel decoding across GPUs
        dist_controller = DistController(
            self.rank, self.local_rank, self.world_size, None
        )
        dist_vae = DistWrapper(dist_vae, dist_controller, None, vae_chunk_dim)
        self.pipe.dist_vae = dist_vae

        # Load autoregressive transformer with action conditioning
        # Use low_cpu_mem_usage to load directly to GPU piece by piece
        transformer_ar_action = WanTransformer3DModel.from_pretrained(
            self.ar_model_path,
            use_safetensors=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
        )
        # Add action parameters to enable camera control
        transformer_ar_action.add_discrete_action_parameters()

        # Load trained checkpoint weights
        # Load directly to GPU to avoid OOM on systems with limited RAM
        state_dict = torch.load(
            self.ckpt_path, map_location=self.device
        )

        state_dict = state_dict["generator"]
        # Clean up state dict keys from training artifacts
        state_dict = {
            key.replace("model.", "") if key.startswith("model.") else key: value
            for key, value in state_dict.items()
        }
        state_dict = {
            (
                key.replace("_fsdp_wrapped_module.", "")
                if key.startswith("_fsdp_wrapped_module.")
                else key
            ): value
            for key, value in state_dict.items()
        }

        transformer_ar_action.load_state_dict(state_dict, strict=True)
        # Transformer already on GPU from device_map, just ensure dtype
        self.pipe.transformer = transformer_ar_action.to(dtype=torch.bfloat16)
        # Ensure entire pipeline is on GPU
        self.pipe.to(self.device)

    def predict(self, input_dict):
        prompt = input_dict.get("prompt", "")
        negative_prompt = input_dict.get("negative_prompt", "")
        num_frames = input_dict.get("num_frames", 189)
        num_inference_steps = input_dict.get("num_inference_steps", 50)
        height = input_dict.get("height", 704)
        width = input_dict.get("width", 1280)
        image_path = input_dict.get("image_path", None)
        use_memory = input_dict.get("use_memory", True)
        context_window_length = input_dict.get("context_window_length", 16)
        pose = input_dict.get("pose", "w-96")  # Default to forward movement
        num_chunk = input_dict.get("num_chunk", 4)

        seed = input_dict.get("seed", 0)
        torch.manual_seed(seed)

        all_video = []
        run_args = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=5.0,  # Higher = more prompt adherence + dynamics
            few_step=True,
            first_chunk_size=CHUNK_SIZE,
            return_dict=False,
            image_path=image_path,
            use_memory=use_memory,
            context_window_length=context_window_length,
        )

        # Convert pose string to pose json
        # Lazy import to avoid hyvideo's parallel state init at module load time
        # (by this point, WAN's distributed init has already completed)
        from hyvideo.generate import pose_string_to_json, pose_to_input
        
        pose_json = pose_string_to_json(pose)
        all_viewmats, all_Ks, all_action = pose_to_input(pose_json, len(pose_json))

        all_save = []
        for chunk_i in range(num_chunk):
            torch.cuda.synchronize()
            begin_time = time.time()

            start_idx = chunk_i * CHUNK_SIZE
            end_idx = start_idx + CHUNK_SIZE

            curr_viewmats = all_viewmats[start_idx:end_idx]
            curr_Ks = all_Ks[start_idx:end_idx]
            curr_action = all_action[start_idx:end_idx]

            curr_viewmat_save = curr_viewmats[0].cpu().numpy().tolist()
            curr_action_save = curr_action[0].item()
            curr_save = {"viewmat": curr_viewmat_save, "action": curr_action_save}
            all_save.append(curr_save)

            self.pipe(
                **run_args,
                chunk_i=chunk_i,
                viewmats=curr_viewmats.unsqueeze(0),
                Ks=curr_Ks.unsqueeze(0),
                action=curr_action.unsqueeze(0),
                output_type="latent",
            )

            if self.rank == 0:
                print(
                    "Generate time for chunk ", chunk_i, "is", time.time() - begin_time
                )

            # decode
            for i in range(4):
                decode_start_time = time.time()
                video = self.pipe.decode_next_latent(output_type="np")
                decode_time = time.time() - decode_start_time
                if self.rank == 0 and i == 0:
                    print(f"Decode latent {i}: {decode_time:.4f} seconds")
                all_video.append(video)

        if self.rank == 0:
            video = np.concatenate(all_video, axis=1)
            return {"video": video, "pose": pose}
        else:
            return {"video": None, "pose": pose}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=961)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument(
        "--input",
        type=str,
        default="First-person view walking around ancient Athens, with Greek architecture and marble structures",
        help="Model input in format 'prompt@image_path' or txt inputs file",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=(
            "色调艳丽,过曝,静态,细节模糊不清,字幕,风格,作品,画作,画面,静止,整体发灰,"
            "最差质量,低质量,JPEG压缩残留,丑陋的,残缺的,多余的手指,画得不好的手部,"
            "画得不好的脸部,畸形的,毁容的,形态畸形的肢体,手指融合,静止不动的画面,"
            "杂乱的背景,三条腿,背景人很多,倒着走"
        ),
    )
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument(
        "--out",
        type=str,
        default="outputs",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/root/model.pt",
    )
    parser.add_argument(
        "--pose",
        type=str,
        default="w-96",
        help="Pose string (e.g. 'w-3, right-0.5, d-4') or path to pose JSON file",
    )
    parser.add_argument("--ar_model_path", type=str, default=None)
    parser.add_argument("--num_chunk", type=int, default=4, help="Number of chunks to generate")
    parser.add_argument(
        "--model_id",
        type=str,
        default="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        help="Model ID for the pretrained model",
    )

    args = parser.parse_args()

    # Get rank info from torchrun environment
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if rank == 0:
        print(f"Running with world_size={world_size}")

    # Initialize runner (distributed initialization handled inside)
    runner = WanRunner(
        model_id=args.model_id,
        ckpt_path=args.ckpt_path,
        ar_model_path=args.ar_model_path,
    )

    # Process input arguments (only rank 0 needs to parse inputs)
    if rank == 0:
        model_inputs = []
        if os.path.isfile(args.input):
            # Read from txt file
            with open(args.input, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        model_inputs.append(line)
        else:
            # Use direct input
            model_inputs = [args.input]

        # Parse model inputs
        parsed_inputs = []
        for model_input in model_inputs:
            if "@" in model_input:
                prompt, image_path = model_input.split("@", 1)
                parsed_inputs.append(
                    {"prompt": prompt, "image_path": image_path if image_path else None}
                )
            else:
                parsed_inputs.append({"prompt": model_input, "image_path": None})

        # Prepare error log file
        save_dir = args.out
        os.makedirs(save_dir, exist_ok=True)
        error_log_path = os.path.join(save_dir, "err.txt")
    else:
        parsed_inputs = []
        save_dir = None
        error_log_path = None

    # Broadcast inputs to all ranks if using distributed
    if world_size > 1:
        import torch.distributed as dist

        # Prepare data for broadcast
        if rank == 0:
            broadcast_data = {
                "parsed_inputs": parsed_inputs,
                "pose": args.pose,
                "save_dir": save_dir,
                "error_log_path": error_log_path,
            }
        else:
            broadcast_data = None

        # Broadcast using object list
        object_list = [broadcast_data]
        dist.broadcast_object_list(object_list, src=0)
        broadcast_data = object_list[0]

        parsed_inputs = broadcast_data["parsed_inputs"]
        pose = broadcast_data["pose"]
        save_dir = broadcast_data["save_dir"]
        error_log_path = broadcast_data["error_log_path"]
    else:
        pose = args.pose

    try:
        # Run inference
        for i, input_data in enumerate(parsed_inputs):
            seed_idx = 0
            if rank == 0:
                prompt_prefix = sanitize_filename(input_data["prompt"], max_length=50)
                # Create filename based on pose string (sanitized)
                pose_prefix = sanitize_filename(
                    pose if not pose.endswith(".json") else os.path.basename(pose),
                    max_length=30,
                )
                output_filename = f"{pose_prefix}_{prompt_prefix}.mp4"
                output_path = os.path.join(save_dir, output_filename)

                # Check if output already exists
                if os.path.exists(output_path):
                    print(f"\n{'='*80}")
                    print(f"Skipping case {i}, pose: {pose}, seed: {seed_idx}")
                    print(f"Output already exists: {output_filename}")
                    print(f"{'='*80}\n")
                    continue

            try:
                # Check if image path is provided and exists
                image_path = input_data["image_path"] or args.image_path
                if rank == 0 and image_path and not os.path.exists(image_path):
                    print(f"\n{'='*80}")
                    print(f"Skipping case {i}, pose: {pose}, seed: {seed_idx}")
                    print(f"Image path does not exist: {image_path}")
                    print(f"{'='*80}\n")
                    continue

                input_dict = {
                    "prompt": input_data["prompt"],
                    "negative_prompt": args.negative_prompt,
                    "num_frames": args.num_frames,
                    "num_inference_steps": args.num_inference_steps,
                    "guidance_scale": 1,
                    "height": 704,
                    "width": 1280,
                    "image_path": image_path,
                    "use_memory": True,
                    "context_window_length": 16,
                    "seed": seed_idx,
                    "pose": pose,
                    "num_chunk": args.num_chunk,
                }

                if rank == 0:
                    print(f"\n{'='*80}")
                    print(f"Processing case {i}, pose: {pose}, seed: {seed_idx}")
                    print(f"{'='*80}\n")

                start_time = time.time()
                result = runner.predict(input_dict)

                # Save results (only rank 0)
                if rank == 0:
                    video = result["video"]
                    export_to_video(video[0], output_path, fps=16)
                    print(f"Video saved to {output_path}")
                    print("Generate time:", time.time() - start_time)

            except Exception as e:
                import traceback
                if rank == 0:
                    error_msg = (
                        f"\n{'='*80}\n"
                        f"Error occurred:\n"
                        f"  Case: {i}\n"
                        f"  Pose: {pose}\n"
                        f"  Seed: {seed_idx}\n"
                        f"  Prompt: {input_data['prompt']}\n"
                        f"  Error: {str(e)}\n"
                        f"  Traceback:\n{traceback.format_exc()}\n"
                        f"{'='*80}\n"
                    )
                    print(error_msg)

                    # Append error to log file
                    with open(error_log_path, "a", encoding="utf-8") as f:
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_msg}\n")

                # Continue with next case
                continue

    finally:
        # Cleanup distributed environment
        if world_size > 1 and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
