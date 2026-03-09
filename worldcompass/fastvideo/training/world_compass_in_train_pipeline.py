# SPDX-License-Identifier: Apache-2.0
import dataclasses
import math
import os
from re import U
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterator
from typing import Any

import imageio
import numpy as np
import torch
import torch.distributed as dist
import torchvision
from diffusers import FlowMatchEulerDiscreteScheduler
from einops import rearrange
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm.auto import tqdm
import re
import json

from fastvideo.models.utils import (
    generate_points_in_sphere,
    select_aligned_memory_frames_context_per_chunk_w_latent_sink_fov_refine_hunyuan,
)

from fastvideo.distributed.parallel_state import get_sp_parallel_rank

import fastvideo.envs as envs
from fastvideo.attention.backends.video_sparse_attn import (
    VideoSparseAttentionMetadataBuilder,
)
from fastvideo.dataset import build_hy_camera_dataloader
from fastvideo.dataset.dataloader.schema import pyarrow_schema_t2v
from fastvideo.dataset.validation_dataset import ValidationDataset
from fastvideo.distributed import (
    cleanup_dist_env_and_memory,
    get_local_torch_device,
    get_sp_group,
    get_world_group,
    get_gpu_group,
)
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.pipelines import (
    ComposedPipelineBase,
    ForwardBatch,
    LoRAPipeline,
    TrainingBatch,
)
from fastvideo.training.activation_checkpoint import apply_activation_checkpointing
from fastvideo.training.training_utils import (
    EMA_FSDP_schedule,
    clip_grad_norm_while_handling_failing_dtensor_cases,
    compute_density_for_timestep_sampling,
    get_scheduler,
    load_checkpoint,
    normalize_dit_input,
    save_checkpoint,
    shard_latents_dim_across_sp,
)
from fastvideo.utils import is_vsa_available, set_random_seed, shallow_asdict
from fastvideo.training.muon import get_muon_optimizer
from fastvideo.models.hyvideo.models.autoencoders import hunyuanvideo_15_vae_w_cache

from reward_function.reward_function import CompassReward

import wandb  # isort: skip
import random
import copy

vsa_available = is_vsa_available()

logger = init_logger(__name__)


def merge_tensor_by_mask(tensor_1, tensor_2, mask, dim):
    assert tensor_1.shape == tensor_2.shape
    # Mask is a 0/1 vector. Choose tensor_2 when the value is 1; otherwise, tensor_1
    masked_indices = torch.nonzero(mask).squeeze(1)
    tmp = tensor_1.clone()
    if dim == 0:
        tmp[masked_indices] = tensor_2[masked_indices]
    elif dim == 1:
        tmp[:, masked_indices] = tensor_2[:, masked_indices]
    elif dim == 2:
        tmp[:, :, masked_indices] = tensor_2[:, :, masked_indices]
    return tmp


def _get_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class WorldCompassInTrainPipeline(LoRAPipeline, ABC):
    """A pipeline for training a model.

    All training pipelines should inherit from this class. All reusable components and code should
    be implemented in this class.
    """

    _required_config_modules = ["scheduler", "transformer"]
    validation_pipeline: ComposedPipelineBase
    train_dataloader: StatefulDataLoader
    train_loader_iter: Iterator[dict[str, Any]]
    current_epoch: int = 0

    def __init__(
        self,
        model_path: str,
        fastvideo_args: TrainingArgs,
        required_config_modules: list[str] | None = None,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> None:
        self.lora_training = fastvideo_args.lora_training
        if self.lora_training and fastvideo_args.lora_rank is None:
            raise ValueError("lora rank must be set when using lora training")

        set_random_seed(fastvideo_args.seed)  # for lora param init
        super().__init__(
            model_path, fastvideo_args, required_config_modules, loaded_modules
        )  # type: ignore

        # Initialize KV cache storage for GRPO sampling
        self.points_local = generate_points_in_sphere(50000, 8.0)
        self.sample_kv_caches = {}  # Dict to store KV caches for all samples and chunks

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        raise RuntimeError(
            "create_pipeline_stages should not be called for training pipeline"
        )

    def set_schemas(self) -> None:
        self.train_dataset_schema = pyarrow_schema_t2v

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        logger.info("Initializing training pipeline...")
        self.device = get_local_torch_device()
        self.training_args = training_args
        world_group = get_world_group()
        self.world_size = world_group.world_size
        self.global_rank = world_group.rank
        self.sp_group = get_sp_group()
        self.gpu_group = get_gpu_group()
        self.rank_in_sp_group = self.sp_group.rank_in_group
        self.sp_world_size = self.sp_group.world_size
        self.local_rank = world_group.local_rank
        self.transformer = self.get_module("transformer")
        # self.vae = self.get_module("vae")

        # Load VAE from config path
        vae_path = training_args.vae_path
        if not vae_path:
            raise ValueError("vae_path must be provided in training_args")

        self.vae = hunyuanvideo_15_vae_w_cache.AutoencoderKLConv3D.from_pretrained(
            vae_path, torch_dtype=torch.float32
        ).to("cpu")
        self.vae = self.vae.to(torch.float32)

        # Initialize reward model with configurable camera estimator
        camera_estimator = getattr(training_args, "camera_estimator", "dav3")
        cache_dir = getattr(training_args, "cache_dir", None) or None
        logger.info(
            f"Initializing reward model with camera_estimator={camera_estimator}"
        )
        self.reward_model = CompassReward(
            device=self.device, camera_estimator=camera_estimator, cache_dir=cache_dir
        )

        self.seed = training_args.seed
        self.set_schemas()
        self.action = training_args.action

        assert self.seed is not None, "seed must be set"
        set_random_seed(self.seed)
        self.transformer.train()
        if training_args.enable_gradient_checkpointing_type is not None:
            self.transformer = apply_activation_checkpointing(
                self.transformer,
                checkpointing_type=training_args.enable_gradient_checkpointing_type,
            )

        self.set_trainable()
        params_to_optimize = self.transformer.parameters()
        params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))

        self.vae.requires_grad = False

        if self.training_args.eval_only:
            self.transformer.requires_grad = False
            self.transformer = self.transformer.to(torch.bfloat16)

        self.optimizer = get_muon_optimizer(
            model=self.transformer,
            lr=training_args.learning_rate,  # Learning rate
            weight_decay=training_args.weight_decay,  # Weight decay
            adamw_betas=(0.9, 0.999),  # AdamW betas for 1D parameters
            adamw_eps=1e-8,  # AdamW epsilon
        )

        self.init_steps = 0
        logger.info("optimizer: %s", self.optimizer)

        self.lr_scheduler = get_scheduler(
            training_args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=training_args.lr_warmup_steps,
            num_training_steps=training_args.max_train_steps,
            num_cycles=training_args.lr_num_cycles,
            power=training_args.lr_power,
            min_lr_ratio=training_args.min_lr_ratio,
            last_epoch=self.init_steps - 1,
        )

        if self.action:
            self.train_dataset, self.train_dataloader = build_hy_camera_dataloader(
                json_path=training_args.json_path,
                causal=training_args.causal,
                window_frames=training_args.window_frames,
                batch_size=training_args.train_batch_size,
                num_data_workers=training_args.dataloader_num_workers,
                drop_last=True,
                drop_first_row=False,
                seed=self.seed,
                cfg_rate=training_args.training_cfg_rate,
                i2v_rate=training_args.i2v_rate,
                random_pose_path=training_args.random_pose_path,
                neg_prompt_path=training_args.neg_prompt_path,
                neg_byt5_prompt_path=training_args.neg_byt5_prompt_path,
            )
            self.eval_dataset, self.eval_dataloader = build_hy_camera_dataloader(
                json_path=training_args.eval_json_path,
                causal=training_args.causal,
                window_frames=training_args.window_frames,
                batch_size=training_args.train_batch_size,
                num_data_workers=training_args.dataloader_num_workers,
                drop_last=True,
                drop_first_row=False,
                seed=self.seed,
                cfg_rate=training_args.training_cfg_rate,
                i2v_rate=training_args.i2v_rate,
                random_pose_path=training_args.random_pose_path,
                neg_prompt_path=training_args.neg_prompt_path,
                neg_byt5_prompt_path=training_args.neg_byt5_prompt_path,
            )

        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader)
            / training_args.gradient_accumulation_steps
            * training_args.sp_size
            / training_args.train_sp_batch_size
        )
        self.num_train_epochs = math.ceil(
            training_args.max_train_steps / self.num_update_steps_per_epoch
        )

        self.current_epoch = 0

        if self.global_rank == 0:
            project = training_args.tracker_project_name or "fastvideo"
            wandb_config = dataclasses.asdict(training_args)
            wandb.login(key=training_args.wandb_key)
            wandb.init(
                config=wandb_config,
                name=training_args.output_dir.split("/")[-1],
                entity=training_args.wandb_entity,
                project=project,
            )

    @abstractmethod
    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        raise NotImplementedError("Training pipelines must implement this method")

    def _prepare_training(self, training_batch: TrainingBatch) -> TrainingBatch:
        self.vae.eval()
        self.transformer.train()
        self.optimizer.zero_grad()
        training_batch.total_loss = 0.0
        training_batch.samples_loss = {}
        training_batch.samples_grad_norm = {}
        return training_batch

    def _get_next_batch(self, training_batch: TrainingBatch) -> TrainingBatch:
        batch = next(self.train_loader_iter, None)  # type: ignore
        if batch is None:
            self.current_epoch += 1
            logger.info("Starting epoch %s", self.current_epoch)
            # Reset iterator for next epoch
            self.train_loader_iter = iter(self.train_dataloader)
            # Get first batch of new epoch
            batch = next(self.train_loader_iter)

        extra_kwargs = {
            "byt5_text_states": batch["byt5_text_states"].to(
                get_local_torch_device(), dtype=torch.bfloat16
            ),
            "byt5_text_mask": batch["byt5_text_mask"].to(
                get_local_torch_device(), dtype=torch.bfloat16
            ),
        }
        multitask_mask = self.get_task_mask("i2v", batch["latent"].shape[2])
        cond_latents = self._prepare_cond_latents(
            "i2v", batch["image_cond"], batch["latent"], multitask_mask
        )

        training_batch.latents = batch["latent"].to(
            get_local_torch_device(), dtype=torch.bfloat16
        )
        training_batch.cond_latents = cond_latents.to(
            get_local_torch_device(), dtype=torch.bfloat16
        )
        training_batch.latents_concat = torch.concat(
            [batch["latent"], cond_latents], dim=1
        ).to(get_local_torch_device(), dtype=torch.bfloat16)
        training_batch.prompt = batch["prompt"]
        training_batch.prompt_embed = batch["prompt_embed"].to(
            get_local_torch_device(), dtype=torch.bfloat16
        )
        training_batch.prompt_mask = batch["prompt_mask"].to(
            get_local_torch_device(), dtype=torch.bfloat16
        )
        training_batch.vision_states = batch["vision_states"].to(
            get_local_torch_device(), dtype=torch.bfloat16
        )
        training_batch.extra_kwargs = extra_kwargs
        training_batch.w2c = batch["w2c"].to(
            get_local_torch_device(), dtype=torch.bfloat16
        )
        training_batch.intrinsic = batch["intrinsic"].to(
            get_local_torch_device(), dtype=torch.bfloat16
        )
        training_batch.action = batch["action"].to(
            get_local_torch_device(), dtype=torch.bfloat16
        )

        return training_batch

    def _build_attention_metadata(self, training_batch: TrainingBatch) -> TrainingBatch:
        latents_shape = training_batch.raw_latent_shape
        patch_size = self.training_args.pipeline_config.dit_config.patch_size
        current_vsa_sparsity = training_batch.current_vsa_sparsity
        assert latents_shape is not None
        assert training_batch.timesteps is not None
        if vsa_available and envs.FASTVIDEO_ATTENTION_BACKEND == "VIDEO_SPARSE_ATTN":
            training_batch.attn_metadata = VideoSparseAttentionMetadataBuilder(  # type: ignore
            ).build(  # type: ignore
                raw_latent_shape=latents_shape[2:5],
                current_timestep=training_batch.timesteps,
                patch_size=patch_size,
                VSA_sparsity=current_vsa_sparsity,
                device=get_local_torch_device(),
            )
        else:
            training_batch.attn_metadata = None

        return training_batch

    def _build_rope_idx(self, training_batch: TrainingBatch) -> TrainingBatch:
        rank_in_sp_group = get_sp_parallel_rank()
        per_sp_seq_length = (
            training_batch.latents.shape[2] * training_batch.per_seq_length
        )

        training_batch.current_start = per_sp_seq_length * rank_in_sp_group
        training_batch.current_end = (
            per_sp_seq_length * rank_in_sp_group + per_sp_seq_length
        )
        return training_batch

    def _clip_grad_norm(self, training_batch: TrainingBatch) -> TrainingBatch:
        max_grad_norm = self.training_args.max_grad_norm

        # TODO(will): perhaps move this into transformer api so that we can do
        # the following:
        # grad_norm = transformer.clip_grad_norm_(max_grad_norm)
        if max_grad_norm is not None:
            model_parts = [self.transformer]
            grad_norm = clip_grad_norm_while_handling_failing_dtensor_cases(
                [p for m in model_parts for p in m.parameters()],
                max_grad_norm,
                foreach=None,
            )
            assert grad_norm is not float("nan") or grad_norm is not float("inf")
            grad_norm = grad_norm.item() if grad_norm is not None else 0.0
        else:
            grad_norm = 0.0
        training_batch.grad_norm = grad_norm
        return training_batch

    def _clip_grad_norm(self, training_batch: TrainingBatch) -> TrainingBatch:
        max_grad_norm = self.training_args.max_grad_norm

        # TODO(will): perhaps move this into transformer api so that we can do
        # the following:
        # grad_norm = transformer.clip_grad_norm_(max_grad_norm)
        if max_grad_norm is not None:
            model_parts = [self.transformer]
            grad_norm = clip_grad_norm_while_handling_failing_dtensor_cases(
                [p for m in model_parts for p in m.parameters()],
                max_grad_norm,
                foreach=None,
            )
            assert grad_norm is not float("nan") or grad_norm is not float("inf")
            grad_norm = grad_norm.item() if grad_norm is not None else 0.0
        else:
            grad_norm = 0.0
        training_batch.grad_norm = grad_norm
        return training_batch

    def _create_sample_kv_cache(self):
        _kv_cache = []
        _kv_cache_neg = []
        transformer_num_layers = len(self.transformer.double_blocks)
        for i in range(transformer_num_layers):
            _kv_cache.append(
                {"k_vision": None, "v_vision": None, "k_txt": None, "v_txt": None}
            )
            _kv_cache_neg.append(
                {"k_vision": None, "v_vision": None, "k_txt": None, "v_txt": None}
            )

        return {"positive": _kv_cache, "negative": _kv_cache_neg}

    def _build_kv_cache_from_previous_chunks(
        self,
        current_kv_cache,
        latents_curr,
        training_batch,
        generate_latent_num,
        update_latent_num,
        stabilization_level,
        negative=False,
    ):
        """Build KV cache from previous chunks to accelerate current chunk sampling."""
        transformer_dtype = self.transformer.dtype
        device = latents_curr.device
        # Calculate the range for KV cache building (previous chunks)
        kv_cache_latent_num = generate_latent_num - update_latent_num
        # kv_cache_latent_num = 4

        if generate_latent_num >= 20:
            selected_frame_indices = select_aligned_memory_frames_context_per_chunk_w_latent_sink_fov_refine_hunyuan(
                training_batch.w2c[:, :generate_latent_num, :, :][0]
                .to(torch.float32)
                .cpu()
                .detach()
                .numpy(),
                generate_latent_num - 4,
                memory_frames=20,
                temporal_context_size=12,
                pred_latent_size=4,
                points_local=self.points_local.to(device),
                device=device,
            )
        else:
            selected_frame_indices = list(range(0, kv_cache_latent_num))

        if kv_cache_latent_num <= 0:
            return current_kv_cache  # No previous chunks to cache

        # Extract latents for KV cache building (previous chunks only)
        latents_for_cache = latents_curr[:, :, selected_frame_indices, :, :]
        cond_latents_for_cache = training_batch.cond_latents[
            :, :, selected_frame_indices, :, :
        ]
        latents_input_for_cache = torch.cat(
            [latents_for_cache, cond_latents_for_cache], dim=1
        )

        action_for_cache = (
            training_batch.action[:, selected_frame_indices] if self.action else None
        )

        # Create timestep input for cache building (use stabilization level for previous chunks)
        timestep_for_cache = torch.full(
            (len(selected_frame_indices),),
            stabilization_level - 1,
            device=latents_curr.device,
            dtype=torch.long,
        )

        viewmats_for_cache = training_batch.w2c[:, selected_frame_indices].to(device)
        Ks_for_cache = training_batch.intrinsic[:, selected_frame_indices].to(device)
        action_for_cache = training_batch.action[:, selected_frame_indices].to(device)

        for kv_cache_layer in current_kv_cache["positive"]:
            kv_cache_layer["k_vision"] = None
            kv_cache_layer["v_vision"] = None

        # Prepare inputs for KV cache building
        cache_kwargs = {
            "hidden_states": latents_input_for_cache,
            "timestep": timestep_for_cache,
            "timestep_r": None,
            "return_dict": False,
            "mask_type": "i2v",
            "action": action_for_cache,
            "viewmats": viewmats_for_cache,
            "Ks": Ks_for_cache,
            "kv_cache": current_kv_cache["positive"],
            "cache_vision": True,
            "rope_temporal_size": latents_for_cache.shape[2],
            "start_rope_start_idx": 0,
        }

        with (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True),
            torch.no_grad(),
        ):
            current_kv_cache["positive"] = self.transformer(
                txt_branch=False,
                input_dict=cache_kwargs,
            )

        return current_kv_cache, selected_frame_indices

    def _split_kv_cache(self, kv_cache, idx):
        new_kv_cache = []

        for layer_idx in range(len(kv_cache)):
            new_kv_cache.append(
                {
                    "k": kv_cache[layer_idx]["k"][idx].unsqueeze(0).detach().clone(),
                    "v": kv_cache[layer_idx]["v"][idx].unsqueeze(0).detach().clone(),
                }
            )

        return new_kv_cache

    def flux_step(
        self,
        model_output: torch.Tensor,
        latents: torch.Tensor,
        eta: float,
        sigmas: torch.Tensor,
        index: int,
        prev_sample: torch.Tensor,
        grpo: bool,
        sde_solver: bool,
    ):
        sigma = sigmas[index]
        dsigma = sigmas[index + 1] - sigma
        prev_sample_mean = latents + dsigma * model_output

        pred_original_sample = latents - sigma * model_output

        delta_t = sigma - sigmas[index + 1]
        std_dev_t = eta * math.sqrt(delta_t)

        if sde_solver:
            score_estimate = -(latents - pred_original_sample * (1 - sigma)) / sigma**2
            log_term = -0.5 * eta**2 * score_estimate
            prev_sample_mean = prev_sample_mean + log_term * dsigma

            prev_sample_mean = (
                prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t
            )

        return prev_sample_mean, pred_original_sample

    def _shard_latents_dim_across_sp(self, latents, timesteps, action):
        latents = shard_latents_dim_across_sp(latents, total_dim=latents.dim())
        timesteps = shard_latents_dim_across_sp(timesteps, total_dim=timesteps.dim())
        if self.action:
            action = shard_latents_dim_across_sp(action, total_dim=action.dim())
        return latents, timesteps, action

    @torch.no_grad()
    def _sample_model_ode(
        self,
        training_batch: TrainingBatch,
        selected_chunk_id,
        noise_latents,
        sampling_steps=None,
    ) -> TrainingBatch:
        self.transformer.eval()

        chunk_latent_num = (
            self.training_args.single_chunk_size
        )  # hard code right now, can be changed later.  ->  done
        stabilization_level = 15

        # 配置sample schedule
        if sampling_steps is None:
            sampling_steps = self.training_args.sampling_steps

        sampling_batch_size = self.training_args.sampling_batch_size

        kv_cache = self._create_sample_kv_cache()

        t_expand_txt = (
            torch.tensor([0]).to(get_local_torch_device()).to(noise_latents.dtype)
        )
        input_dict = {
            "timestep_txt": t_expand_txt,
            "text_states": training_batch.prompt_embed,
            "encoder_attention_mask": training_batch.prompt_mask,
            "vision_states": training_batch.vision_states,
            "mask_type": "i2v",
            "extra_kwargs": training_batch.extra_kwargs,
            "kv_cache": kv_cache["positive"],
            "cache_txt": True,
        }
        with (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True),
            torch.no_grad(),
        ):
            kv_cache["positive"] = self.transformer(
                txt_branch=True,
                input_dict=input_dict,
            )

        sigma_schedule = torch.linspace(1, 0, sampling_steps + 1)
        shift = 5
        sigma_schedule = (shift * sigma_schedule) / (1 + (shift - 1) * sigma_schedule)

        for chunk_i in tqdm(
            range(0, selected_chunk_id - 1), desc="ODE Chunk", leave=True
        ):
            torch.cuda.empty_cache()

            generate_latent_num = (chunk_i + 1) * chunk_latent_num

            latents_curr = noise_latents[:, :, :generate_latent_num, :, :]
            cond_latent_curr = training_batch.cond_latents[
                :, :, :generate_latent_num, :, :
            ]

            w2c_curr = training_batch.w2c[:, :generate_latent_num, :, :]
            intrinsic_curr = training_batch.intrinsic[:, :generate_latent_num, :, :]
            action_curr = training_batch.action[:, :generate_latent_num]

            update_latent_num = chunk_latent_num

            selected_frame_indices = []
            if chunk_i > 0:
                kv_cache, selected_frame_indices = (
                    self._build_kv_cache_from_previous_chunks(
                        kv_cache,
                        latents_curr,
                        training_batch,
                        generate_latent_num,
                        update_latent_num,
                        stabilization_level,
                    )
                )

            for i in range(0, sampling_steps):
                sigma = sigma_schedule[i]
                timestep_value = int(sigma * 1000)

                timestep_input = torch.full(
                    (
                        1,
                        update_latent_num,
                    ),
                    timestep_value,
                    device=latents_curr.device,
                    dtype=torch.long,
                )

                latent_concat_curr = torch.cat([latents_curr, cond_latent_curr], dim=1)

                input_dict = {
                    "hidden_states": latent_concat_curr[
                        :, :, -update_latent_num:, :, :
                    ],
                    "timestep": timestep_input,
                    "timestep_r": None,
                    "return_dict": False,
                    "mask_type": "i2v",
                    "action": action_curr[:, -update_latent_num:],
                    "viewmats": w2c_curr[:, -update_latent_num:, :, :],
                    "Ks": intrinsic_curr[:, -update_latent_num:, :, :],
                    "kv_cache": kv_cache["positive"],
                    "cache_vision": False,
                    "rope_temporal_size": len(selected_frame_indices)
                    + update_latent_num,
                    "start_rope_start_idx": len(selected_frame_indices),
                }

                with (
                    torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16, enabled=True
                    ),
                    torch.no_grad(),
                ):
                    model_pred = self.transformer(
                        txt_branch=False,
                        input_dict=input_dict,
                    )[0]

                z, pred_original = self.flux_step(
                    model_pred[:, :, -update_latent_num:, :, :].to(torch.float32),
                    latents_curr[:, :, -update_latent_num:, :, :].to(torch.float32),
                    eta=0.0,
                    sigmas=sigma_schedule,
                    index=i,
                    prev_sample=None,
                    grpo=False,
                    sde_solver=False,
                )
                latents_curr[:, :, -update_latent_num:, :, :] = z

            latents_curr[:, :, -update_latent_num:, :, :] = pred_original

        return latents_curr

    @torch.no_grad()
    def _sample_reference_model(self, training_batch: TrainingBatch) -> TrainingBatch:

        # if training_batch.current_timestep % 8 == 0:
        #     self.transformer = self.ema_generator.replace_parameters_with_ema(self.transformer)

        with self.ema_generator.apply_policy_shadow_to_model(self.transformer):
            self.transformer.eval()

            chunk_latent_num = (
                self.training_args.single_chunk_size
            )  # hard code right now, can be changed later.  ->  done
            stabilization_level = 15

            # 配置sample schedule
            sampling_steps = self.training_args.sampling_steps
            sampling_batch_size = self.training_args.sampling_batch_size

            sigma_schedule = torch.linspace(1, 0, self.training_args.sampling_steps + 1)
            shift = 5
            sigma_schedule = (shift * sigma_schedule) / (
                1 + (shift - 1) * sigma_schedule
            )

            latents = training_batch.latents
            bsz, latent_channels, _, latent_h, latent_w = latents.shape

            latent_t = self.training_args.window_frames
            # 随机选择一个chunk作为优化目标
            # 计算当前步数对应的chunk_id，从1逐渐增加到7
            # MIN to MAX

            # Get chunk selection parameters from training_args
            min_chunk_id = self.training_args.min_chunk_id
            max_chunk_id = self.training_args.max_chunk_id
            chunk_strategy = self.training_args.chunk_selection_strategy

            if chunk_strategy == "min2max":
                selected_chunk_id = min_chunk_id + (
                    (training_batch.current_timestep - 1) % max_chunk_id
                )
            elif chunk_strategy == "max2min":
                selected_chunk_id = max_chunk_id - (
                    (training_batch.current_timestep - 1)
                    % (max_chunk_id - min_chunk_id + 1)
                )
            else:
                raise ValueError(
                    f'Invalid chunk_selection_strategy: {chunk_strategy}, must be one of "min2max", "max2min", or "random".'
                )

            training_batch.selected_chunk_id = selected_chunk_id

            noise = torch.randn(
                (1, latent_channels, latent_t, latent_h, latent_w),  # （c,t,h,w)
                device=latents.device,
                dtype=latents.dtype,
            )
            noise = self.gpu_group.all_gather(noise, dim=0)

            context_latents = None
            context_num = 0
            if selected_chunk_id > 1:
                context_latents = self._sample_model_ode(
                    training_batch, selected_chunk_id, noise
                )
                context_num = context_latents.shape[2]

            training_batch.sample_kwargs = {}

            if selected_chunk_id > 1:
                if (
                    hasattr(self.vae.config, "shift_factor")
                    and self.vae.config.shift_factor
                ):
                    context_latents = (
                        context_latents / self.vae.config.scaling_factor
                        + self.vae.config.shift_factor
                    )
                else:
                    context_latents = context_latents / self.vae.config.scaling_factor

                self.vae.to(self.device)
                context_video_frames = self.vae.decode(
                    context_latents.to(torch.float32), return_dict=False
                )[0]
                self.vae.to("cpu")
                context_video_frames = (
                    (context_video_frames / 2 + 0.5).clamp(0, 1).cpu().float()
                )
                context_video_frames = np.transpose(
                    context_video_frames[0], (1, 2, 3, 0)
                )
                context_video_frames = (
                    (context_video_frames * 255).numpy().astype(np.uint8)
                )
            else:
                context_video_frames = None

            kv_cache = self._create_sample_kv_cache()

            t_expand_txt = (
                torch.tensor([0]).to(get_local_torch_device()).to(noise.dtype)
            )
            input_dict = {
                "timestep_txt": t_expand_txt,
                "text_states": training_batch.prompt_embed,
                "encoder_attention_mask": training_batch.prompt_mask,
                "vision_states": training_batch.vision_states,
                "mask_type": "i2v",
                "extra_kwargs": training_batch.extra_kwargs,
                "kv_cache": kv_cache["positive"],
                "cache_txt": True,
            }
            with (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True),
                torch.no_grad(),
            ):
                kv_cache["positive"] = self.transformer(
                    txt_branch=True,
                    input_dict=input_dict,
                )

            if selected_chunk_id > 1:
                noise_latents = torch.cat(
                    [context_latents, noise[:, :, context_num:, :, :]], dim=2
                )
                start_chunk_id = selected_chunk_id - 1
                generate_latent_num = selected_chunk_id * chunk_latent_num
                kv_cache, selected_frame_indices = (
                    self._build_kv_cache_from_previous_chunks(
                        kv_cache,
                        noise_latents[:, :, :generate_latent_num, :, :],
                        training_batch,
                        generate_latent_num,
                        chunk_latent_num,
                        stabilization_level,
                    )
                )
                training_batch.kv_cache = kv_cache
            else:
                start_chunk_id = 0
                selected_frame_indices = []
                training_batch.kv_cache = kv_cache

            for sample_idx in tqdm(
                range(0, self.training_args.grpo_generation_num),
                desc="GRPO Batch Sampling",
                leave=False,
            ):
                new_noise = torch.randn_like(noise)

                if context_latents is not None:
                    noise_latents = torch.cat(
                        [context_latents, new_noise[:, :, context_num:, :, :]], dim=2
                    )
                else:
                    noise_latents = new_noise

                video_num_in_each_gpu = (
                    sampling_batch_size // self.training_args.gpu_para
                )
                rank_in_gpu_group = self.gpu_group.rank_in_group

                for chunk_i in range(start_chunk_id, selected_chunk_id):
                    generate_latent_num = (chunk_i + 1) * chunk_latent_num

                    latents_curr = noise_latents[:, :, :generate_latent_num, :, :]
                    cond_latent_curr = training_batch.cond_latents[
                        :, :, :generate_latent_num, :, :
                    ]

                    w2c_curr = training_batch.w2c[:, :generate_latent_num, :, :]
                    intrinsic_curr = training_batch.intrinsic[
                        :, :generate_latent_num, :, :
                    ]
                    action_curr = training_batch.action[:, :generate_latent_num]

                    update_latent_num = chunk_latent_num

                    for i in range(0, sampling_steps):
                        sigma = sigma_schedule[i]
                        timestep_value = int(sigma * 1000)

                        latent_concat_curr = torch.cat(
                            [latents_curr, cond_latent_curr], dim=1
                        )
                        timestep_input = torch.full(
                            (
                                1,
                                update_latent_num,
                            ),
                            timestep_value,
                            device=latents_curr.device,
                            dtype=torch.long,
                        )

                        if i == 0:
                            tmp_kwargs = {
                                "hidden_states": latent_concat_curr[
                                    :, :, -update_latent_num:, :, :
                                ],
                                "timestep": timestep_input,
                                "timestep_r": None,
                                "return_dict": False,
                                "mask_type": "i2v",
                                "action": action_curr[:, -update_latent_num:],
                                "viewmats": w2c_curr[:, -update_latent_num:, :, :],
                                "Ks": intrinsic_curr[:, -update_latent_num:, :, :],
                                "kv_cache": kv_cache["positive"],
                                "cache_vision": False,
                                "rope_temporal_size": len(selected_frame_indices)
                                + update_latent_num,
                                "start_rope_start_idx": len(selected_frame_indices),
                            }

                        input_dict = {
                            "hidden_states": latent_concat_curr[
                                :, :, -update_latent_num:, :, :
                            ],
                            "timestep": timestep_input,
                            "timestep_r": None,
                            "return_dict": False,
                            "mask_type": "i2v",
                            "action": action_curr[:, -update_latent_num:],
                            "viewmats": w2c_curr[:, -update_latent_num:, :, :],
                            "Ks": intrinsic_curr[:, -update_latent_num:, :, :],
                            "kv_cache": kv_cache["positive"],
                            "cache_vision": False,
                            "rope_temporal_size": len(selected_frame_indices)
                            + update_latent_num,
                            "start_rope_start_idx": len(selected_frame_indices),
                        }

                        with (
                            torch.autocast(
                                device_type="cuda", dtype=torch.bfloat16, enabled=True
                            ),
                            torch.no_grad(),
                        ):
                            model_pred = self.transformer(
                                txt_branch=False,
                                input_dict=input_dict,
                            )[0]

                        z, pred_original = self.flux_step(
                            model_pred[:, :, -update_latent_num:, :, :].to(
                                torch.float32
                            ),
                            latents_curr[:, :, -update_latent_num:, :, :].to(
                                torch.float32
                            ),
                            eta=0.0,
                            sigmas=sigma_schedule,
                            index=i,
                            prev_sample=None,
                            grpo=False,
                            sde_solver=False,
                        )

                        latents_curr[:, :, -update_latent_num:, :, :] = z

                    latents_curr[:, :, -update_latent_num:, :, :] = pred_original
                    tmp_kwargs["pred_latents"] = (
                        latents_curr[:, :, -chunk_latent_num:, :, :].detach().clone()
                    )

                with torch.no_grad():
                    if latents_curr.shape[2] > chunk_latent_num:
                        latents_curr = torch.cat(
                            [
                                latents_curr[:, :, :1, :, :],
                                latents_curr[:, :, -(chunk_latent_num + 2) :, :, :],
                            ],
                            dim=2,
                        )

                    if (
                        hasattr(self.vae.config, "shift_factor")
                        and self.vae.config.shift_factor
                    ):
                        latents = (
                            latents_curr / self.vae.config.scaling_factor
                            + self.vae.config.shift_factor
                        )
                    else:
                        latents = latents_curr / self.vae.config.scaling_factor

                    self.vae.to(self.device)
                    video_frames = self.vae.decode(
                        latents.to(torch.float32), return_dict=False
                    )[0]
                    self.vae.to("cpu")
                    video_frames = (video_frames / 2 + 0.5).clamp(0, 1).cpu().float()
                    video_frames = np.transpose(video_frames[0], (1, 2, 3, 0))
                    video_frames = (video_frames * 255).numpy().astype(np.uint8)

                    if context_video_frames is not None:
                        video_frames = np.concatenate(
                            [context_video_frames, video_frames[-16:]], axis=0
                        )

                    # Use generated_videos_dir from training_args
                    generated_videos_base = self.training_args.generated_videos_dir
                    if not generated_videos_base:
                        generated_videos_base = os.path.join(
                            self.training_args.output_dir, "generated_videos"
                        )
                    video_path = os.path.join(
                        generated_videos_base,
                        self.training_args.output_dir.split("/")[-1],
                        f"step_{training_batch.current_timestep}",
                    )

                    os.makedirs(video_path, exist_ok=True)

                    for local_video_idx in range(video_num_in_each_gpu):
                        # global_video_idx = self.rank_in_sp_group * video_num_in_each_gpu + sample_idx * sampling_batch_size + local_video_idx
                        global_video_idx = (
                            rank_in_gpu_group * video_num_in_each_gpu
                            + sample_idx * sampling_batch_size
                            + local_video_idx
                        )

                        save_video_path = os.path.join(
                            video_path,
                            f"{self.global_rank // self.training_args.gpu_para}_{global_video_idx}.mp4",
                        )
                        imageio.mimsave(save_video_path, video_frames)

                        absolute_path = os.path.abspath(save_video_path)

                        reward_info = self.reward_model.reward(
                            absolute_path,
                            gt_camera_pose=w2c_curr,
                            gt_action=action_curr[:, -update_latent_num:],
                            caption=training_batch.prompt,
                            interval=1,
                            update_latent_num=chunk_latent_num,
                        )
                        action_acc = reward_info["action_acc"]
                        fine_action_acc = reward_info["fine_action_acc"]
                        hpsv3_acc = reward_info["hpsv3_acc"]
                        hpsv3_quality_acc = reward_info["hpsv3_quality_acc"]
                        hpsv3_quality_drift_score = reward_info[
                            "hpsv3_quality_drift_score"
                        ]

                        action_reward = torch.tensor(action_acc).to(latents_curr)
                        fine_action_reward = torch.tensor(fine_action_acc).to(
                            latents_curr
                        )
                        hpsv3_reward = torch.tensor(hpsv3_acc).to(latents_curr)
                        hpsv3_quality_reward = torch.tensor(hpsv3_quality_acc).to(
                            latents_curr
                        )
                        hpsv3_quality_drift_reward = torch.tensor(
                            hpsv3_quality_drift_score
                        ).to(latents_curr)

                        new_filename = f"{self.global_rank // self.training_args.gpu_para}_{global_video_idx}_chunk_{selected_chunk_id}_action_{round(action_acc, 1)}_Faction_{round(fine_action_acc, 1)}_hpsv3_{round(hpsv3_acc, 1)}_quality_{round(hpsv3_quality_acc, 1)}_drift_{round(hpsv3_quality_drift_score, 1)}.mp4"
                        new_absolute_path = os.path.join(
                            os.path.dirname(absolute_path), new_filename
                        )
                        if os.path.exists(absolute_path):
                            os.rename(absolute_path, new_absolute_path)

                        training_batch.sample_kwargs[f"sample_{global_video_idx}"] = {
                            "pred_latents": tmp_kwargs["pred_latents"][local_video_idx]
                            .unsqueeze(0)
                            .detach()
                            .clone(),
                            "action": tmp_kwargs["action"] if self.action else None,
                            "viewmats": tmp_kwargs["viewmats"] if self.action else None,
                            "Ks": tmp_kwargs["Ks"] if self.action else None,
                            "rope_temporal_size": tmp_kwargs["rope_temporal_size"],
                            "start_rope_start_idx": tmp_kwargs["start_rope_start_idx"],
                            "cache_vision": tmp_kwargs["cache_vision"],
                            "return_dict": False,
                            "action_reward": action_reward,
                            "fine_action_reward": fine_action_reward,
                            "hpsv3_reward": hpsv3_reward,
                            "hpsv3_quality_reward": hpsv3_quality_reward,
                            "hpsv3_quality_drift_reward": hpsv3_quality_drift_reward,
                            "selected_chunk_id": selected_chunk_id,
                        }

                torch.cuda.empty_cache()
            training_batch.sigma_schedule = sigma_schedule
            training_batch.selected_chunk_id = selected_chunk_id

        return training_batch

    def _prepare_grpo_inputs(self, training_batch: TrainingBatch) -> TrainingBatch:

        action_rewards = []
        fine_action_rewards = []
        hpsv3_rewards = []
        hpsv3_quality_rewards = []
        hpsv3_quality_drift_rewards = []

        for sample_key in training_batch.sample_kwargs.keys():
            action_rewards.append(
                training_batch.sample_kwargs[sample_key]["action_reward"]
            )
            fine_action_rewards.append(
                training_batch.sample_kwargs[sample_key]["fine_action_reward"]
            )
            hpsv3_rewards.append(
                training_batch.sample_kwargs[sample_key]["hpsv3_reward"]
            )
            hpsv3_quality_rewards.append(
                training_batch.sample_kwargs[sample_key]["hpsv3_quality_reward"]
            )
            hpsv3_quality_drift_rewards.append(
                training_batch.sample_kwargs[sample_key]["hpsv3_quality_drift_reward"]
            )

        action_rewards = torch.tensor(action_rewards).to(self.device)
        fine_action_rewards = torch.tensor(fine_action_rewards).to(self.device)
        hpsv3_rewards = torch.tensor(hpsv3_rewards).to(self.device)
        hpsv3_quality_rewards = torch.tensor(hpsv3_quality_rewards).to(self.device)
        hpsv3_quality_drift_rewards = torch.tensor(hpsv3_quality_drift_rewards).to(
            self.device
        )

        world_group = get_world_group()

        group_action_rewards = self.gpu_group.all_gather(action_rewards, dim=0)
        group_action_rewards_mean = group_action_rewards.mean()
        group_action_rewards_std = group_action_rewards.std() + 1e-8
        world_action_rewards = world_group.all_gather(group_action_rewards, dim=0)
        world_action_rewards_std = world_group.all_gather(
            group_action_rewards_std.unsqueeze(0), dim=0
        )

        group_fine_action_rewards = self.gpu_group.all_gather(
            fine_action_rewards, dim=0
        )
        group_fine_action_rewards_mean = group_fine_action_rewards.mean()
        group_fine_action_rewards_std = group_fine_action_rewards.std() + 1e-8
        world_fine_action_rewards = world_group.all_gather(
            group_fine_action_rewards, dim=0
        )
        world_fine_action_rewards_std = world_group.all_gather(
            group_fine_action_rewards_std.unsqueeze(0), dim=0
        )

        group_hpsv3_rewards = self.gpu_group.all_gather(hpsv3_rewards, dim=0)
        group_hpsv3_rewards_mean = group_hpsv3_rewards.mean()
        group_hpsv3_rewards_std = group_hpsv3_rewards.std() + 1e-8
        world_hpsv3_rewards = world_group.all_gather(group_hpsv3_rewards, dim=0)
        world_hpsv3_rewards_std = world_group.all_gather(
            group_hpsv3_rewards_std.unsqueeze(0), dim=0
        )

        group_hpsv3_quality_rewards = self.gpu_group.all_gather(
            hpsv3_quality_rewards, dim=0
        )
        group_hpsv3_quality_rewards_mean = group_hpsv3_quality_rewards.mean()
        group_hpsv3_quality_rewards_std = group_hpsv3_quality_rewards.std() + 1e-8
        world_hpsv3_quality_rewards = world_group.all_gather(
            group_hpsv3_quality_rewards, dim=0
        )
        world_hpsv3_quality_rewards_std = world_group.all_gather(
            group_hpsv3_quality_rewards_std.unsqueeze(0), dim=0
        )

        group_hpsv3_quality_drift_rewards = self.gpu_group.all_gather(
            hpsv3_quality_drift_rewards, dim=0
        )
        group_hpsv3_quality_drift_rewards_mean = (
            group_hpsv3_quality_drift_rewards.mean()
        )
        group_hpsv3_quality_drift_rewards_std = (
            group_hpsv3_quality_drift_rewards.std() + 1e-8
        )
        world_hpsv3_quality_drift_rewards = world_group.all_gather(
            group_hpsv3_quality_drift_rewards, dim=0
        )
        world_hpsv3_quality_drift_rewards_std = world_group.all_gather(
            group_hpsv3_quality_drift_rewards_std.unsqueeze(0), dim=0
        )

        if self.training_args.std_type == "sample_max":
            group_action_rewards_std = world_action_rewards_std.max()
            group_fine_action_rewards_std = world_fine_action_rewards_std.max()
            group_hpsv3_rewards_std = world_hpsv3_rewards_std.max()
            group_hpsv3_quality_rewards_std = world_hpsv3_quality_rewards_std.max()
            group_hpsv3_quality_drift_rewards_std = (
                world_hpsv3_quality_drift_rewards_std.max()
            )
        elif self.training_args.std_type == "global":
            group_action_rewards_std = world_action_rewards.std() + 1e-8
            group_fine_action_rewards_std = world_fine_action_rewards.std() + 1e-8
            group_hpsv3_rewards_std = world_hpsv3_rewards.std() + 1e-8
            group_hpsv3_quality_rewards_std = world_hpsv3_quality_rewards.std() + 1e-8
            group_hpsv3_quality_drift_rewards_std = (
                world_hpsv3_quality_drift_rewards.std() + 1e-8
            )
        elif self.training_args.std_type == "sample":
            group_action_rewards_std = group_action_rewards_std
            group_fine_action_rewards_std = group_fine_action_rewards_std
            group_hpsv3_rewards_std = group_hpsv3_rewards_std
            group_hpsv3_quality_rewards_std = group_hpsv3_quality_rewards_std
            group_hpsv3_quality_drift_rewards_std = (
                group_hpsv3_quality_drift_rewards_std
            )

        all_action_rewards = world_group.all_gather(action_rewards, dim=0)
        all_fine_action_rewards = world_group.all_gather(fine_action_rewards, dim=0)
        all_hpsv3_rewards = world_group.all_gather(hpsv3_rewards, dim=0)
        all_hpsv3_quality_rewards = world_group.all_gather(hpsv3_quality_rewards, dim=0)
        all_hpsv3_quality_drift_rewards = world_group.all_gather(
            hpsv3_quality_drift_rewards, dim=0
        )

        training_batch.action_reward_mean = all_action_rewards.mean()
        training_batch.fine_action_reward_mean = all_fine_action_rewards.mean()
        training_batch.hpsv3_reward_mean = all_hpsv3_rewards.mean()
        training_batch.hpsv3_quality_reward_mean = all_hpsv3_quality_rewards.mean()
        training_batch.hpsv3_quality_drift_reward_mean = (
            all_hpsv3_quality_drift_rewards.mean()
        )

        if self.training_args.action_reward_type == "fine_action":
            action_advantages = []
            for sample_key in training_batch.sample_kwargs.keys():
                fine_action_reward = training_batch.sample_kwargs[sample_key][
                    "fine_action_reward"
                ]
                action_advantages.append(
                    (fine_action_reward - group_fine_action_rewards_mean)
                    / group_fine_action_rewards_std
                )
                training_batch.sample_kwargs[sample_key]["action_advantages"] = (
                    fine_action_reward - group_fine_action_rewards_mean
                ) / group_fine_action_rewards_std
            action_advantages = torch.tensor(action_advantages)
        elif self.training_args.action_reward_type == "action":
            action_advantages = []
            for sample_key in training_batch.sample_kwargs.keys():
                action_reward = training_batch.sample_kwargs[sample_key][
                    "action_reward"
                ]
                action_advantages.append(
                    (action_reward - group_action_rewards_mean)
                    / group_action_rewards_std
                )
                training_batch.sample_kwargs[sample_key]["action_advantages"] = (
                    action_reward - group_action_rewards_mean
                ) / group_action_rewards_std
            action_advantages = torch.tensor(action_advantages)

        hpsv3_advantages = []
        for sample_key in training_batch.sample_kwargs.keys():
            hpsv3_reward = training_batch.sample_kwargs[sample_key]["hpsv3_reward"]
            hpsv3_advantages.append(
                (hpsv3_reward - group_hpsv3_rewards_mean) / group_hpsv3_rewards_std
            )
            training_batch.sample_kwargs[sample_key]["hpsv3_advantages"] = (
                hpsv3_reward - group_hpsv3_rewards_mean
            ) / group_hpsv3_rewards_std
        hpsv3_advantages = torch.tensor(hpsv3_advantages)

        hpsv3_quality_advantages = []
        for sample_key in training_batch.sample_kwargs.keys():
            hpsv3_quality_reward = training_batch.sample_kwargs[sample_key][
                "hpsv3_quality_reward"
            ]
            hpsv3_quality_advantages.append(
                (hpsv3_quality_reward - group_hpsv3_quality_rewards_mean)
                / group_hpsv3_quality_rewards_std
            )
            training_batch.sample_kwargs[sample_key]["hpsv3_quality_advantages"] = (
                hpsv3_quality_reward - group_hpsv3_quality_rewards_mean
            ) / group_hpsv3_quality_rewards_std
        hpsv3_quality_advantages = torch.tensor(hpsv3_quality_advantages)

        hpsv3_quality_drift_advantages = []
        for sample_key in training_batch.sample_kwargs.keys():
            hpsv3_quality_drift_reward = training_batch.sample_kwargs[sample_key][
                "hpsv3_quality_drift_reward"
            ]
            hpsv3_quality_drift_advantages.append(
                (hpsv3_quality_drift_reward - group_hpsv3_quality_drift_rewards_mean)
                / group_hpsv3_quality_drift_rewards_std
            )
            training_batch.sample_kwargs[sample_key][
                "hpsv3_quality_drift_advantages"
            ] = (
                hpsv3_quality_drift_reward - group_hpsv3_quality_drift_rewards_mean
            ) / group_hpsv3_quality_drift_rewards_std
        hpsv3_quality_drift_advantages = torch.tensor(hpsv3_quality_drift_advantages)

        overall_reward = (
            self.training_args.action_reward_weight * action_advantages
            + self.training_args.hpsv3_reward_weight * hpsv3_advantages
            + self.training_args.hpsv3_quality_reward_weight * hpsv3_quality_advantages
            + self.training_args.hpsv3_quality_drift_reward_weight * hpsv3_quality_drift_advantages
        )

        sorted_indices = torch.argsort(overall_reward)

        top_indices = sorted_indices[
            -(self.training_args.bestofn // 2) // self.gpu_group.world_size :
        ]
        bottom_indices = sorted_indices[
            : (self.training_args.bestofn // 2) // self.gpu_group.world_size
        ]
        selected_indices = torch.cat([top_indices, bottom_indices])

        shuffled_order = torch.randperm(
            len(selected_indices), device=selected_indices.device
        )
        selected_indices = selected_indices[shuffled_order]
        training_batch.sample_kwargs["shuffled_sample_keys"] = [
            list(training_batch.sample_kwargs.keys())[idx] for idx in selected_indices
        ]

        return training_batch

    def _nft_forward_and_compute_loss(
        self, training_batch: TrainingBatch
    ) -> TrainingBatch:
        self.transformer.train()
        if vsa_available and envs.FASTVIDEO_ATTENTION_BACKEND == "VIDEO_SPARSE_ATTN":
            assert training_batch.attn_metadata is not None
        else:
            assert training_batch.attn_metadata is None

        with set_forward_context(
            current_timestep=training_batch.current_timestep,
            attn_metadata=training_batch.attn_metadata,
        ):
            sigma_schedule = torch.linspace(1, 0, self.training_args.sampling_steps + 1)
            shift = 5
            sigma_schedule = (shift * sigma_schedule) / (
                1 + (shift - 1) * sigma_schedule
            )
            training_num = int(
                self.training_args.sampling_steps
                * self.training_args.train_timestep_fraction
            )
            sigma_schedule = sigma_schedule[:-1]

            print(
                f"sample_for_training: {len(training_batch.sample_kwargs['shuffled_sample_keys'])}"
            )
            for sample_idx, sample_key in tqdm(
                list(enumerate(training_batch.sample_kwargs["shuffled_sample_keys"])),
                desc="NFT Training Over Samples",
                leave=False,
            ):
                train_sigma_schedule = random.sample(
                    list(sigma_schedule.cpu().numpy()), training_num
                )
                random.shuffle(train_sigma_schedule)

                for timestep_idx in tqdm(
                    range(len(train_sigma_schedule)),
                    desc="NFT Training Over Timesteps",
                    leave=False,
                ):
                    adv_clip_max = self.training_args.adv_clip_max
                    sigma = train_sigma_schedule[timestep_idx]

                    x0 = training_batch.sample_kwargs[sample_key]["pred_latents"]
                    noise = torch.randn_like(x0)

                    device = training_batch.sample_kwargs[sample_key]["viewmats"].device
                    dtype = training_batch.sample_kwargs[sample_key]["viewmats"].dtype

                    update_latent_num = self.training_args.single_chunk_size

                    latent_num = x0.shape[2]
                    stabilization_level = 15
                    timestep_value = int(sigma * 1000)
                    timestep_input = torch.full(
                        (latent_num,),
                        stabilization_level - 1,
                        device=get_local_torch_device(),
                        dtype=torch.long,
                    )
                    timestep_input[-update_latent_num:] = timestep_value

                    noisy_latents = x0.detach().clone()

                    noisy_latents = (1 - sigma) * x0 + sigma * noise
                    noisy_latents = noisy_latents.to(device).to(dtype)

                    start_idx = training_batch.sample_kwargs[sample_key][
                        "start_rope_start_idx"
                    ]
                    end_idx = training_batch.sample_kwargs[sample_key][
                        "rope_temporal_size"
                    ]

                    cond_latent_curr = training_batch.cond_latents[
                        :, :, start_idx:end_idx, :, :
                    ]

                    latents_for_model = torch.cat(
                        [noisy_latents, cond_latent_curr], dim=1
                    )

                    input_dict = {
                        "hidden_states": latents_for_model,
                        "timestep": timestep_input,
                        "timestep_r": None,
                        "return_dict": False,
                        "mask_type": "i2v",
                        "action": training_batch.sample_kwargs[sample_key]["action"]
                        if self.action
                        else None,
                        "viewmats": training_batch.sample_kwargs[sample_key][
                            "viewmats"
                        ][:, -update_latent_num:, :, :],
                        "Ks": training_batch.sample_kwargs[sample_key]["Ks"][
                            :, -update_latent_num:, :, :
                        ],
                        "kv_cache": training_batch.kv_cache["positive"],
                        "cache_vision": False,
                        "rope_temporal_size": training_batch.sample_kwargs[sample_key][
                            "rope_temporal_size"
                        ],
                        "start_rope_start_idx": training_batch.sample_kwargs[
                            sample_key
                        ]["start_rope_start_idx"],
                    }

                    with torch.no_grad():
                        with self.ema_generator.apply_policy_shadow_to_model(
                            self.transformer
                        ):
                            model_pred_old = model_pred = self.transformer(
                                txt_branch=False,
                                input_dict=input_dict,
                            )[0]

                    model_pred_old = model_pred = self.transformer(
                        txt_branch=False,
                        input_dict=input_dict,
                    )[0]

                    action_weight = self.training_args.action_reward_weight
                    hpsv3_weight = self.training_args.hpsv3_reward_weight
                    hpsv3_quality_weight = (
                        self.training_args.hpsv3_quality_reward_weight
                    )
                    hpsv3_quality_drift_weight = (
                        self.training_args.hpsv3_quality_drift_reward_weight
                    )

                    sum_weights = (
                        action_weight
                        + hpsv3_weight
                        + hpsv3_quality_weight
                        + hpsv3_quality_drift_weight
                    )

                    action_weight_ = action_weight / sum_weights
                    hpsv3_weight_ = hpsv3_weight / sum_weights
                    hpsv3_quality_weight_ = hpsv3_quality_weight / sum_weights
                    hpsv3_quality_drift_weight_ = (
                        hpsv3_quality_drift_weight / sum_weights
                    )

                    total_advantages = (
                        action_weight_
                        * training_batch.sample_kwargs[sample_key]["action_advantages"]
                        + hpsv3_weight_
                        * training_batch.sample_kwargs[sample_key]["hpsv3_advantages"]
                        + hpsv3_quality_weight_
                        * training_batch.sample_kwargs[sample_key][
                            "hpsv3_quality_advantages"
                        ]
                        + hpsv3_quality_drift_weight_
                        * training_batch.sample_kwargs[sample_key][
                            "hpsv3_quality_drift_advantages"
                        ]
                    )

                    total_advantages = torch.clamp(
                        total_advantages,
                        -adv_clip_max,
                        adv_clip_max,
                    )
                    normalized_advantages_clip = (
                        total_advantages / adv_clip_max
                    ) / 2.0 + 0.5
                    r = torch.clamp(normalized_advantages_clip, 0, 1)

                    positive_prediction = model_pred
                    negative_prediction = 2 * model_pred_old.detach() - model_pred

                    positive_x0 = (
                        noisy_latents[:, :, -update_latent_num:, :, :]
                        - sigma * positive_prediction[:, :, -update_latent_num:, :, :]
                    )
                    x0 = x0[:, :, -update_latent_num:, :, :]
                    with torch.no_grad():
                        weight_factor = (
                            torch.abs(positive_x0.double() - x0.double())
                            .mean(dim=tuple(range(1, positive_x0.ndim)), keepdim=True)
                            .clip(min=0.00001)
                        )
                    positive_loss = ((positive_x0 - x0) ** 2 / weight_factor).mean(
                        dim=tuple(range(1, positive_x0.ndim))
                    )

                    negative_x0 = (
                        noisy_latents[:, :, -update_latent_num:, :, :]
                        - sigma * negative_prediction[:, :, -update_latent_num:, :, :]
                    )
                    with torch.no_grad():
                        weight_factor = (
                            torch.abs(negative_x0.double() - x0.double())
                            .mean(dim=tuple(range(1, negative_x0.ndim)), keepdim=True)
                            .clip(min=0.00001)
                        )
                    negative_loss = ((negative_x0 - x0) ** 2 / weight_factor).mean(
                        dim=tuple(range(1, negative_x0.ndim))
                    )

                    policy_loss = r * positive_loss + (1.0 - r) * negative_loss

                    final_loss = policy_loss / (
                        self.training_args.gradient_accumulation_steps
                    )

                    # print({
                    #     'sample_idx': sample_idx,
                    #     'timestep_idx': timestep_idx,
                    #     'action_advantages': round(training_batch.sample_kwargs[sample_key]['action_advantages'].item(), 2),
                    #     'hpsv3_advantages': round(training_batch.sample_kwargs[sample_key]['hpsv3_advantages'].item(), 2),
                    #     'total_advantages': round(total_advantages.item(), 2),
                    #     'r': round(r.item(), 2),
                    #     'policy_loss': round(policy_loss.item(), 2),
                    #     'final_loss': round(final_loss.item(), 2),
                    # })
                    # print(f'output_dir: {self.training_args.output_dir}')

                    final_loss.backward()
                    avg_loss = final_loss.detach().clone()

                    world_group = get_world_group()
                    avg_loss = world_group.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
                    training_batch.total_loss += avg_loss.item()

                if (
                    sample_idx + 1
                ) % self.training_args.gradient_accumulation_steps == 0:
                    training_batch = self._clip_grad_norm(training_batch)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    self.ema_generator.update_ckpt_shadow(self.transformer)
                    self.ema_generator.update_policy_shadow(
                        self.transformer, training_batch.current_timestep
                    )

                    training_batch.samples_grad_norm[sample_key] = (
                        training_batch.grad_norm
                    )

        return training_batch

    def _clear_cache(self, training_batch: TrainingBatch) -> TrainingBatch:
        del training_batch
        torch.cuda.empty_cache()

    def train_one_step(self, training_batch: TrainingBatch) -> TrainingBatch:
        training_batch = self._prepare_training(training_batch)

        training_batch = self._get_next_batch(training_batch)

        training_batch = self._sample_reference_model(training_batch)

        training_batch = self._prepare_grpo_inputs(training_batch)

        training_batch = self._nft_forward_and_compute_loss(training_batch)

        self._clear_cache(training_batch)

        training_batch.total_loss = training_batch.total_loss
        training_batch.grad_norm = training_batch.grad_norm
        return training_batch

    def _resume_from_checkpoint(self) -> None:
        logger.info(
            "Loading checkpoint from %s", self.training_args.resume_from_checkpoint
        )
        resumed_step = load_checkpoint(
            self.transformer,
            self.global_rank,
            self.training_args.resume_from_checkpoint,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
            self.noise_random_generator,
        )
        if resumed_step > 0:
            self.init_steps = resumed_step
            logger.info("Successfully resumed from step %s", resumed_step)
        else:
            logger.warning("Failed to load checkpoint, starting from step 0")
            self.init_steps = 0

    def get_task_mask(self, task_type, latent_target_length):
        if task_type == "t2v":
            mask = torch.zeros(latent_target_length)
        elif task_type == "i2v":
            mask = torch.zeros(latent_target_length)
            mask[0] = 1.0
        else:
            raise ValueError(f"{task_type} is not supported !")
        return mask

    def _prepare_cond_latents(self, task_type, cond_latents, latents, multitask_mask):
        """Prepare conditional latents and mask for multitask training.

        Args:
            task_type: Type of task ("i2v" or "t2v").
            cond_latents: Conditional latents tensor.
            latents: Main latents tensor.
            multitask_mask: Multitask mask tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - latents_concat: Concatenated conditional latents.
                - mask_concat: Concatenated mask tensor.
        """
        latents_concat = None
        mask_concat = None

        if cond_latents is not None and task_type == "i2v":
            latents_concat = cond_latents.repeat(1, 1, latents.shape[2], 1, 1)
            latents_concat[:, :, 1:, :, :] = 0.0
        else:
            latents_concat = torch.zeros(
                latents.shape[0],
                latents.shape[1],
                latents.shape[2],
                latents.shape[3],
                latents.shape[4],
            ).to(latents.device)

        mask_zeros = torch.zeros(
            latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4]
        )
        mask_ones = torch.ones(
            latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4]
        )
        mask_concat = merge_tensor_by_mask(
            mask_zeros.cpu(), mask_ones.cpu(), mask=multitask_mask.cpu(), dim=2
        ).to(device=latents.device)

        cond_latents = torch.concat([latents_concat, mask_concat], dim=1)

        return cond_latents

    def _eval(self, step: int) -> None:
        with torch.no_grad():
            action_acc = []
            hpsv3_acc = []
            hpsv3_quality_acc = []
            hpsv3_drift_score = []

            with self.ema_generator.apply_ckpt_shadow_to_model(self.transformer):
                self.vae.eval()
                self.transformer.eval()

                eval_batch = TrainingBatch()

                sampling_steps = 20
                latent_t = 32

                chunk_latent_num = self.training_args.single_chunk_size
                video_chunk_num = latent_t // chunk_latent_num

                for batch_idx, batch in enumerate(self.eval_dataloader):
                    latents = batch["latent"]
                    prompt = batch["prompt"]
                    w2c = batch["w2c"]
                    action = batch["action"]

                    bsz, latent_channels, _, latent_h, latent_w = latents.shape
                    noise = torch.randn(
                        (
                            1,
                            latent_channels,
                            latent_t,
                            latent_h,
                            latent_w,
                        ),  # （c,t,h,w)
                        device=get_local_torch_device(),
                        dtype=latents.dtype,
                    )

                    extra_kwargs = {
                        "byt5_text_states": batch["byt5_text_states"].to(
                            get_local_torch_device(), dtype=torch.bfloat16
                        ),
                        "byt5_text_mask": batch["byt5_text_mask"].to(
                            get_local_torch_device(), dtype=torch.bfloat16
                        ),
                    }
                    multitask_mask = self.get_task_mask("i2v", batch["latent"].shape[2])
                    cond_latents = self._prepare_cond_latents(
                        "i2v", batch["image_cond"], batch["latent"], multitask_mask
                    )

                    eval_batch.cond_latents = cond_latents.to(
                        get_local_torch_device(), dtype=torch.bfloat16
                    )
                    eval_batch.latents_concat = torch.concat(
                        [batch["latent"], cond_latents], dim=1
                    ).to(get_local_torch_device(), dtype=torch.bfloat16)
                    eval_batch.prompt_embed = batch["prompt_embed"].to(
                        get_local_torch_device(), dtype=torch.bfloat16
                    )
                    eval_batch.prompt_mask = batch["prompt_mask"].to(
                        get_local_torch_device(), dtype=torch.bfloat16
                    )
                    eval_batch.vision_states = batch["vision_states"].to(
                        get_local_torch_device(), dtype=torch.bfloat16
                    )
                    eval_batch.extra_kwargs = extra_kwargs
                    eval_batch.w2c = batch["w2c"].to(
                        get_local_torch_device(), dtype=torch.bfloat16
                    )
                    eval_batch.intrinsic = batch["intrinsic"].to(
                        get_local_torch_device(), dtype=torch.bfloat16
                    )
                    eval_batch.action = batch["action"].to(
                        get_local_torch_device(), dtype=torch.bfloat16
                    )

                    latents = self._sample_model_ode(
                        eval_batch,
                        video_chunk_num + 1,
                        noise,
                        sampling_steps=sampling_steps,
                    )

                    if (
                        hasattr(self.vae.config, "shift_factor")
                        and self.vae.config.shift_factor
                    ):
                        latents = (
                            latents / self.vae.config.scaling_factor
                            + self.vae.config.shift_factor
                        )
                    else:
                        latents = latents / self.vae.config.scaling_factor

                    self.vae.to(self.device)
                    video_frames = self.vae.decode(
                        latents.to(torch.float32), return_dict=False
                    )[0]
                    self.vae.to("cpu")
                    video_frames = (video_frames / 2 + 0.5).clamp(0, 1).cpu().float()
                    video_frames = np.transpose(video_frames[0], (1, 2, 3, 0))
                    video_frames = (video_frames * 255).numpy().astype(np.uint8)

                    # Use generated_videos_dir from training_args
                    generated_videos_base = self.training_args.generated_videos_dir
                    if not generated_videos_base:
                        generated_videos_base = os.path.join(
                            self.training_args.output_dir, "generated_videos"
                        )
                    video_path = os.path.join(
                        generated_videos_base,
                        self.training_args.output_dir.split("/")[-1],
                        "000_eval",
                        f"sample_{self.global_rank + (batch_idx * self.world_size)}",
                    )
                    os.makedirs(video_path, exist_ok=True)
                    save_video_path = os.path.join(video_path, f"step_{step}.mp4")
                    print(f"video_frames shape: {video_frames.shape}")

                    imageio.mimsave(save_video_path, video_frames)
                    absolute_path = os.path.abspath(save_video_path)

                    score = self.reward_model.score_video(
                        absolute_path,
                        caption=prompt,
                        gt_camera_pose=w2c,
                        gt_action=action,
                        interval=1,
                        latent_num=latent_t,
                    )

                    new_filename = (
                        f"step_{step}"
                        f"_action_[{torch.tensor(score['action_acc']).mean().item():.1f}]"
                        f"_hpsv3_[{torch.tensor(score['hps_acc']).mean().item():.1f}]"
                        f"_quality_[{torch.tensor(score['hps_quality_acc']).mean().item():.1f}]"
                        f"_drift_[{torch.tensor(score['hps_drift_score']).mean().item():.1f}]"
                        ".mp4"
                    )

                    new_absolute_path = os.path.join(
                        os.path.dirname(absolute_path), new_filename
                    )
                    if os.path.exists(absolute_path):
                        os.rename(absolute_path, new_absolute_path)

                    action_acc += score["action_acc"]
                    hpsv3_acc += score["hps_acc"]
                    hpsv3_quality_acc += score["hps_quality_acc"]
                    hpsv3_drift_score += score["hps_drift_score"]

                world_group = get_world_group()
                all_action_acc = world_group.all_gather(
                    torch.tensor(action_acc).to(get_local_torch_device()).unsqueeze(0),
                    dim=0,
                )
                all_hpsv3_acc = world_group.all_gather(
                    torch.tensor(hpsv3_acc).to(get_local_torch_device()).unsqueeze(0),
                    dim=0,
                )
                all_hpsv3_quality_acc = world_group.all_gather(
                    torch.tensor(hpsv3_quality_acc)
                    .to(get_local_torch_device())
                    .unsqueeze(0),
                    dim=0,
                )
                all_hpsv3_drift_score = world_group.all_gather(
                    torch.tensor(hpsv3_drift_score)
                    .to(get_local_torch_device())
                    .unsqueeze(0),
                    dim=0,
                )

                ave_action_acc = (
                    torch.tensor(action_acc).mean().to(get_local_torch_device())
                )
                ave_hpsv3_acc = (
                    torch.tensor(hpsv3_acc).mean().to(get_local_torch_device())
                )
                ave_hpsv3_quality_acc = (
                    torch.tensor(hpsv3_quality_acc).mean().to(get_local_torch_device())
                )
                ave_hpsv3_drift_score = (
                    torch.tensor(hpsv3_drift_score).mean().to(get_local_torch_device())
                )

                ave_action_acc = world_group.all_reduce(
                    ave_action_acc, op=dist.ReduceOp.AVG
                )
                ave_hpsv3_acc = world_group.all_reduce(
                    ave_hpsv3_acc, op=dist.ReduceOp.AVG
                )
                ave_hpsv3_quality_acc = world_group.all_reduce(
                    ave_hpsv3_quality_acc, op=dist.ReduceOp.AVG
                )
                ave_hpsv3_drift_score = world_group.all_reduce(
                    ave_hpsv3_drift_score, op=dist.ReduceOp.AVG
                )

                return {
                    "ave_action_acc": ave_action_acc.cpu().item(),
                    "ave_hpsv3_acc": ave_hpsv3_acc.cpu().item(),
                    "ave_hpsv3_quality_acc": ave_hpsv3_quality_acc.cpu().item(),
                    "ave_hpsv3_drift_score": ave_hpsv3_drift_score.cpu().item(),
                    "all_action_acc": all_action_acc.cpu().tolist(),
                    "all_hpsv3_acc": all_hpsv3_acc.cpu().tolist(),
                    "all_hpsv3_quality_acc": all_hpsv3_quality_acc.cpu().tolist(),
                    "all_hpsv3_drift_score": all_hpsv3_drift_score.cpu().tolist(),
                }

    def train(self) -> None:
        assert self.seed is not None, "seed must be set"
        set_random_seed(self.seed + self.global_rank)
        logger.info(
            "rank: %s: start training", self.global_rank, local_main_process_only=False
        )

        if not self.post_init_called:
            self.post_init()
        num_trainable_params = _get_trainable_params(self.transformer)
        logger.info(
            "Starting training with %s B trainable parameters",
            round(num_trainable_params / 1e9, 3),
        )

        import subprocess

        kill_cmds = [
            "ps -ef | grep run.py | awk '{print $2}' | head -n -1 | xargs kill -9"
        ]
        for cmd in kill_cmds:
            try:
                subprocess.run(cmd, shell=True, check=False)
            except Exception as e:
                logger.warning(f"Error running kill command: {cmd} -- {e}")

        # Set random seeds for deterministic training
        self.noise_random_generator = torch.Generator(device="cpu").manual_seed(
            self.seed
        )
        self.noise_gen_cuda = torch.Generator(device="cuda").manual_seed(self.seed)
        self.validation_random_generator = torch.Generator(device="cpu").manual_seed(
            self.seed
        )
        logger.info("Initialized random seeds with seed: %s", self.seed)

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler()

        if self.training_args.resume_from_checkpoint:
            self._resume_from_checkpoint()

        self.train_loader_iter = iter(self.train_dataloader)

        step_times: deque[float] = deque(maxlen=100)

        self._log_training_info()

        self.ema_generator = EMA_FSDP_schedule(
            self.transformer,
            min_decay=self.training_args.ema_min_decay,
            max_decay=self.training_args.ema_max_decay,
            step_decay=self.training_args.ema_step_decay,
            ckpt_decay=self.training_args.ema_ckpt_decay,
        )

        # Train!
        progress_bar = tqdm(
            range(0, self.training_args.max_train_steps),
            initial=self.init_steps,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=self.local_rank > 0,
        )
        for step in range(self.init_steps + 1, self.training_args.max_train_steps + 1):
            if step == 1:
                eval_results = self._eval(step - 1)
                if self.training_args.eval_only:
                    print(eval_results)
                    break
                eval_results = self._eval(step - 1)
                if self.global_rank == 0:
                    wandb.log(
                        {
                            f"eval_action_acc": eval_results["ave_action_acc"],
                            f"eval_hpsv3_acc": eval_results["ave_hpsv3_acc"],
                            f"eval_hpsv3_quality_acc": eval_results[
                                "ave_hpsv3_quality_acc"
                            ],
                            f"ave_hpsv3_drift_score": eval_results[
                                "ave_hpsv3_drift_score"
                            ],
                        },
                        step=step,
                    )

            start_time = time.perf_counter()
            if vsa_available:
                vsa_sparsity = self.training_args.VSA_sparsity
                vsa_decay_rate = self.training_args.VSA_decay_rate
                vsa_decay_interval_steps = self.training_args.VSA_decay_interval_steps
                current_decay_times = min(
                    step // vsa_decay_interval_steps, vsa_sparsity // vsa_decay_rate
                )
                current_vsa_sparsity = current_decay_times * vsa_decay_rate
            else:
                current_vsa_sparsity = 0.0

            training_batch = TrainingBatch()
            training_batch.current_timestep = step
            training_batch.current_vsa_sparsity = current_vsa_sparsity
            training_batch = self.train_one_step(training_batch)

            samples_loss = training_batch.samples_loss
            samples_grad_norm = training_batch.samples_grad_norm

            step_time = time.perf_counter() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)

            total_loss = training_batch.total_loss / len(samples_grad_norm)
            total_grad_norm = sum(samples_grad_norm.values()) / len(samples_grad_norm)

            progress_bar.set_postfix(
                {
                    "loss": total_loss,
                    "step_time": f"{step_time:.2f}s",
                    "grad_norm": total_grad_norm,
                }
            )
            progress_bar.update(1)

            if self.global_rank == 0:
                wandb.log(
                    {
                        f"ave_loss": training_batch.total_loss / len(samples_grad_norm),
                        f"ave_grad_norm": sum(samples_grad_norm.values())
                        / len(samples_grad_norm),
                    },
                    step=step,
                )

                wandb.log(
                    {
                        "learning_rate": self.lr_scheduler.get_last_lr()[0],
                        "step_time": step_time,
                        "avg_step_time": avg_step_time,
                        "action_reward": training_batch.action_reward_mean,
                        "fine_action_reward": training_batch.fine_action_reward_mean,
                        "hpsv3_reward": training_batch.hpsv3_reward_mean,
                        "hpsv3_quality_reward": training_batch.hpsv3_quality_reward_mean,
                    },
                    step=step,
                )

            training_batch = TrainingBatch()

            if step % self.training_args.checkpointing_steps == 0:
                eval_results = self._eval(step - 1)
                if self.global_rank == 0:
                    wandb.log(
                        {
                            f"eval_action_acc": eval_results["ave_action_acc"],
                            f"eval_hpsv3_acc": eval_results["ave_hpsv3_acc"],
                            f"eval_hpsv3_quality_acc": eval_results[
                                "ave_hpsv3_quality_acc"
                            ],
                            f"ave_hpsv3_drift_score": eval_results[
                                "ave_hpsv3_drift_score"
                            ],
                        },
                        step=step,
                    )

                with self.ema_generator.apply_ckpt_shadow_to_model(self.transformer):
                    save_checkpoint(
                        self.transformer,
                        self.global_rank,
                        self.training_args.output_dir,
                        step,
                        self.optimizer,
                        self.train_dataloader,
                        self.lr_scheduler,
                        self.noise_random_generator,
                    )
                self.transformer.train()
                self.sp_group.barrier()

        wandb.finish()
        save_checkpoint(
            self.transformer,
            self.global_rank,
            self.training_args.output_dir,
            self.training_args.max_train_steps,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
            self.noise_random_generator,
        )

        if get_sp_group():
            cleanup_dist_env_and_memory()

    def _log_training_info(self) -> None:
        total_batch_size = (
            self.world_size
            * self.training_args.gradient_accumulation_steps
            / self.training_args.sp_size
            * self.training_args.train_sp_batch_size
        )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %s", len(self.train_dataset))
        logger.info("  Dataloader size = %s", len(self.train_dataloader))
        logger.info("  Num Epochs = %s", self.num_train_epochs)
        logger.info("  Resume training from step %s", self.init_steps)  # type: ignore
        logger.info(
            "  Instantaneous batch size per device = %s",
            self.training_args.train_batch_size,
        )
        logger.info(
            "  Total train batch size (w. data & sequence parallel, accumulation) = %s",
            total_batch_size,
        )
        logger.info(
            "  Gradient Accumulation steps = %s",
            self.training_args.gradient_accumulation_steps,
        )
        logger.info(
            "  Total optimization steps = %s", self.training_args.max_train_steps
        )
        logger.info(
            "  Total training parameters per FSDP shard = %s B",
            round(_get_trainable_params(self.transformer) / 1e9, 3),
        )
        # print dtype
        logger.info(
            "  Master weight dtype: %s", self.transformer.parameters().__next__().dtype
        )

        gpu_memory_usage = torch.cuda.memory_allocated() / 1024**2
        logger.info("GPU memory usage before train_one_step: %s MB", gpu_memory_usage)
        logger.info("VSA validation sparsity: %s", self.training_args.VSA_sparsity)
