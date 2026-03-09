#!/bin/bash

source ~/.bashrc

export PYTHONUNBUFFERED=1

if [ $# -ge 1 ]; then
    RANK=$1
else
    RANK=0
fi

if [ $# -ge 2 ]; then
    NODES=$2
else
    NODES=1
fi

NUM_GPUS=8                  # Number of GPUs per node
NODES=$NODES                # Number of nodes

export MASTER_ADDR=""       # [TODO] Set master address
export WORLD_SIZE=$NODES
export RANK=$RANK           # Node rank, ranged from 0 to N-1
export MASTER_PORT=27858
export NUM_NODES=$WORLD_SIZE

CACHE_DIR=""              # [TODO] Path to the overall checkpoint cache directory
HUNYUAN_CHECKPOINT=""     # [TODO] Path to the HunYuan checkpoint directory
WORLDPLAY_CHECKPOINT=""   # [TODO] Path to the WorldPlay checkpoint directory

TRAIN_LATENTS_DIR=""      # [TODO] Path to the train latents directory
EVAL_LATENTS_DIR=""       # [TODO] Path to the eval latents directory
POSE_PATH=""              # [TODO] Path to the custom action pose json
OUTPUT_DIR=""             # [TODO] Path to the output directory

exp_name="WorldCompass"

# Optimizer arguments
optimizer_args=(
  --learning_rate 1e-5
  --mixed_precision "bf16"
  --checkpointing_steps 6
  --weight_decay 1e-4
  --max_grad_norm 2.0
)

# Model arguments
model_args=(
  --cls_name "HunyuanTransformer3DARActionModel"
  --load_from_dir "${HUNYUAN_CHECKPOINT}/transformer/480p_i2v/"
  --ar_action_load_from_dir "${WORLDPLAY_CHECKPOINT}/ar_model/diffusion_pytorch_model.safetensors"
  --eval_only False
)

# Training arguments
training_args=(
  # Path related arguments
  --json_path "${TRAIN_LATENTS_DIR}/latents.json"                            # Path to train latents json file
  --eval_json_path "${EVAL_LATENTS_DIR}/latents.json"                        # Path to eval latents json file
  --random_pose_path "${POSE_PATH}"                                          # Path to random pose json file
  --vae_path "${HUNYUAN_CHECKPOINT}/vae/"                                    # Path to VAE model
  --cache_dir "${CACHE_DIR}"                                                 # Path to cache directory

  --output_dir "${OUTPUT_DIR}/${exp_name}"                                   # Path to save checkpoints and logs
  --generated_videos_dir "${OUTPUT_DIR}/generated_videos/${exp_name}"        # Path to save generated videos

  # Reward model argument
  --camera_estimator "dav3"                                                  # Camera estimator: "dav3" or "worldmirror"

  # wandb arguments
  --wandb_key ""                                                             # [TODO] Set wandb key
  --wandb_entity ""                                                          # [TODO] Set wandb entity
  --tracker_project_name "worldcompass"

  # Basic training parameters
  --max_train_steps 100                                                      # Number of training iterations
  --train_batch_size 1                                                       # Training batch size
  --train_sp_batch_size 1                                                    # Training special batch size
  --window_frames 64                                                         # Video window length for training (longest training video)
  --single_chunk_size 4                                                      # Length of single chunk for training

  # Training
  --gradient_accumulation_steps 2
  --enable_gradient_checkpointing_type "full"

  # GRPO sampling and training parameters (for reinforcement learning)
  --sampling_steps 40                                                        # Number of rollout steps
  --grpo_generation_num 12                                                   # Number of rollout samples generated per input
  --train_timestep_fraction 0.5                                              # Fraction of total timesteps used for training
  --bestofn 6                                                                # Select best n samples from all rollouts for training
  --sampling_batch_size 1                                                    # Batch size for rollout sampling

  # Sampling and training chunk selection strategy
  --chunk_selection_strategy "min2max"                                       # Chunk selection strategy: "min2max" or "max2min"
  --min_chunk_id 1                                                           # Minimum chunk ID (usually 1)
  --max_chunk_id 16                                                          # Maximum chunk ID (usually window_frames // single_chunk_size)

  # Reward arguments
  --action_reward_weight 2.0
  --hpsv3_reward_weight 0.0
  --hpsv3_quality_reward_weight 0.0
  --hpsv3_quality_drift_reward_weight 1.0
  --action_reward_type 'fine_action'
  --adv_clip_max 2.0
  --std_type "global"

  # EMA arguments
  --ema_min_decay 0.2
  --ema_max_decay 0.9
  --ema_step_decay 0.01
  --ema_ckpt_decay 0.9

  # Model task type arguments
  --causal
  --action
  --i2v_rate 1.0
)

# Parallelism arguments
parallel_args=(
  --num_gpus $(( NUM_GPUS * NODES ))    # Total GPU count used (NUM_GPUS per node * NODES)
  --sp_size 1
  --tp_size 1
  --gpu_para 1
  --hsdp_replicate_dim $NODES
  --hsdp_shard_dim $NUM_GPUS
)

# Dataset arguments
dataset_args=(
  --neg_prompt_path ''                  # Not used
  --neg_byt5_prompt_path ''             # Not used
  --data_path ''                        # Not used
  --model_path ''                       # Not used
  --pretrained_model_name_or_path ''    # Not used
  --dataloader_num_workers 4
)

# Miscellaneous arguments
miscellaneous_args=(
  --training_cfg_rate 0.0
)

echo "Begin training!"
python -u -m torch.distributed.run \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=$NUM_NODES \
    --node_rank=$RANK \
    fastvideo/training/world_compass_train_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${miscellaneous_args[@]}"
