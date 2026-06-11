
export WANDB_BASE_URL="https://api.wandb.ai"
# export WANDB_MODE=online
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
# export TRAINER_ATTENTION_BACKEND=TORCH_SDPA

TRANSFORMER_MODEL_PATH=./model_ckpts/HunyuanVideo-1.5/transformer/480p_i2v
AR_DISTILL_ACTION_MODEL_PATH=./model_ckpts/HY-WorldPlay/ar_distilled_action_model/diffusion_pytorch_model.safetensors
MODEL_PATH=./model_ckpts/HunyuanVideo-1.5
VALIDATION_DATASET_FILE=    # Path to validation json file                                      

NUM_GPUS=4                                                                                                                                                                    # 节点数量
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Training arguments
training_args=(
  --json_path ./preprocessed_gamefactory_f129/dataset_index.json
  --causal                                                                                    
  --action                                                                                    
  --i2v_rate 0.2
  --train_time_shift 3.0
  --window_frames 24                                                                        
  --wandb_key # Your wandb key
  --wandb_entity # Your wandb entity
  --tracker_project_name # Your wandb project name
  --output_dir # Path to output directory
  --max_train_steps 200000                                                                    
  --train_batch_size 1                                                                        
  --train_sp_batch_size 1                                                                     
  --gradient_accumulation_steps 1
  --num_latent_t 9                 # not used (these parameters are mainly for testing during training, but testing during training is now turned off)                                                         
  --num_height 480                                                                            
  --num_width 832                                                                             
  --num_frames 77                                                                             
  --enable_gradient_checkpointing_type "full"
  --seed 3208
  --weighting_scheme "logit_normal"
  --logit_mean 0.0
  --logit_std 1.0
)

# Parallel arguments
parallel_args=(
  --num_gpus $((NUM_GPUS * 1))          # gpu number for each node
  --sp_size 4           # gpu number for each sp group, usually set to 4 or 8, but need to ensure window_frames % sp == 0
  --tp_size 1
  --hsdp_replicate_dim 1
  --hsdp_shard_dim $NUM_GPUS
)

# Model arguments
model_args=(
  --cls_name "HunyuanTransformer3DARActionModel"
  --load_from_dir $TRANSFORMER_MODEL_PATH
  --ar_action_load_from_dir $AR_DISTILL_ACTION_MODEL_PATH
  --model_path $MODEL_PATH
  --pretrained_model_name_or_path $MODEL_PATH
)

# Dataset arguments
dataset_args=(
  --dataloader_num_workers 1
)

validation_args=(
  # --log_validation
  # --validation_dataset_file $VALIDATION_DATASET_FILE
  --validation_steps 200
  --validation_sampling_steps "50"
  --validation_guidance_scale "6.0"
)

# Optimizer arguments
optimizer_args=(
  --learning_rate 1e-5
  --mixed_precision "bf16"
  --checkpointing_steps 500
  --weight_decay 1e-4
  --max_grad_norm 1.0
)

# Miscellaneous arguments
miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 3
  --training_cfg_rate 0.1
  --multi_phased_distill_schedule "4000-1"
  --not_apply_cfg_solver
  --dit_precision "fp32"
  --num_euler_timesteps 50
  --ema_start_step 0
  # --enable_gradient_checkpointing_type "full"
)

export MASTER_PORT=29611

torchrun \
        --master_port=$MASTER_PORT \
        --nproc_per_node=$NUM_GPUS \
        --nnodes 1 \
        trainer/training/ar_hunyuan_w_mem_training_pipeline.py \
        "${parallel_args[@]}" \
        "${model_args[@]}" \
        "${dataset_args[@]}" \
        "${training_args[@]}" \
        "${optimizer_args[@]}" \
        "${validation_args[@]}" \
        "${miscellaneous_args[@]}"