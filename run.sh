export T2V_REWRITE_BASE_URL="<your_vllm_server_base_url>"
export T2V_REWRITE_MODEL_NAME="<your_model_name>"
export I2V_REWRITE_BASE_URL="<your_vllm_server_base_url>"
export I2V_REWRITE_MODEL_NAME="<your_model_name>"

PROMPT='A paved pathway leads towards a stone arch bridge spanning a calm body of water.  Lush green trees and foliage line the path and the far bank of the water. A traditional-style pavilion with a tiered, reddish-brown roof sits on the far shore. The water reflects the surrounding greenery and the sky.  The scene is bathed in soft, natural light, creating a tranquil and serene atmosphere. The pathway is composed of large, rectangular stones, and the bridge is constructed of light gray stone.  The overall composition emphasizes the peaceful and harmonious nature of the landscape.'

IMAGE_PATH=./assets/img/test.png # Now we only provide the i2v model, so the path cannot be None
SEED=1
ASPECT_RATIO=16:9
RESOLUTION=480p                  # Now we only provide the 480p model
OUTPUT_PATH=./outputs/
MODEL_PATH=                      # Path to pretrained hunyuanvideo-1.5 model
AR_ACTION_MODEL_PATH=            # Path to our HY-World 1.5 autoregressive checkpoints
BI_ACTION_MODEL_PATH=            # Path to our HY-World 1.5 bidirectional checkpoints
AR_DISTILL_ACTION_MODEL_PATH=    # Path to our HY-World 1.5 autoregressive distilled checkpoints
POSE_JSON_PATH=./assets/pose/test_forward_32_latents.json   # Path to the customized camera trajectory
NUM_FRAMES=125
WIDTH=832
HEIGHT=480

# Configuration for faster inference
# The maximum number recommended is 8.
N_INFERENCE_GPU=8 # Parallel inference GPU count.

# Configuration for better quality
REWRITE=false # Enable prompt rewriting. Please ensure rewrite vLLM server is deployed and configured.
ENABLE_SR=false # Enable super resolution. When the NUM_FRAMES <= 121, you can set it to true

# inference with bidirectional model
torchrun --nproc_per_node=$N_INFERENCE_GPU generate.py  \
  --prompt "$PROMPT" \
  --image_path $IMAGE_PATH \
  --resolution $RESOLUTION \
  --aspect_ratio $ASPECT_RATIO \
  --video_length $NUM_FRAMES \
  --seed $SEED \
  --rewrite $REWRITE \
  --sr $ENABLE_SR --save_pre_sr_video \
  --pose_json_path $POSE_JSON_PATH \
  --output_path $OUTPUT_PATH \
  --model_path $MODEL_PATH \
  --action_ckpt $BI_ACTION_MODEL_PATH \
  --few_step false \
  --width $WIDTH \
  --height $HEIGHT \
  --model_type 'bi'

# inference with autoregressive model
#torchrun --nproc_per_node=$N_INFERENCE_GPU generate.py  \
#  --prompt "$PROMPT" \
#  --image_path $IMAGE_PATH \
#  --resolution $RESOLUTION \
#  --aspect_ratio $ASPECT_RATIO \
#  --video_length $NUM_FRAMES \
#  --seed $SEED \
#  --rewrite $REWRITE \
#  --sr $ENABLE_SR --save_pre_sr_video \
#  --pose_json_path $POSE_JSON_PATH \
#  --output_path $OUTPUT_PATH \
#  --model_path $MODEL_PATH \
#  --action_ckpt $AR_ACTION_MODEL_PATH \
#  --few_step false \
#  --width $WIDTH \
#  --height $HEIGHT \
#  --model_type 'ar'

# inference with autoregressive distilled model
#torchrun --nproc_per_node=$N_INFERENCE_GPU generate.py  \
#  --prompt "$PROMPT" \
#  --image_path $IMAGE_PATH \
#  --resolution $RESOLUTION \
#  --aspect_ratio $ASPECT_RATIO \
#  --video_length $NUM_FRAMES \
#  --seed $SEED \
#  --rewrite $REWRITE \
#  --sr $ENABLE_SR --save_pre_sr_video \
#  --pose_json_path $POSE_JSON_PATH \
#  --output_path $OUTPUT_PATH \
#  --model_path $MODEL_PATH \
#  --action_ckpt $AR_DISTILL_ACTION_MODEL_PATH \
#  --few_step true \
#  --num_inference_steps 4 \
#  --width $WIDTH \
#  --height $HEIGHT \
#  --model_type 'ar'