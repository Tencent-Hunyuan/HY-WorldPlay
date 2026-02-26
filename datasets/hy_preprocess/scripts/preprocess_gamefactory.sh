#!/bin/bash

# GameFactory/Minecraft dataset preprocessing

# Activate virtual environment (if exists)
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "===================================="
echo "GameFactory/Minecraft Data Preprocessing"
echo "===================================="

# Set PYTHONPATH
export PYTHONPATH="$(pwd)"

# Output paths
OUTPUT_DIR="./preprocessed_gamefactory_f129"
OUTPUT_JSON="dataset_index.json"

# Model path
MODEL_PATH="./model_ckpts/HunyuanVideo-1.5"


# Target resolution
TARGET_HEIGHT=480
TARGET_WIDTH=832

# Other options
DEVICE="cuda"
NUM_SAMPLES=  # Leave empty to process all segments; set a number for testing (e.g. NUM_SAMPLES=2)
TARGET_NUM_FRAMES=129  # Empty=no resampling; e.g. 129 means index interpolation to 129 frames

echo ""
echo "Configuration:"
echo "  Dataset root: $DATA_ROOT"
echo "  Output dir: $OUTPUT_DIR"
echo "  Model path: $MODEL_PATH"
echo "  Target resolution: ${TARGET_WIDTH}x${TARGET_HEIGHT}"
echo "  Device: $DEVICE"
echo "  Num segments: ${NUM_SAMPLES:-all}"
echo "  Target frames: ${TARGET_NUM_FRAMES:-unchanged}"
echo ""

# Run preprocessing
python3 datasets/hy_preprocess/preprocess_gamefactory_dataset.py \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --output_json "$OUTPUT_JSON" \
    --model_path "$MODEL_PATH" \
    --target_height "$TARGET_HEIGHT" \
    --target_width "$TARGET_WIDTH" \
    --device "$DEVICE" \
    ${NUM_SAMPLES:+--num_samples $NUM_SAMPLES} \
    ${TARGET_NUM_FRAMES:+--target_num_frames $TARGET_NUM_FRAMES}
