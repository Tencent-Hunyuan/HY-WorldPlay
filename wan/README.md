# WAN Pipeline Inference

The WAN pipeline provides a lightweight alternative with distributed inference support. It's optimized for multi-GPU setups and offers faster inference with lower memory footprint.

## Prerequisites

Make sure you have completed the base environment setup from the [main README](../README.md), including:
- Creating the `worldplay` conda environment
- Installing `requirements.txt`
- Installing **SageAttention** (required for WAN pipeline)

**Important:** All WAN commands must be run from the **repository root directory**, and require `PYTHONPATH` to include both the repo root and the `wan/` subdirectory:

```bash
cd /path/to/HY-WorldPlay
export PYTHONPATH=$(pwd):$(pwd)/wan:$PYTHONPATH
```

## Download WAN Models

The WAN models are included in the `tencent/HY-WorldPlay` repository. You can download them with:

```bash
# Download both wan_transformer and wan_distilled_model into a local directory
huggingface-cli download tencent/HY-WorldPlay wan_transformer wan_distilled_model --local-dir /path/to/models
```

This creates the following structure:
```
/path/to/models/
├── wan_transformer/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
└── wan_distilled_model/
    └── model.pt
```

Then use:
- `--ar_model_path /path/to/models/wan_transformer`
- `--ckpt_path /path/to/models/wan_distilled_model/model.pt`

> **Tip:** If you already have these files in your HuggingFace cache, you can find the snapshot path with:
> ```bash
> python -c "from huggingface_hub import snapshot_download; print(snapshot_download('tencent/HY-WorldPlay', local_files_only=True))"
> ```
> Then use `<snapshot_path>/wan_transformer` and `<snapshot_path>/wan_distilled_model/model.pt` directly.

## Configuration Parameters

The WAN pipeline accepts the following parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input` | Text prompt or path to txt file with prompts | Required |
| `--image_path` | Input image path for I2V generation | None (T2V mode) |
| `--num_chunk` | Number of chunks to generate (each chunk = 4 latents) | 4 |
| `--pose` | Camera trajectory (pose string or JSON file) | `w-96` |
| `--ar_model_path` | Path to WAN transformer model directory | Required |
| `--ckpt_path` | Path to trained checkpoint file (model.pt) | Required |
| `--out` | Output directory for generated videos | `outputs` |

> **Note:** The total pose duration in latents should match `num_chunk * 4`. For example, with `num_chunk=1`, use `--pose "w-4"` (4 latents).

## Single-GPU Inference

```bash
export PYTHONPATH=$(pwd):$(pwd)/wan:$PYTHONPATH

torchrun --nproc_per_node=1 wan/generate.py \
  --input "First-person view walking around ancient Athens, with Greek architecture and marble structures" \
  --num_chunk 1 \
  --pose "w-4" \
  --ar_model_path /path/to/models/wan_transformer \
  --ckpt_path /path/to/models/wan_distilled_model/model.pt \
  --out outputs
```

## Multi-GPU Distributed Inference

For multi-GPU inference with better performance:

```bash
export PYTHONPATH=$(pwd):$(pwd)/wan:$PYTHONPATH

# Using 4 GPUs
torchrun --nproc_per_node=4 wan/generate.py \
  --input "First-person view walking around ancient Athens, with Greek architecture and marble structures" \
  --num_chunk 4 \
  --pose "w-16" \
  --ar_model_path /path/to/models/wan_transformer \
  --ckpt_path /path/to/models/wan_distilled_model/model.pt \
  --out outputs
```

## Batch Processing with Text Files

You can process multiple prompts by providing a text file:

```bash
export PYTHONPATH=$(pwd):$(pwd)/wan:$PYTHONPATH

# Create a prompts.txt file with one prompt per line
echo "First-person view of a medieval castle" > prompts.txt
echo "Walking through a cyberpunk city at night" >> prompts.txt
echo "Exploring an underwater coral reef" >> prompts.txt

# Run inference on all prompts
torchrun --nproc_per_node=1 wan/generate.py \
  --input prompts.txt \
  --ar_model_path /path/to/models/wan_transformer \
  --ckpt_path /path/to/models/wan_distilled_model/model.pt
```

## Camera Control with WAN

WAN uses the same camera control system as the HunyuanVideo pipeline:

**Pose String Format:**
```bash
# Forward movement for 4 latents (with num_chunk=1)
--pose "w-4"

# Complex trajectory
--pose "w-20, right-10, d-30, up-36"
```

**Supported Actions:**
- **Movement**: `w` (forward), `s` (backward), `a` (left), `d` (right)
- **Rotation**: `up` (pitch up), `down` (pitch down), `left` (yaw left), `right` (yaw right)
- **Format**: `action-duration` where duration specifies the number of latents corresponding to the given action.
