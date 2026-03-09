# HunyuanWorldMirror - Inference Only

This is a minimal inference-only version of HunyuanWorldMirror, stripped down
for use in reward functions.

## Structure

```
HunyuanWorldMirror/
├── __init__.py              # Main package init
├── infer.py                 # Standalone inference script
├── src/
│   ├── models/
│   │   ├── heads/           # Camera and dense prediction heads
│   │   ├── layers/          # Transformer layers
│   │   ├── models/          # Core WorldMirror model
│   │   └── utils/           # Model utilities
│   └── utils/               # Inference utilities
├── License.txt
└── Notice.txt
```

## Usage

### As a module:

```python
from HunyuanWorldMirror import WorldMirror

# Load model
model = WorldMirror.from_pretrained("tencent/HunyuanWorldMirror")
model.eval()

# Prepare images
# ... (see infer.py for full example)

# Run inference
with torch.no_grad():
    predictions = model(images)
```

### Standalone inference:

```bash
python infer.py --input_path <path_to_images_or_video> --output_dir <output_directory>
```

## Changes from Original

- Removed all training code (`training/` directory)
- Removed demo applications (`app.py`, `demo/`)
- Removed examples and documentation
- Removed submodules
- All imports updated to use `HunyuanWorldMirror.src.*` prefix for proper module
  resolution
- Cleaned up cache files and assets

## Dependencies

Core dependencies for inference:

- torch
- numpy
- PIL
- cv2
- gsplat (for Gaussian splatting)
- einops
- huggingface_hub

Optional dependencies:

- onnxruntime (for sky segmentation)
- pycolmap (for reconstruction)
- moviepy (for video rendering)
