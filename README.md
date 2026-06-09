<div align="center">
  <img src="assets/teaser.webp">

<h1>🎮 HY-World 1.5: A Systematic Framework for Interactive World Modeling with Real-Time Latency and Geometric Consistency</h1>

[English](README.md) | [简体中文](README_zh.md)



</div>

<div align="center">
  <a href=https://3d.hunyuan.tencent.com/sceneTo3D?tab=worldplay target="_blank"><img src=https://img.shields.io/badge/Official%20Site-333399.svg?logo=homepage height=22px></a>
  <a href=https://huggingface.co/tencent/HY-WorldPlay target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg height=22px></a>
  <a href=https://3d-models.hunyuan.tencent.com/world/ target="_blank"><img src= https://img.shields.io/badge/Page-bb8a2e.svg?logo=github height=22px></a>
  <a href=https://3d-models.hunyuan.tencent.com/world/world1_5/HYWorld_1.5_Tech_Report.pdf target="_blank"><img src=https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv height=22px></a>
  <a href=https://discord.gg/dNBrdrGGMa target="_blank"><img src= https://img.shields.io/badge/Discord-white.svg?logo=discord height=22px></a>
  <a href=https://x.com/TencentHunyuan target="_blank"><img src=https://img.shields.io/badge/Tencent%20HY-black.svg?logo=x height=22px></a>
 <a href="#community-resources" target="_blank"><img src=https://img.shields.io/badge/Community-lavender.svg?logo=homeassistantcommunitystore height=22px></a>
</div>

[//]: # (  <a href=# target="_blank"><img src=https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv height=22px></a>)

[//]: # (  <a href=# target="_blank"><img src= https://img.shields.io/badge/Colab-8f2628.svg?logo=googlecolab height=22px></a>)

[//]: # (  <a href="#"><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/v/mulankit?logo=pypi"  height=22px></a>)

<br>

<p align="center">
  <i>"Hold Infinity in the Palm of Your Hand, and Eternity in an Hour"</i>
</p>

## 🎥 Video

https://github.com/user-attachments/assets/9fd12b40-41ab-4201-8667-8b333db1123d

## 🔥 News
- April 16, 2026: 🤗 We release [HY-World-2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0), state-of-the-art 3D world model! 
- March 8, 2026: 🚀 We release the RL post-training code 🧭 [WorldCompass](worldcompass/README.md) for WorldPlay-8B model (based on HY Video)! Read more in [our new paper](https://arxiv.org/abs/2602.09022).
- January 6, 2026: 🚀 We release the training code for WorldPlay-8B model (based on HY Video), enabling the community to train and fine-tune their own world models!
- January 6, 2026: 🎯 We open-source WorldPlay-5B model (based on WAN), a new lightweight model that fits into small-VRAM GPUs (but with compromised quality)!
- January 3, 2026: ⚡ We update the inference code with quantization and engineering optimization for even faster inference speeds!
- December 17, 2025: 👋 We present the [technical report](https://3d-models.hunyuan.tencent.com/world/world1_5/HYWorld_1.5_Tech_Report.pdf) (and [research paper](https://arxiv.org/abs/2512.14614)) of HY-World 1.5 (WorldPlay), please check out the details and spark some discussion!
- December 17, 2025: 🤗 We release the first open-source, real-time interactive, and long-term geometric consistent world model, HY-World 1.5 (WorldPlay)!

> Join our **[Wechat](#)** and **[Discord](https://discord.gg/dNBrdrGGMa)** group to discuss and find help from us.

| Wechat Group                                     | Xiaohongshu                                           | X                                           | Discord                                           |
|--------------------------------------------------|-------------------------------------------------------|---------------------------------------------|---------------------------------------------------|
| <img src="assets/qrcode/wechat.png"  height=140> | <img src="assets/qrcode/xiaohongshu.png"  height=140> | <img src="assets/qrcode/x.png"  height=140> | <img src="assets/qrcode/discord.png"  height=140> | 


## 📋 Table of Contents
- [🎥 Video](#-video)
- [🔥 News](#-news)
- [📋 Table of Contents](#-table-of-contents)
- [📖 Introduction](#-introduction)
- [✨ Highlights](#-highlights)
- [📜 System Requirements](#-system-requirements)
- [🛠️ Dependencies and Installation](#️-dependencies-and-installation)
  - [1. Create Environment](#1-create-environment)
  - [2. Install Attention Libraries](#2-install-attention-libraries)
  - [3. Install AngelSlim and DeepGEMM (Optional)](#3-install-angelslim-and-deepgemm-optional-for-quantizationfp8-acceleration)
  - [4. Download All Required Models](#4-download-all-required-models)
- [🎮 Quick Start](#-quick-start)
- [🧱 Model Checkpoints](#-model-checkpoints)
- [🔑 Inference](#-inference)
  - [Configure Model Paths](#configure-model-paths)
  - [Configuration Options](#configuration-options)
  - [Model Selection](#model-selection)
  - [Camera Trajectory Control](#camera-trajectory-control)
    - [Option 1: Pose String (Recommended for Quick Testing)](#option-1-pose-string-recommended-for-quick-testing)
    - [Option 2: Custom JSON Files](#option-2-custom-json-files)
  - [Prompt Rewriting (Optional)](#prompt-rewriting-optional)
  - [Run Inference](#run-inference)
- [⚙️Training](#️training)
- [📊 Evaluation](#-evaluation)
- [🎬 More Examples](#-more-examples)
- [📝 TODO](#-todo)
- [📚 Citation](#-citation)
- [Contact](#contact)
- [🙏 Acknowledgements](#-acknowledgements)

## 📖 Introduction
While **HY-World 1.0** is capable of generating immersive 3D worlds, it relies on a lengthy offline generation process and lacks real-time interaction. **HY-World 1.5** bridges this gap with **WorldPlay**, a streaming video diffusion model that enables real-time, interactive world modeling with long-term geometric consistency, resolving the trade-off between speed and memory that limits current methods.  Our model draws power from four key designs. 1) We use a Dual Action Representation to enable robust action control in response to the user's keyboard and mouse inputs. 2) To enforce long-term consistency, our Reconstituted Context Memory dynamically rebuilds context from past frames and uses temporal reframing to keep geometrically important but long-past frames accessible, effectively alleviating memory attenuation. 3) We design WorldCompass, a novel Reinforcement Learning (RL) post-training framework designed to directly improve the action-following and visual quality of the long-horizon, autoregressive video model. 4) We also propose Context Forcing, a novel distillation method designed for memory-aware models. Aligning memory context between the teacher and student preserves the student's capacity to use long-range information, enabling real-time speeds while preventing error drift.  Taken together,  HY-World 1.5 generates long-horizon streaming video at 24 FPS with superior consistency, comparing favorably with existing techniques. Our model shows strong generalization across diverse scenes,  supporting first-person and third-person perspectives in both real-world and stylized environments, enabling versatile applications such as 3D reconstruction, promptable events, and infinite world extension. 

<p align="center">
  <img src="assets/teaser_2.png">
</p>

## ✨ Highlights

- **Systematic Overview**
  
  HY-World 1.5 has open-sourced a systematic and comprehensive training framework for real-time world models, covering the entire pipeline and all stages, including data, training, and inference deployment. The technical report discloses detailed training specifics for model pre-training, middle-training, reinforcement learning post-training, and memory-aware model distillation. In addition, the report introduces a series of engineering techniques aimed at reducing network transmission latency and model inference latency, thereby achieving a real-time streaming inference experience for users.

<p align="center">
  <img src="assets/overview.png">
</p>

- **Inference Pipeline**
  
  Given a single image or text prompt to describe a world, our model performs a next chunk (16 video frames) prediction task to generate future videos conditioned on action from users. For the generation of each chunk, we dynamically reconstitute context memory from past chunks to enforce long-term temporal and geometric consistency.

<p align="center">
  <img src="assets/pipeline.png">
</p>



## 📜 System Requirements

- **GPU**: NVIDIA GPU with CUDA support
- **GPU Memory cost**:
  - Inference with AR distilled models (based on HunyuanVideo1.5 with 125 frames):
    - sp = 8: memory=28G
    - sp = 4: memory=34G
    - sp = 1: memory=72G
  - Training (based on HunyuanVideo1.5 with 125 frames):
    - sp = 8: memory=60G


## 🛠️ Dependencies and Installation

### 1. Create Environment
```bash
conda create --name worldplay python=3.10 -y
conda activate worldplay
pip install -r requirements.txt
```

### 2. Install Attention Libraries
* SageAttention (**Required** for the WAN pipeline, optional for HunyuanVideo):
  ```bash
  pip install sageattention
  ```
  Alternatively, you can build from source for potentially better performance:
  ```bash
  git clone https://github.com/cooper1637/SageAttention.git
  cd SageAttention
  export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32 # Optional
  python3 setup.py install
  ```

* Flash Attention (optional, for faster HunyuanVideo inference):
  ```bash
  pip install flash-attn --no-build-isolation
  ```
  Detailed instructions: [Flash Attention](https://github.com/Dao-AILab/flash-attention)

### 3. Install AngelSlim and DeepGEMM (Optional, for quantization/fp8 acceleration)
* AngelSlim:
  Install AngelSlim to quantize transformer. Only needed if you enable `--use_fp8_gemm true` in `run.sh`.
  ```bash
  pip install angelslim==0.2.2
  ```

* DeepGEMM:
  To enable fp8 gemm for transformer. Only needed if you enable `--use_fp8_gemm true` in `run.sh`.
  ```bash
  git clone --recursive git@github.com:deepseek-ai/DeepGEMM.git
  cd DeepGEMM
  ./develop.sh
  ./install.sh
  ```

### 4. Download All Required Models

Download the pretrained HunyuanVideo-1.5 base model by following the [HunyuanVideo-1.5 download instructions](https://huggingface.co/tencent/HunyuanVideo-1.5#-download-pretrained-models). The pipeline uses the [480P-I2V model](https://huggingface.co/tencent/HunyuanVideo-1.5/tree/main/transformer/480p_i2v).

You also need the following models placed under the HunyuanVideo-1.5 model directory (`MODEL_PATH`):
- **Text encoder**: [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) → `text_encoder/llm/`
- **ByT5 encoder**: [google/byt5-small](https://huggingface.co/google/byt5-small) → `text_encoder/byt5-small/`
- **Glyph encoder**: [AI-ModelScope/Glyph-SDXL-v2](https://modelscope.cn/models/AI-ModelScope/Glyph-SDXL-v2) → `text_encoder/Glyph-SDXL-v2/`
- **Vision encoder**: [FLUX.1-Redux-dev](https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev) (gated, requires access) → `vision_encoder/siglip/`

**Alternatively**, we provide a download script that automatically downloads and organizes all required models:

```bash
python download_models.py --hf_token <your_huggingface_token>
```

**Important:** The vision encoder requires access to a gated model. Before running:
1. Request access at: https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev
2. Wait for approval (usually instant)
3. Create/get your access token at: https://huggingface.co/settings/tokens (select "Read" permission)

If you don't have FLUX access yet, you can skip the vision encoder (only for WAN pipeline; the HunyuanVideo pipeline requires the vision encoder):
```bash
python download_models.py --skip_vision_encoder
```

After download completes, the script will print the model paths to add to `run.sh`.

## 🎮 Quick Start

We provide a demo for the HY-World 1.5 model for quick start.

https://github.com/user-attachments/assets/643a33a4-b677-4eff-ad1d-32205c594274


Try our **online demo** without installation: https://3d.hunyuan.tencent.com/sceneTo3D

## 🧱 Model Checkpoints

| Model |  Description | Download |
|-------|--------------|----------|
| HY-World1.5-Bidirectional-480P-I2V | Bidirectional attention model with reconstituted context memory. | [Link](https://huggingface.co/tencent/HY-WorldPlay/tree/main/bidirectional_model) |
| HY-World1.5-Autoregressive-480P-I2V | Autoregressive model with reconstituted context memory. | [Link](https://huggingface.co/tencent/HY-WorldPlay/tree/main/ar_model) |
| HY-World1.5-Autoregressive-480P-I2V-rl | Autoregressive model with RL post-training. | [Link](https://huggingface.co/tencent/HY-WorldPlay/tree/main/ar_rl_model) |
| HY-World1.5-Autoregressive-480P-I2V-distill | Distilled autoregressive model optimized for fast inference (4 steps). | [Link](https://huggingface.co/tencent/HY-WorldPlay/tree/main/ar_distilled_action_model) |   
| HY-World1.5-Autoregressive-480P-I2V-rl-distill | Distilled autoregressive model with RL post-training. | To be released | 

<p align="center">
  <img src="assets/model_zoo.png">
</p>  

## 🔑 Inference

We provide two inference pipelines for WorldPlay:
1. **HunyuanVideo-based Pipeline** (recommended): Better action control and long-term memory, with HunyuanVideo-8B as backbone
2. **WAN Pipeline** (lightweight): Small VRAM but action control and long-term memory are compromised, with WAN-5B as backbone

### HunyuanVideo-based Inference

#### Configure Model Paths

After running `download_models.py`, update `run.sh` with the printed model paths:

```bash
# These paths are printed by download_models.py after download completes
MODEL_PATH=<path_printed_by_download_script>
AR_ACTION_MODEL_PATH=<path_printed_by_download_script>/ar_model/diffusion_pytorch_model.safetensors
AR_RL_ACTION_MODEL_PATH=<path_printed_by_download_script>/ar_rl_model/diffusion_pytorch_model.safetensors
BI_ACTION_MODEL_PATH=<path_printed_by_download_script>/bidirectional_model/diffusion_pytorch_model.safetensors
AR_DISTILL_ACTION_MODEL_PATH=<path_printed_by_download_script>/ar_distilled_action_model/diffusion_pytorch_model.safetensors
```

> **Note:** The action model paths (`AR_ACTION_MODEL_PATH`, etc.) should point to the **`.safetensors` file**, not the directory.

#### Configuration Options

In `run.sh`, you can configure:

| Parameter | Description                                                                                         |
|-----------|-----------------------------------------------------------------------------------------------------|
| `PROMPT` | Text description of the scene                                                                       |
| `IMAGE_PATH` | Input image path (required for I2V)                                                                 |
| `NUM_FRAMES` | Number of frames to generate (default: 125). **Important Note:** Must satisfy `(num_frames-1) % 4 == 0`. For bidirectional models: `[(num_frames-1) // 4 + 1] % 16 == 0`. For autoregressive models: `[(num_frames-1) // 4 + 1] % 4 == 0` |
| `N_INFERENCE_GPU` | Number of GPUs for parallel inference                                                               |
| `POSE` | Camera trajectory: pose string (e.g., `w-31` means generating `[1 + 31]` latents) or JSON file path |

#### Model Selection

Uncomment one of the three inference commands in `run.sh`:

1. **Bidirectional Model**:
   ```bash
   --action_ckpt $BI_ACTION_MODEL_PATH --model_type 'bi'
   ```

2. **Autoregressive Model**:
   ```bash
   --action_ckpt $AR_ACTION_MODEL_PATH --model_type 'ar'
   ```
  
3. **Autoregressive + RL Model**:
   ```bash
   --action_ckpt $AR_RL_ACTION_MODEL_PATH --model_type 'ar'
   ```

4. **Distilled Model**:
   ```bash
   --action_ckpt $AR_DISTILL_ACTION_MODEL_PATH --few_step true --num_inference_steps 4 --model_type 'ar'
   ```

#### Camera Trajectory Control

You have two options to control camera trajectories:

##### Option 1: Pose String (Recommended for Quick Testing)

Use intuitive pose strings by setting the `POSE` variable in `run.sh`:

```bash
POSE='w-31'
```

**Supported Actions:**
- **Movement**: `w` (forward), `s` (backward), `a` (left), `d` (right)
- **Rotation**: `up` (pitch up), `down` (pitch down), `left` (yaw left), `right` (yaw right)
- **Format**: `action-duration` where duration specifies the number of latents corresponding to the given action.

**Examples:**
```bash
# Move forward for 31 latents (default). Generate [1 + 31] latents
POSE='w-31'

# Move forward 3 latents, rotate right 1 latents, move right 4 latents. Generate [1 + 3 + 1 + 4] latents
POSE='w-3, right-1, d-4'

# Complex trajectory. Generate [1 + 2 + 1 + 2 + 4] latents
POSE='w-2, right-1, d-2, up-4'
```

##### Option 2: Custom JSON Files

For more complex trajectories, use `generate_custom_trajectory.py`:

```bash
python generate_custom_trajectory.py
```

Then set the JSON file path in `run.sh`:
```bash
POSE='./assets/pose/your_custom_trajectory.json'
```

#### Prompt Rewriting (Optional)

For better prompts, you can enable prompt rewriting with a vLLM server:

```bash
export T2V_REWRITE_BASE_URL="<your_vllm_server_base_url>"
export T2V_REWRITE_MODEL_NAME="<your_model_name>"
REWRITE=true  # in run.sh
```

#### Run Inference

After editing `run.sh` to configure your settings, run:

```bash
bash run.sh
```

---

### WAN Pipeline Inference

For detailed information about  WAN-based WorldPlay pipeline, please refer to [wan/README.md](wan/README.md).

## ⚙️Training

We provide a detailed documentation in [Training Documentation](./trainer/README.md).

## 📊 Evaluation

HY-World 1.5 surpasses existing methods across various quantitative metrics, including reconstruction metrics for different video lengths and human evaluations.

| Model                      | Real-time |  | | Short-term | | |  | | Long-term | | |
|:---------------------------| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|                            | | **PSNR** ⬆ | **SSIM** ⬆ | **LPIPS** ⬇ | **$R_{dist}$** ⬇ | **$T_{dist}$** ⬇ | **PSNR** ⬆ | **SSIM** ⬆ | **LPIPS** ⬇ | **$R_{dist}$** ⬇ | **$T_{dist}$** ⬇ |
| CameraCtrl                 | ❌ | 17.93 | 0.569 | 0.298 | 0.037 | 0.341 | 10.09 | 0.241 | 0.549 | 0.733 | 1.117 |
| SEVA                       | ❌ | 19.84 | 0.598 | 0.313 | 0.047 | 0.223 | 10.51 | 0.301 | 0.517 | 0.721 | 1.893 |
| ViewCrafter                | ❌ | 19.91 | 0.617 | 0.327 | 0.029 | 0.543 | 9.32 | 0.271 | 0.661 | 1.573 | 3.051 |
| Gen3C                      | ❌ | 21.68 | 0.635 | 0.278 | **0.024** | 0.477 | 15.37 | 0.431 | 0.483 | 0.357 | 0.979 |
| VMem                       | ❌ | 19.97 | 0.587 | 0.316 | 0.048 | 0.219 | 12.77 | 0.335 | 0.542 | 0.748 | 1.547 |
| Matrix-Game-2.0            | ✅ | 17.26 | 0.505 | 0.383 | 0.287 | 0.843 | 9.57 | 0.205 | 0.631 | 2.125 | 2.742 |
| GameCraft                  | ❌ | 21.05 | 0.639 | 0.341 | 0.151 | 0.617 | 10.09 | 0.287 | 0.614 | 2.497 | 3.291 |
| Ours (w/o Context Forcing) | ❌ | 21.27 | 0.669 | 0.261 | 0.033 | 0.157 | 16.27 | 0.425 | 0.495 | 0.611 | 0.991 |
| **Ours (full)**            | ✅ | **21.92** | **0.702** | **0.247** | 0.031 | **0.121** | **18.94** | **0.585** | **0.371** | **0.332** | **0.797** |




<p align="center">
  <img src="assets/human_eval.png">
</p>

## 🎬 More Examples

https://github.com/user-attachments/assets/51fcb28c-bd6e-44e5-adac-e3c6660f24f7

https://github.com/user-attachments/assets/b9060cd1-a442-4d67-9f16-daa7a2e6f2c8

https://github.com/user-attachments/assets/b883a748-cc77-480f-b6a0-e94b6ce9efea

## 📝 TODO
- [x] Open-source WorldCompass post-training framework
- [x] Open-source training code
- [x] Open-source quantized & accelerated inference
- [x] Open-source Lite model

## 📚 Citation

```bibtex
@article{hyworld2025,
  title={HY-World 1.5: A Systematic Framework for Interactive World Modeling with Real-Time Latency and Geometric Consistency},
  author={Team HunyuanWorld},
  journal={arXiv preprint},
  year={2025}
}

@article{worldplay2025,
    title={WorldPlay: Towards Long-Term Geometric Consistency for Real-Time Interactive World Model},
    author={Wenqiang Sun and Haiyu Zhang and Haoyuan Wang and Junta Wu and Zehan Wang and Zhenwei Wang and Yunhong Wang and Jun Zhang and Tengfei Wang and Chunchao Guo},
    year={2025},
    journal={arXiv preprint}
}

@article{wang2026worldcompass,
  title={WorldCompass: Reinforcement Learning for Long-Horizon World Models},
  author={Wang, Zehan and Wang, Tengfei and Zhang, Haiyu and Zuo, Xuhui and Wu, Junta and Wang, Haoyuan and Sun, Wenqiang and Wang, Zhenwei and Cao, Chenjie and Zhao, Hengshuang and others},
  journal={arXiv preprint},
  year={2026}
}
```


## Contact
Please send emails to tengfeiwang12@gmail.com if there is any question

## 🙏 Acknowledgements
We would like to thank [HunyuanWorld](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0), [HunyuanWorld-Mirror](https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror), [HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5), and [FastVideo](https://github.com/hao-ai-lab/FastVideo) for their great work.
