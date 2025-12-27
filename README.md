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
- December 17, 2025: 👋 We present the [technical report](https://3d-models.hunyuan.tencent.com/world/world1_5/HYWorld_1.5_Tech_Report.pdf) and [research paper](https://arxiv.org/abs/2512.14614) of HY-World 1.5 (WorldPlay), please check out the details and spark some discussion!
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
  - [2. Install Flash Attention (Optional but Recommended)](#2-install-flash-attention-optional-but-recommended)
  - [3. Download All Required Models](#3-download-all-required-models)
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
- **Minimum GPU Memory**: 14 GB (with model offloading enabled)
  
  > **Note:** The memory requirements above are measured with model offloading enabled. If your GPU has sufficient memory, you may disable offloading for improved inference speed.


## 🛠️ Dependencies and Installation

### 1. Create Environment
```bash
conda create --name worldplay python=3.10 -y
conda activate worldplay
pip install -r requirements.txt
```

### 2. Install Flash Attention (Optional but Recommended)
Install Flash Attention for faster inference and reduced GPU memory consumption:
```bash
pip install flash-attn --no-build-isolation
```
Detailed instructions: [Flash Attention](https://github.com/Dao-AILab/flash-attention)

### 3. Download All Required Models

We provide a download script that automatically downloads all required models:

```bash
python download_models.py --hf_token <your_huggingface_token>
```

**Important:** The vision encoder requires access to a gated model. Before running:
1. Request access at: https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev
2. Wait for approval (usually instant)
3. Create/get your access token at: https://huggingface.co/settings/tokens (select "Read" permission)

If you don't have FLUX access yet, you can skip the vision encoder:
```bash
python download_models.py --skip_vision_encoder
```

The script downloads:
- **HY-WorldPlay** action models (~32GB each)
- **HunyuanVideo-1.5** base model (vae, scheduler, 480p transformer)
- **Qwen2.5-VL-7B-Instruct** text encoder (~15GB)
- **ByT5** encoders (byt5-small + Glyph-SDXL-v2)
- **SigLIP** vision encoder (from FLUX.1-Redux-dev)

After download completes, the script will print the model paths to add to `run.sh`.

## 🎮 Quick Start

We provide a demo for the HY-World 1.5 model for quick start.

https://github.com/user-attachments/assets/643a33a4-b677-4eff-ad1d-32205c594274


Try our **online demo** without installation: https://3d.hunyuan.tencent.com/sceneTo3D

## 🧱 Model Checkpoints

| Model |  Description | Download |
|-------|--------------|----------|
| HY-World1.5-Bidirectional-480P-I2V | Bidirectional attention model with full context awareness. | [Link](https://huggingface.co/tencent/HY-WorldPlay/tree/main/bidirectional_model) |
| HY-World1.5-Autoregressive-480P-I2V | Autoregressive model with reconstituted context memory. | [Link](https://huggingface.co/tencent/HY-WorldPlay/tree/main/ar_model) |
| HY-World1.5-Autoregressive-480P-I2V-rl | Autoregressive model with RL post-training. | To be released |
| HY-World1.5-Autoregressive-480P-I2V-distill | Distilled autoregressive model optimized for fast inference (4 steps). | [Link](https://huggingface.co/tencent/HY-WorldPlay/tree/main/ar_distilled_action_model) |   
| HY-World1.5-Autoregressive-480P-I2V-rl-distill | Distilled autoregressive model with RL post-training. | To be released | 

<p align="center">
  <img src="assets/model_zoo.png">
</p>  

## 🔑 Inference

### Configure Model Paths

After running `download_models.py`, update `run.sh` with the printed model paths:

```bash
# These paths are printed by download_models.py after download completes
MODEL_PATH=<path_printed_by_download_script>
AR_ACTION_MODEL_PATH=<path_printed_by_download_script>/ar_model
BI_ACTION_MODEL_PATH=<path_printed_by_download_script>/bidirectional_model
AR_DISTILL_ACTION_MODEL_PATH=<path_printed_by_download_script>/ar_distilled_action_model
```

### Configuration Options

In `run.sh`, you can configure:

| Parameter | Description |
|-----------|-------------|
| `PROMPT` | Text description of the scene |
| `IMAGE_PATH` | Input image path (required for I2V) |
| `NUM_FRAMES` | Number of frames to generate (default: 125) |
| `N_INFERENCE_GPU` | Number of GPUs for parallel inference |
| `POSE` | Camera trajectory: pose string (e.g., `'w-3, right-0.5'`) or JSON file path |

### Model Selection

Uncomment one of the three inference commands in `run.sh`:

1. **Bidirectional Model**:
   ```bash
   --action_ckpt $BI_ACTION_MODEL_PATH --model_type 'bi'
   ```

2. **Autoregressive Model**:
   ```bash
   --action_ckpt $AR_ACTION_MODEL_PATH --model_type 'ar'
   ```

3. **Distilled Model**:
   ```bash
   --action_ckpt $AR_DISTILL_ACTION_MODEL_PATH --few_step true --num_inference_steps 4 --model_type 'ar'
   ```

### Camera Trajectory Control

You have two options to control camera trajectories:

#### Option 1: Pose String (Recommended for Quick Testing)

Use intuitive pose strings by setting the `POSE` variable in `run.sh`:

```bash
POSE='w-3, right-0.5, d-4'
```

**Supported Actions:**
- **Movement**: `w` (forward), `s` (backward), `a` (left), `d` (right)
- **Rotation**: `up` (pitch up), `down` (pitch down), `left` (yaw left), `right` (yaw right)
- **Format**: `action-duration` where duration is in seconds

**Examples:**
```bash
# Move forward for 6 seconds (default)
POSE='w-6'

# Move forward 3s, rotate right 0.5s, move right 4s
POSE='w-3, right-0.5, d-4'

# Complex trajectory
POSE='w-2, right-1, d-1.5, up-0.5'
```

#### Option 2: Custom JSON Files

For more complex trajectories, use `generate_custom_trajectory.py`:

```bash
python generate_custom_trajectory.py
```

Then set the JSON file path in `run.sh`:
```bash
POSE='./assets/pose/your_custom_trajectory.json'
```

### Prompt Rewriting (Optional)

For better prompts, you can enable prompt rewriting with a vLLM server:

```bash
export T2V_REWRITE_BASE_URL="<your_vllm_server_base_url>"
export T2V_REWRITE_MODEL_NAME="<your_model_name>"
REWRITE=true  # in run.sh
```

### Run Inference

After editing `run.sh` to configure your settings, run:

```bash
bash run.sh
```


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

https://github.com/user-attachments/assets/6aac8ad7-3c64-4342-887f-53b7100452ed

https://github.com/user-attachments/assets/531bf0ad-1fca-4d76-bb65-84701368926d

https://github.com/user-attachments/assets/f165f409-5a74-4e19-a32c-fc98d92259e1

## 📝 TODO
- [ ] Acceleration & Quantization
- [ ] Open-source training code

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

@inproceedings{wang2025compass,
  title={WorldCompass: Reinforcement Learning for Long-Horizon World Models},
  author={Zehan Wang and Tengfei Wang and Haiyu Zhang and Wenqiang Sun and Junta Wu and Haoyuan Wang and Zhenwei Wang and Hengshuang Zhao and Chunchao Guo and Zhou Zhao},
  journal = {arXiv preprint},
  year = {2025}
}
```


## Contact
Please send emails to tengfeiwang12@gmail.com if there is any question

## 🙏 Acknowledgements
We would like to thank [HunyuanWorld](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0), [HunyuanWorld-Mirror](https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror), [HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5), and [FastVideo](https://github.com/hao-ai-lab/FastVideo) for their great work.
