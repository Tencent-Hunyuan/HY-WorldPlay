<div align="center">
  <img src="assets/teaser.png">

<h1>🎮 HY-World 1.5: 实时延迟和几何一致性的交互式世界建模系统框架</h1>




</div>

<div align="center">
  <a href=https://3d.hunyuan.tencent.com/sceneTo3D?tab=worldplay target="_blank"><img src=https://img.shields.io/badge/Official%20Site-333399.svg?logo=homepage height=22px></a>
  <a href=https://huggingface.co/tencent/HunyuanWorld-1 target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg height=22px></a>
  <a href=https://3d-models.hunyuan.tencent.com/world/ target="_blank"><img src= https://img.shields.io/badge/Page-bb8a2e.svg?logo=github height=22px></a>
  <a href=https://arxiv.org/abs/2507.21809 target="_blank"><img src=https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv height=22px></a>
  <a href=https://discord.gg/dNBrdrGGMa target="_blank"><img src= https://img.shields.io/badge/Discord-white.svg?logo=discord height=22px></a>
  <a href=https://x.com/TencentHunyuan target="_blank"><img src=https://img.shields.io/badge/Tencent%20HY-black.svg?logo=x height=22px></a>
 <a href="#community-resources" target="_blank"><img src=https://img.shields.io/badge/Community-lavender.svg?logo=homeassistantcommunitystore height=22px></a>
</div>

[//]: # (  <a href=# target="_blank"><img src=https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv height=22px></a>)

[//]: # (  <a href=# target="_blank"><img src= https://img.shields.io/badge/Colab-8f2628.svg?logo=googlecolab height=22px></a>)

[//]: # (  <a href="#"><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/v/mulankit?logo=pypi"  height=22px></a>)

<br>

<p align="center">
  <i>"一掌握无限，一瞬即永恒"</i>
</p>

## 🔥 新闻
- 2025年12月17日: 👋 我们发布了 HY-World 1.5 (WorldPlay) 的[技术报告](https://arxiv.org/abs/2507.21809)和[研究论文](https://arxiv.org/abs/2507.21809)，欢迎查看详情并展开讨论！
- 2025年12月17日: 🤗 我们发布了首个开源、实时交互、长期几何一致性的世界模型 HY-World 1.5 (WorldPlay)！

> 加入我们的 **[微信群](#)** 和 **[Discord](https://discord.gg/dNBrdrGGMa)** 群组进行讨论。

| 微信群                                     | 小红书                                           | X                                           | Discord                                           |
|--------------------------------------------------|-------------------------------------------------------|---------------------------------------------|---------------------------------------------------|
| <img src="assets/qrcode/wechat.png"  height=140> | <img src="assets/qrcode/xiaohongshu.png"  height=140> | <img src="assets/qrcode/x.png"  height=140> | <img src="assets/qrcode/discord.png"  height=140> |


## 📋 目录
- [🔥 新闻](#-新闻)
- [📋 目录](#-目录)
- [📖 介绍](#-介绍)
- [✨ 亮点](#-亮点)
- [📜 系统要求](#-系统要求)
- [🛠️ 依赖与安装](#️-依赖与安装)
- [🎮 快速开始](#-快速开始)
- [🧱 下载预训练模型](#-下载预训练模型)
- [🔑 推理](#-推理)
- [📊 评估](#-评估)
- [🎬 更多示例](#-更多示例)
- [📚 引用](#-引用)
- [🙏 致谢](#-致谢)

## 📖 介绍
我们之前发布了 **HY-World 1.0**，能够生成沉浸式 3D 世界，但它依赖漫长的离线生成过程，缺乏实时交互能力。**HY-World 1.5** 通过 **WorldPlay** 弥补了这一差距，这是一个流式视频扩散模型，能够实现具有长期几何一致性的实时交互式世界建模，解决了当前方法速度和内存之间的权衡。我们的模型源自四个关键设计。1) 使用双重动作表示来实现对用户键盘和鼠标输入的强大动作控制。2) 为了强制长期一致性，我们的重构上下文记忆、从过去的帧动态重建上下文，并使用时间重构保持几何上重要但时间久远的帧可访问性，有效缓解记忆衰减。3) 我们设计了 WorldCompass，这是一个新颖的强化学习(RL)后训练框架，旨在直接改善长时域自回归视频模型的动作跟随和视觉质量。4) 我们还提出了上下文强制，这是一种为记忆感知模型设计的新颖蒸馏方法。在教师和学生之间对齐记忆上下文保留了学生使用长距离信息的能力，实现了实时速度的同时，可以防止误差漂移。总的来说，HY-World 1.5 以 24 FPS 生成一致性的长时域流式视频，与现有技术相比表现优异。我们的模型在不同场景中展现出强大的泛化能力，支持真实世界和风格化环境中的第一人称和第三人称视角，实现了 3D 重建、可提示事件和无限世界扩展等多样化应用。

<p align="center">
  <img src="assets/teaser_2.png">
</p>

## ✨ 亮点

- **系统性概览**

  HY-World 1.5 开源了一个系统全面的实时世界模型训练框架，涵盖整个流程和所有阶段，包括数据、训练和推理部署。技术报告介绍了模型预训练、中期训练、强化学习后训练和记忆感知模型蒸馏的详细训练细节。此外，报告介绍了一系列旨在减少网络传输延迟和模型推理延迟的工程技术，从而为用户实现实时流式推理体验。

<p align="center">
  <img src="assets/overview.png">
</p>

- **推理流程**

  给定单张图像或文本提示来描述一个世界，我们的模型执行下一块（16个视频帧）预测任务，根据用户的动作生成未来视频。对于每个块的生成，我们从过去的块动态重构上下文记忆，以强制长期时间和几何一致性。

<p align="center">
  <img src="assets/pipeline.png">
</p>



## 📜 系统要求

- **GPU**: 支持 CUDA 的 NVIDIA GPU
- **最小 GPU 内存**: 14 GB（启用模型卸载）

  > **注意:** 上述内存要求是在启用模型卸载的情况下测量的。如果您的 GPU 有足够的内存，您可以禁用卸载以提高推理速度。


## 🛠️ 依赖与安装
```bash
conda create --name worldplay python=3.10 -y
conda activate worldplay
pip install -r requirements.txt
```

- Flash Attention: 安装 Flash Attention 以获得更快的推理速度和更低的 GPU 内存消耗。详细的安装说明可参考 [Flash Attention](https://github.com/Dao-AILab/flash-attention)。


## 🎮 快速开始

我们提供了 HY-World 1.5 模型的演示供快速开始。

https://github.com/user-attachments/assets/63e5e5ec-34b2-4160-b7d2-4dd18cf25d71


免安装试用我们的**在线服务**: https://3d.hunyuan.tencent.com/sceneTo3D

## 🧱 下载预训练模型
我们提供了使用混元视频-1.5 的实现，这是最强大的开源视频扩散模型之一。模型权重可在 xxx 中找到。

|模型名称| 下载                     |\
|-|-------------------------------------------|
HY-World1.5-Bidirectional-480P-I2V |     |
HY-World1.5-Autoregressive-480P-I2V |    |
HY-World1.5-Autoregressive-480P-I2V-distill |     |

## 🔑 推理
我们开源了双向和自回归扩散模型的推理代码。对于提示重写，我们建议使用 Gemini 或通过 vLLM 部署的模型。此代码库目前仅支持与 vLLM API 兼容的模型。如果您希望使用 Gemini，您需要实现自己的接口调用。详情可参考 [HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5)。

我们建议使用 `generate_custom_trajectory.py` 生成自定义相机轨迹。

```bash
export T2V_REWRITE_BASE_URL="<your_vllm_server_base_url>"
export T2V_REWRITE_MODEL_NAME="<your_model_name>"
export I2V_REWRITE_BASE_URL="<your_vllm_server_base_url>"
export I2V_REWRITE_MODEL_NAME="<your_model_name>"

PROMPT='一条铺好的小路通向一座横跨平静水面的石拱桥。郁郁葱葱的绿树和植被沿着小路和水的远岸排列。一座传统风格的凉亭，带有分层的红褐色屋顶，坐落在远岸。水面倒映着周围的绿色植物和天空。场景沐浴在柔和的自然光中，营造出宁静祥和的氛围。小路由大块的矩形石头组成，桥梁由浅灰色石头建造。整体构图强调了景观的和平与和谐。'

IMAGE_PATH=./assets/img/test.png # 现在我们只提供 i2v 模型，所以路径不能为 None
SEED=1
ASPECT_RATIO=16:9
RESOLUTION=480p                  # 现在我们只提供 480p 模型
OUTPUT_PATH=./outputs/
MODEL_PATH=                      # 预训练 hunyuanvideo-1.5 模型的路径
AR_ACTION_MODEL_PATH=            # 我们的 HY-World 1.5 自回归模型权重的路径
BI_ACTION_MODEL_PATH=            # 我们的 HY-World 1.5 双向模型权重的路径
AR_DISTILL_ACTION_MODEL_PATH=    # 我们的 HY-World 1.5 自回归蒸馏模型权重的路径
POSE_JSON_PATH=./assets/pose/test_forward_32_latents.json   # 自定义相机轨迹的路径
NUM_FRAMES=125

# 更快推理的配置
# 对于 AR 推理，建议的最大数量是 4。对于双向模型，可以设置为 8。
N_INFERENCE_GPU=4 # 并行推理 GPU 数量。

# 更好质量的配置
REWRITE=false # 启用提示重写。请确保重写 vLLM 服务器已部署和配置。
ENABLE_SR=false # 启用超分辨率。当 NUM_FRAMES == 121 时，您可以将其设置为 true

# 使用双向模型推理
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
  --model_type 'bi'

# 使用自回归模型推理
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
#  --model_type 'ar'

# 使用自回归蒸馏模型推理
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
#  --model_type 'ar'
```


## 📊 评估

HY-World 1.5 在各种定量指标上超越现有方法，包括不同视频长度的重建指标和人工评估。

| 模型                      | 实时 |  | | 短期 | | |  | | 长期 | | |
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

## 🎬 更多示例

https://github.com/user-attachments/assets/6aac8ad7-3c64-4342-887f-53b7100452ed

https://github.com/user-attachments/assets/531bf0ad-1fca-4d76-bb65-84701368926d

https://github.com/user-attachments/assets/f165f409-5a74-4e19-a32c-fc98d92259e1

## 📚 引用

```bibtex
@article{hyworld2025,
  title={HY-World 1.5: A Systematic Framework for Interactive World Modeling with Real-Time Latency and Geometric Consistency},
  author={Team HunyuanWorld},
  journal={arXiv preprint},
  year={2025}
}
```


## 🙏 致谢
我们要感谢 [HunyuanWorld](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0)、[HunyuanWorld-Mirror
](https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror)、[FlashWorld](https://github.com/imlixinyang/FlashWorld)、[HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) 和 [FastVideo](https://github.com/hao-ai-lab/FastVideo) 的出色工作。
