<div align="center">
  <img src="assets/teaser.webp">

<h1>🎮 HY-World 1.5: 实时、几何一致的交互式世界建模系统框架</h1>

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
  <i>"一掌握无限，一瞬即永恒"</i>
</p>

## 🎥 宣传视频

https://github.com/user-attachments/assets/19a8ef47-3d1a-4c29-a80f-500b342bfc80

## 🔥 新闻
- 2026年3月8日: 🚀 我们发布了强化学习后训练框架 [WorldCompass](worldcompass/README.md)！
- 2026年1月6日: 🚀 我们开源了WorldPlay-8B模型（基于HunyuanVideo）的训练代码，让社区能够训练和微调自己的世界模型！
- 2026年1月6日: 🎯 我们开源了WorldPlay-5B模型（基于WAN），部署需要显存更少！（但控制和记忆能力有退化）
- 2026年1月3日: ⚡ 我们更新了推理代码的量化及加速版本，推理速度更快！
- 2025年12月17日: 👋 我们发布了 HY-World 1.5 (WorldPlay) 的[技术报告](https://3d-models.hunyuan.tencent.com/world/world1_5/HYWorld_1.5_Tech_Report.pdf)（和[研究论文](https://arxiv.org/abs/2512.14614)），欢迎查看详情并展开讨论！
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
- [📝 待办事项](#-待办事项)
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

- HunyuanVideo-1.5 基础模型: 按照 [HunyuanVideo-1.5 下载说明](https://huggingface.co/tencent/HunyuanVideo-1.5#-download-pretrained-models) 下载预训练的 HunyuanVideo-1.5 模型。在使用 HY-World 1.5 权重之前,需要先下载此基础模型，其中使用的是 [480P-I2V 模型](https://huggingface.co/tencent/HunyuanVideo-1.5/tree/main/transformer/480p_i2v)。


## 🎮 快速开始

我们提供了 HY-World 1.5 模型的演示供快速开始。

https://github.com/user-attachments/assets/643a33a4-b677-4eff-ad1d-32205c594274


免安装试用我们的**在线服务**: https://3d.hunyuan.tencent.com/sceneTo3D

## 🧱 模型权重

| 模型 | 描述 | 下载 |
|------|------|------|
| HY-World1.5-Bidirectional-480P-I2V | 双向注意力模型，具有重构上下文记忆机制 | [下载地址](https://huggingface.co/tencent/HY-WorldPlay/tree/main/bidirectional_model) |
| HY-World1.5-Autoregressive-480P-I2V | 自回归模型，具有重构上下文记忆机制和单向的注意力机制以实现长期几何一致性 | [下载地址](https://huggingface.co/tencent/HY-WorldPlay/tree/main/ar_model) |
| HY-World1.5-Autoregressive-480P-I2V-rl | 经过强化学习的自回归模型. | [下载地址](https://huggingface.co/tencent/HY-WorldPlay/tree/main/ar_rl_model) |
| HY-World1.5-Autoregressive-480P-I2V-distill | 自回归模型的蒸馏版，针对推理优化（4步） | [下载地址](https://huggingface.co/tencent/HY-WorldPlay/tree/main/ar_distilled_action_model) |
| HY-World1.5-Autoregressive-480P-I2V-rl-distill |  经过强化学习的自回归模型的蒸馏版. | To be released | 

<p align="center">
  <img src="assets/model_zoo.png">
</p>  

## 🔑 推理

我们提供两种推理管道:
1. **基于 HunyuanVideo 的管道** (推荐): 基于HunyuanVideo模型，并支持更好的动作控制和更一致的记忆能力
2. **WAN 管道** (轻量级): 基于WAN模型，需要显存更少，但动作控制和记忆能力退化
### 基于 HunyuanVideo 的推理

#### 配置模型路径

运行 `download_models.py` 后，使用打印的模型路径更新 `run.sh`：

```bash
# 这些路径由 download_models.py 在下载完成后打印
MODEL_PATH=<download_script打印的路径>
AR_ACTION_MODEL_PATH=<download_script打印的路径>/ar_model
AR_RL_ACTION_MODEL_PATH=<download_script打印的路径>/ar_rl_model
BI_ACTION_MODEL_PATH=<download_script打印的路径>/bidirectional_model
AR_DISTILL_ACTION_MODEL_PATH=<download_script打印的路径>/ar_distilled_action_model
```

#### 配置选项

在 `run.sh` 中，您可以配置：

| 参数 | 描述                                                   |
|------|------------------------------------------------------|
| `PROMPT` | 场景的文本描述                                              |
| `IMAGE_PATH` | 输入图像路径（I2V 必需）                                       |
| `NUM_FRAMES` | 要生成的帧数（默认：125）。**重要说明：** 必须满足 `(num_frames-1) % 4 == 0`。对于双向模型须满足：`[(num_frames-1) // 4 + 1] % 16 == 0`。对于单向模型须满足：`[(num_frames-1) // 4 + 1] % 4 == 0` |
| `N_INFERENCE_GPU` | 并行推理的 GPU 数量                                         |
| `POSE` | 相机轨迹：姿态字符串（如 `w-31`代表生成`[1 + 31]`latents）或 JSON 文件路径 |

#### 模型选择

在 `run.sh` 中取消注释三个推理命令之一：

1. **双向模型**：
   ```bash
   --action_ckpt $BI_ACTION_MODEL_PATH --model_type 'bi'
   ```

2. **自回归模型**：
   ```bash
   --action_ckpt $AR_ACTION_MODEL_PATH --model_type 'ar'
   ```

3. **自回归+强化学习模型**：
   ```bash
   --action_ckpt $AR_RL_ACTION_MODEL_PATH --model_type 'ar'
   ```

4. **蒸馏模型**：
   ```bash
   --action_ckpt $AR_DISTILL_ACTION_MODEL_PATH --few_step true --num_inference_steps 4 --model_type 'ar'
   ```

#### 相机轨迹控制

您有两种方式来控制相机轨迹：

##### 方式 1：姿态字符串（推荐用于快速测试）

在 `run.sh` 中设置 `POSE` 变量使用直观的姿态字符串：

```bash
POSE='w-31'
```

**支持的动作：**
- **移动**: `w` (前进), `s` (后退), `a` (左移), `d` (右移)
- **旋转**: `up` (俯仰向上), `down` (俯仰向下), `left` (偏航向左), `right` (偏航向右)
- **格式**: `动作-时长`，时长代表动作对应的latents数量

**示例：**
```bash
# 前进31个latents (默认), 总共生成[1 + 31]个latents
POSE='w-31'

# 前进3个latents, 向右旋转1个latent, 向右走4个latents, 总共生成[1 + 3 + 1 + 4]个latents
POSE='w-3, right-1, d-4'

# 复杂轨迹, 总共生成[1 + 2 + 1 + 2 + 4]个latents
POSE='w-2, right-1, d-2, up-4'
```

##### 方式 2：自定义 JSON 文件

对于更复杂的轨迹，使用 `generate_custom_trajectory.py`：

```bash
python generate_custom_trajectory.py
```

然后在 `run.sh` 中设置 JSON 文件路径：
```bash
POSE='./assets/pose/your_custom_trajectory.json'
```

#### 提示重写（可选）

为获得更好的提示，您可以使用 vLLM 服务器启用提示重写：

```bash
export T2V_REWRITE_BASE_URL="<your_vllm_server_base_url>"
export T2V_REWRITE_MODEL_NAME="<your_model_name>"
REWRITE=true  # 在 run.sh 中
```

#### 运行推理

在编辑 `run.sh` 配置设置后，运行：

```bash
bash run.sh
```

---

### WAN 管道推理

关于 WAN 管道（轻量级替代模型）的详细信息，请参阅 [wan/README_zh.md](wan/README_zh.md)。

---

配置好任一管道后，您就可以开始生成视频了！


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

## 📝 待办事项
- [x] 开源WorldCompass后训练框架
- [x] 开源训练代码
- [x] 开源量化及工程优化的推理代码
- [x] 开源轻量级模型

## 📚 引用

```bibtex
@article{hyworld2025,
  title={HY-World 1.5: A Systematic Framework for Interactive World Modeling with Real-Time Latency and Geometric Consistency},
  author={Team HunyuanWorld},
  journal={arXiv preprint},
  year={2025}
}

@article{wang2026worldcompass,
  title={WorldCompass: Reinforcement Learning for Long-Horizon World Models},
  author={Wang, Zehan and Wang, Tengfei and Zhang, Haiyu and Zuo, Xuhui and Wu, Junta and Wang, Haoyuan and Sun, Wenqiang and Wang, Zhenwei and Cao, Chenjie and Zhao, Hengshuang and others},
  journal={arXiv preprint},
  year={2026}
}

@article{worldplay2025,
    title={WorldPlay: Towards Long-Term Geometric Consistency for Real-Time Interactive World Model},
    author={Wenqiang Sun and Haiyu Zhang and Haoyuan Wang and Junta Wu and Zehan Wang and Zhenwei Wang and Yunhong Wang and Jun Zhang and Tengfei Wang and Chunchao Guo},
    year={2025},
    journal={arXiv preprint}
}

```


## 🙏 致谢
我们要感谢 [HunyuanWorld](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0)、[HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) 和 [FastVideo](https://github.com/hao-ai-lab/FastVideo) 的出色工作。
