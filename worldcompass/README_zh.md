## WorldCompass 简介

WorldCompass 借鉴了GRPO的思想，旨在提升自回归视频生成中的动作跟随能力与视觉质量。

## 训练指南

训练流程包括：

1. **环境配置** - 安装依赖
2. **模型下载** - 下载预训练检查点
3. **数据集准备** - 准备图文潜变量及动作轨迹
4. **训练** - 运行强化学习训练

---

### 第一步：环境配置

---

**1.1 创建 Conda 环境**

```bash
# 创建并激活环境
conda create -n worldcompass python=3.10 -y
conda activate worldcompass

# 安装依赖
pip install -r requirements.txt
pip install transformers==4.50.0

# 安装 Flash Attention（推荐，可加速训练）
pip install flash-attn==2.7.3 --no-build-isolation
```

---

### 第二步：模型准备

---

**2.1 下载模型检查点**

```bash
# 下载所有所需模型
python download_models_worldcompass.py --hf_token <your_huggingface_token> --cache_dir <your_cache_dir>
```

**注意**：您需要获得 FLUX.1-Redux-dev 的访问权限以使用视觉编码器：

1. 申请访问权限：https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev
2. 创建 Token：https://huggingface.co/settings/tokens（选择"Read"权限）

该脚本将下载以下内容：

- **HunyuanVideo-1.5** 基础模型（VAE、调度器、480p Transformer）
- **HY-WorldPlay** 动作模型（AR 模型、双向模型、蒸馏模型）
- **Qwen2.5-VL-7B-Instruct** 文本编码器
- **ByT5** 编码器（byt5-small + Glyph-SDXL-v2）
- **SigLIP** 视觉编码器（来自 FLUX.1-Redux-dev）
- **DepthAnythingV3** 相机位姿估计模型 - 方案一
- **Hunyuan-WorldMirror** 相机位姿估计模型 - 方案二

---

**2.2 克隆 DepthAnythingV3 仓库**

```bash
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git DepthAnythingV3
mv ./DepthAnythingV3/src/depth_anything_3 ./depth_anything_3
```

---

### 第三步：数据集准备

---

**3.1 准备输入数据**

创建一个包含训练数据的 JSON 文件：

```json
[
  {
    "image_path": "/path/to/image1.jpg",
    "caption": "A serene park with trees and a bridge over water"
  },
  {
    "image_path": "/path/to/image2.png",
    "caption": "A modern city street at sunset"
  }
]
```

---

**3.2 提取图文潜变量**

该步骤使用 VAE 和文本编码器将图像与文本编码为潜变量特征。

**单卡**：

```bash
python prepare_dataset/prepare_image_text_latent_simple.py \
    --input_json /path/to/train.json \
    --output_dir /path/to/train_latents \
    --hunyuan_checkpoint_path /path/to/hunyuanvideo_1_5
```

**多卡（推荐）**：

```bash
torchrun --nproc_per_node=8 prepare_dataset/prepare_image_text_latent_simple.py \
    --input_json /path/to/train.json \
    --output_dir /path/to/train_latents \
    --hunyuan_checkpoint_path /path/to/hunyuanvideo_1_5
```

**输出结构**：

```
/path/to/train_latents/
├── latents/
│   ├── 0_000000.pt
│   ├── 0_000001.pt
│   └── ...
└── latents.json          # 训练索引文件
```

每个 `.pt` 文件包含：

- `latent`：VAE 编码的图像特征 [1, C, T, H, W]
- `image_cond`：首帧条件 [1, C, 1, H, W]
- `prompt_embeds`：文本嵌入 [1, L, D]
- `prompt_mask`：文本注意力掩码 [1, L]
- `vision_states`：视觉特征 [1, N, D]
- `byt5_text_states`：ByT5 文本特征 [1, 256, 1472]
- `byt5_text_mask`：ByT5 注意力掩码 [1, 256]

---

**3.3 准备评估数据**

为了在训练过程中在固定子集上观察模型性能，建议构建一个小型评估数据集。JSON 文件格式与训练数据相同。推荐的样本数量应能被预期的 GPU 数量整除，通常 16 或 32 个样本即可满足效率需求。

```bash
python prepare_dataset/prepare_image_text_latent_simple.py \
    --input_json /path/to/eval.json \
    --output_dir /path/to/eval_latents \
    --hunyuan_checkpoint_path /path/to/hunyuanvideo_1_5
```

---

**3.4 生成动作轨迹**

为训练生成随机相机轨迹序列：

```bash
python prepare_dataset/prepare_custom_action.py
```

**默认输出**：`prepare_dataset/harder_random_poses.json`（1000 条轨迹，每条 128 个动作，默认设置倾向于生成复合动作）

**自定义**：编辑 `prepare_custom_action.py` 以调整轨迹数量、每条轨迹的动作数量以及轨迹模板合成规则。

---

### 第四步：开始训练

---

**4.1 配置训练脚本**

修改 `scripts/train_worldcompass.sh` 中标有 TODO 的配置项：

```bash

export MASTER_ADDR=""       # [TODO] 设置主节点地址

# 注意：运行 download_models.py（第二步）可获取以下变量的精确路径。
CACHE_DIR=""              # [TODO] 填写整体检查点缓存目录（由 download_models.py 输出）
HUNYUAN_CHECKPOINT=""     # [TODO] 填写 HunYuan 检查点目录
WORLDPLAY_CHECKPOINT=""   # [TODO] 填写 WorldPlay 检查点目录

TRAIN_LATENTS_DIR=""      # [TODO] 训练潜变量目录路径
EVAL_LATENTS_DIR=""       # [TODO] 评估潜变量目录路径
POSE_PATH=""              # [TODO] 自定义动作位姿 JSON 文件路径
OUTPUT_DIR=""             # [TODO] 输出目录路径

--wandb_key ""            # [TODO] 设置 wandb key
--wandb_entity ""         # [TODO] 设置 wandb entity
```

---

**4.2 开始训练！！！**

在 8 张 GPU 上训练可以初步看到改善效果，但使用更多节点通常能带来更稳定的训练过程和更好的最终结果。

**单节点（8 张 GPU）**：

```bash
bash scripts/train_worldcompass.sh 0 1
```

**多节点训练**：

```bash
# 在每个节点上以相应的 rank 运行
# 节点 0（主节点）：
bash scripts/train_worldcompass.sh 0 4  # 共 4 个节点

# 节点 1：
bash scripts/train_worldcompass.sh 1 4

# 节点 2：
bash scripts/train_worldcompass.sh 2 4

# 节点 3：
bash scripts/train_worldcompass.sh 3 4
```

**注意**：请确保所有节点均可通过 `MASTER_ADDR` 和 `MASTER_PORT` 相互通信。

**4.3 监控训练**

训练日志和检查点将保存至：

```
${CKPT_DIR}/${exp_name}/
├── checkpoint-{step}/
│   ├── transformer/
│   │   └── diffusion_pytorch_model.safetensors
│   └── training_state.pt
└── generated_videos/           # 训练过程中生成的样例视频
```

如果已配置 WandB，指标将记录到您的 WandB 仪表盘。

### 4.4 硬件要求

我们的实验在具有 96GB 显存的 GPU 上进行。如果遇到 GPU 显存不足（OOM）问题，可以考虑：

- 减小 `window_frames`
- 将 VAE 转换为 `torch.bf16`（可能导致训练不稳定）

## 结果

我们在WorldPlay基准模型上进行了全面的验证。评估结果表明，在经过 WorldCompass 后训练后，该模型的能力实现了明显提升，特别是在交互指令遵循和视觉质量这两个方面。

<p align="center">
  <img src="assets/results.png">
</p>
<p align="center">
  <img src="assets/image.png">
</p>

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

## 联系方式

如有任何问题，请发送邮件至 tengfeiwang12@gmail.com

## 🙏 致谢

我们衷心感谢
[HunyuanWorld](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0)、
[HunyuanWorld-Mirror](https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror)、
[HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5)、
[DiffusionNFT](https://github.com/NVlabs/DiffusionNFT) 以及
[FastVideo](https://github.com/hao-ai-lab/FastVideo) 的出色工作。
