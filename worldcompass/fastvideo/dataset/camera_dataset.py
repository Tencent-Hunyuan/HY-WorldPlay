import json
import os
import sys

sys.path.append(os.path.abspath("."))
import torch
import pandas as pd
import numpy as np
import random
from pathlib import Path

from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from scipy.spatial.transform import Rotation as R

from fastvideo.distributed import (
    get_sp_world_size,
    get_world_group,
    get_world_rank,
    get_world_size,
    get_gpu_world_size,
    get_gpu_group,
)
from fastvideo.logger import init_logger

logger = init_logger(__name__)


def camera_center_normalization(w2c, nframe):
    c2w_view0 = w2c[::nframe].inverse()  # [B,4,4]
    c2w_view0 = c2w_view0.repeat_interleave(nframe, dim=0)  # [BF,4,4]
    w2c = c2w_view0 @ w2c
    return w2c


# action keys
ACTION_KEYS = [
    "D",
    "DL",
    "DR",
]


class DP_SP_BatchSampler(Sampler[list[int]]):
    """A simple sequential batch sampler that yields batches of indices."""

    def __init__(
        self,
        batch_size: int,
        dataset_size: int,
        num_sp_groups: int,
        sp_world_size: int,
        global_rank: int,
        drop_last: bool = True,
        drop_first_row: bool = False,
        seed: int = 0,
    ):
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.drop_last = drop_last
        self.seed = seed
        self.num_sp_groups = num_sp_groups
        self.global_rank = global_rank
        self.sp_world_size = sp_world_size

        # ── epoch-level RNG ────────────────────────────────────────────────
        rng = torch.Generator().manual_seed(self.seed)
        # Create a random permutation of all indices
        global_indices = torch.randperm(self.dataset_size, generator=rng)

        if drop_first_row:
            # drop 0 in global_indices
            global_indices = global_indices[global_indices != 0]
            self.dataset_size = self.dataset_size - 1

        if self.drop_last:
            # For drop_last=True, we:
            # 1. Ensure total samples is divisible by (batch_size * num_sp_groups)
            # 2. This guarantees each SP group gets same number of complete batches
            # 3. Prevents uneven batch sizes across SP groups at end of epoch
            num_batches = self.dataset_size // self.batch_size
            num_global_batches = num_batches // self.num_sp_groups
            global_indices = global_indices[
                : num_global_batches * self.num_sp_groups * self.batch_size
            ]
        else:
            if self.dataset_size % (self.num_sp_groups * self.batch_size) != 0:
                # add more indices to make it divisible by (batch_size * num_sp_groups)
                padding_size = self.num_sp_groups * self.batch_size - (
                    self.dataset_size % (self.num_sp_groups * self.batch_size)
                )
                logger.info(
                    "Padding the dataset from %d to %d",
                    self.dataset_size,
                    self.dataset_size + padding_size,
                )
                global_indices = torch.cat(
                    [global_indices, global_indices[:padding_size]]
                )

        # shard the indices to each sp group
        ith_sp_group = self.global_rank // self.sp_world_size
        sp_group_local_indices = global_indices[
            ith_sp_group :: self.num_sp_groups
        ]
        self.sp_group_local_indices = sp_group_local_indices
        logger.info(
            "Dataset size for each sp group: %d", len(sp_group_local_indices)
        )

    def __iter__(self):
        indices = self.sp_group_local_indices
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            yield batch_indices.tolist()

    def __len__(self):
        return len(self.sp_group_local_indices) // self.batch_size


def angle_bucket(theta):
    """把角度映射到 [-1, 0, +1] 三类."""
    out = torch.zeros_like(theta, dtype=torch.int)
    out[(theta >= 0) & (theta < 30)] = 1  # 正
    out[(theta >= 150) & (theta <= 180)] = -1  # 负
    # 30–60 或 120–150 → 斜，标记为 2 或 -2
    out[(theta >= 30) & (theta < 60)] = 2  # 正斜
    out[(theta >= 120) & (theta < 150)] = -2  # 负斜
    return out


def angle_to_onehot(x):
    """
    x: (B,2)，分别是 (θx, θy)，范围 [0,180]
    return: (B,4) one-hot 向量 [右, 左, 后, 前]
    """
    B = x.size(0)
    one_hot = torch.zeros(B, 4, dtype=torch.int)

    bucket_x = angle_bucket(x[:, 0])
    bucket_z = angle_bucket(x[:, 1])

    # θx 控制前/后
    one_hot[bucket_x == 1, 3] = 1  # 前
    one_hot[bucket_x == -1, 2] = 1  # 后
    one_hot[bucket_x == 2, 3] = 1  # 前斜 → 前
    one_hot[bucket_x == -2, 2] = 1  # 后斜 → 后

    # θy 控制左/右
    one_hot[bucket_z == 1, 0] = 1  # 右
    one_hot[bucket_z == -1, 1] = 1  # 左
    one_hot[bucket_z == 2, 0] = 1  # 右斜 → 右
    one_hot[bucket_z == -2, 1] = 1  # 左斜 → 左

    return one_hot


class HunyuanImageJsonDataset(Dataset):
    def __init__(
        self,
        json_path,
        causal,
        window_frames,
        batch_size,
        cfg_rate,
        i2v_rate,
        drop_last,
        drop_first_row,
        seed,
        random_pose_path="",
        neg_prompt_path="",
        neg_byt5_prompt_path="",
    ):
        self.json_data = json.load(open(json_path, "r"))
        self.all_length = len(self.json_data)
        self.causal = causal
        self.window_frames = window_frames
        self.cfg_rate = cfg_rate
        self.rng = random.Random(seed)
        self.i2v_rate = i2v_rate

        # Load random pose from config path
        if not random_pose_path:
            raise ValueError("random_pose_path must be provided")
        self.random_pose = json.load(open(random_pose_path, "r"))

        self.sampler = DP_SP_BatchSampler(
            batch_size=batch_size,
            dataset_size=self.all_length,
            num_sp_groups=get_world_size() // get_gpu_world_size(),
            sp_world_size=get_gpu_world_size(),
            global_rank=get_world_rank(),
            drop_last=drop_last,
            drop_first_row=drop_first_row,
            seed=seed,
        )
        self.mapping = {
            (0, 0, 0, 0): 0,
            (1, 0, 0, 0): 1,
            (0, 1, 0, 0): 2,
            (0, 0, 1, 0): 3,
            (0, 0, 0, 1): 4,
            (1, 0, 1, 0): 5,
            (1, 0, 0, 1): 6,
            (0, 1, 1, 0): 7,
            (0, 1, 0, 1): 8,
        }

        # Load negative prompt from config path
        # if not neg_prompt_path:
        #     raise ValueError("neg_prompt_path must be provided")
        # self.neg_prompt_pt = torch.load(
        #     neg_prompt_path,
        #     map_location="cpu",
        #     weights_only=True,
        # )

        # # Load negative byt5 prompt from config path
        # if not neg_byt5_prompt_path:
        #     raise ValueError("neg_byt5_prompt_path must be provided")
        # self.neg_byt5_pt = torch.load(
        #     neg_byt5_prompt_path,
        #     map_location="cpu",
        #     weights_only=True,
        # )

    def __len__(self):
        return self.all_length

    def camera_center_normalization(self, w2c):
        c2w = np.linalg.inv(w2c)
        C0_inv = np.linalg.inv(c2w[0])
        c2w_aligned = np.array([C0_inv @ C for C in c2w])
        return np.linalg.inv(c2w_aligned)

    def one_hot_to_one_dimension(self, one_hot):
        y = torch.tensor([self.mapping[tuple(row.tolist())] for row in one_hot])
        return y

    def __getitem__(self, idx):
        while True:
            try:
                json_data = self.json_data[idx]

                latent_pt_path = json_data["latent_path"]

                latent_pt = torch.load(
                    os.path.join(latent_pt_path),
                    map_location="cpu",
                    weights_only=True,
                )

                # Extract latent from the new structure
                latent = latent_pt["latent"]
                if latent.ndim == 5:  # [1, C, T, H, W]
                    latent = latent[0]  # Remove batch dimension -> [C, T, H, W]
                latent_length = latent.shape[1]

                # 将latent的第三维zero padding到self.window_frames
                if latent.shape[1] < self.window_frames:
                    pad_size = self.window_frames - latent.shape[1]
                    pad_shape = list(latent.shape)
                    pad_shape[1] = pad_size
                    latent = torch.cat(
                        [latent, torch.zeros(pad_shape, dtype=latent.dtype)],
                        dim=1,
                    )
                elif latent.shape[1] > self.window_frames:
                    latent = latent[:, : self.window_frames, ...]

                # Get caption from json_data directly
                prompt = json_data.get("caption", "")
                if not prompt:
                    raise ValueError("caption is None or empty")

                # Extract text embeddings from the new structure
                prompt_embed = latent_pt["prompt_embeds"]
                if prompt_embed.ndim == 3:  # [1, L, D]
                    prompt_embed = prompt_embed[
                        0
                    ]  # Remove batch dimension -> [L, D]

                prompt_mask = latent_pt["prompt_mask"]
                if prompt_mask.ndim == 2:  # [1, L]
                    prompt_mask = prompt_mask[
                        0
                    ]  # Remove batch dimension -> [L]

                # Extract image condition from the new structure
                image_cond = latent_pt["image_cond"]
                if image_cond.ndim == 5:  # [1, C, 1, H, W]
                    image_cond = image_cond[
                        0
                    ]  # Remove batch dimension -> [C, 1, H, W]

                # Extract vision states
                vision_states = latent_pt["vision_states"]
                if vision_states.ndim == 3:  # [1, N, D]
                    vision_states = vision_states[
                        0
                    ]  # Remove batch dimension -> [N, D]

                # Extract byT5 embeddings
                byt5_text_states = latent_pt["byt5_text_states"]
                if byt5_text_states.ndim == 3:  # [1, 256, 1472]
                    byt5_text_states = byt5_text_states[
                        0
                    ]  # Remove batch dimension -> [256, 1472]

                byt5_text_mask = latent_pt["byt5_text_mask"]
                if byt5_text_mask.ndim == 2:  # [1, 256]
                    byt5_text_mask = byt5_text_mask[
                        0
                    ]  # Remove batch dimension -> [256]

                pose_json = self.random_pose[idx % len(self.random_pose)]
                pose_keys = list(pose_json.keys())

                intrinsic_list = []
                w2c_list = []
                for i in range(self.window_frames):
                    t_key = pose_keys[i]

                    c2w = np.array(pose_json[t_key]["extrinsic"])
                    w2c = np.linalg.inv(c2w)

                    w2c_list.append(w2c)
                    intrinsic = np.array(pose_json[t_key]["K"])
                    if (intrinsic > 2000.0).any():
                        raise ValueError("intrinsic > 2000")
                    intrinsic[0, 0] /= intrinsic[0, 2] * 2
                    intrinsic[1, 1] /= intrinsic[1, 2] * 2
                    intrinsic[0, 2] = 0.5
                    intrinsic[1, 2] = 0.5
                    intrinsic_list.append(intrinsic)

                w2c_list = np.array(w2c_list)
                # w2c_list = self.camera_center_normalization(w2c_list)
                intrinsic_list = torch.tensor(np.array(intrinsic_list))

                # 从这里开始是计算离散action
                c2ws = np.linalg.inv(w2c_list)
                C_inv = np.linalg.inv(c2ws[:-1])
                relative_c2w = np.zeros_like(c2ws)
                relative_c2w[0, ...] = c2ws[0, ...]
                relative_c2w[1:, ...] = C_inv @ c2ws[1:, ...]
                trans_one_hot = np.zeros(
                    (relative_c2w.shape[0], 4), dtype=np.int32
                )
                rotate_one_hot = np.zeros(
                    (relative_c2w.shape[0], 4), dtype=np.int32
                )

                move_norm_valid = 0.0001
                for i in range(1, relative_c2w.shape[0]):
                    move_dirs = relative_c2w[i, :3, 3]  # 方向向量
                    move_norms = np.linalg.norm(move_dirs)
                    # print(move_norms)
                    if move_norms > move_norm_valid:  # 认为有移动
                        move_norm_dirs = move_dirs / move_norms
                        # print(np.linalg.norm(move_norm_dirs))
                        angles_rad = np.arccos(
                            move_norm_dirs.clip(-1.0, 1.0)
                        )  # 防止数值误差超出[-1,1]
                        trans_angles_deg = angles_rad * (
                            180.0 / torch.pi
                        )  # 转成角度 0-180，这里是translation的角度
                    else:
                        trans_angles_deg = torch.zeros(3)

                    R_rel = relative_c2w[i, :3, :3]
                    r = R.from_matrix(R_rel)
                    rot_angles_deg = r.as_euler(
                        "xyz", degrees=True
                    )  # 输出欧拉角，单位度

                    # 现在需要解决的就是scale的问题，norms以及多少度算是转动
                    if move_norms > move_norm_valid:  # 认为有移动
                        # 先判断前进后退，这个就根据z轴的来就行
                        if trans_angles_deg[2] < 60:
                            trans_one_hot[i, 0] = 1  # 前
                        elif trans_angles_deg[2] > 120:
                            trans_one_hot[i, 1] = 1  # 后

                        if trans_angles_deg[0] < 60:
                            trans_one_hot[i, 2] = 1  # 右
                        elif trans_angles_deg[0] > 120:
                            trans_one_hot[i, 3] = 1  # 左

                    # 接下来是rotate, 对于matrix来说角度为9比较合适，对于其他数据来说需要找一个合适的角度
                    if rot_angles_deg[1] > 5e-2:
                        rotate_one_hot[i, 0] = 1  # 右
                    elif rot_angles_deg[1] < -5e-2:
                        rotate_one_hot[i, 1] = 1  # 左

                    if rot_angles_deg[0] > 5e-2:
                        rotate_one_hot[i, 2] = 1  # 上
                    elif rot_angles_deg[0] < -5e-2:
                        rotate_one_hot[i, 3] = 1  # 下

                trans_one_hot = torch.tensor(trans_one_hot)
                rotate_one_hot = torch.tensor(rotate_one_hot)
                action = torch.cat([trans_one_hot, rotate_one_hot], dim=1)

                trans_one_label = self.one_hot_to_one_dimension(trans_one_hot)
                rotate_one_label = self.one_hot_to_one_dimension(rotate_one_hot)
                action_for_pe = trans_one_label * 9 + rotate_one_label

                i2v_mask = torch.ones_like(latent)

                batch = {
                    "i2v_mask": i2v_mask,
                    "latent": latent,
                    "prompt_embed": prompt_embed,
                    "w2c": torch.tensor(w2c_list),
                    "intrinsic": intrinsic_list,
                    "action": action_for_pe,
                    "action_for_pe": action_for_pe,
                    "context_frames_list": None,  # selected context frames for each chunk
                    "prompt": prompt,
                    "image_cond": image_cond,
                    "vision_states": vision_states,
                    "prompt_mask": prompt_mask,
                    "byt5_text_states": byt5_text_states,
                    "byt5_text_mask": byt5_text_mask,
                    # "all_channel_latents": all_channel_latents,
                }
                break
            except Exception as e:
                print("error:", e, latent_pt_path)
                idx = self.rng.randint(0, self.all_length - 1)

        return batch


def hunyuan_latent_collate_function(batch):
    latent = torch.stack([b["latent"] for b in batch], dim=0)
    prompt_embed = torch.stack([b["prompt_embed"] for b in batch], dim=0)
    w2c = torch.stack([b["w2c"] for b in batch], dim=0)
    intrinsic = torch.stack([b["intrinsic"] for b in batch], dim=0)
    action = torch.stack([b["action"] for b in batch], dim=0)
    action_for_pe = torch.stack([b["action_for_pe"] for b in batch], dim=0)
    i2v_mask = torch.stack([b["i2v_mask"] for b in batch], dim=0)

    image_cond = torch.stack([b["image_cond"] for b in batch], dim=0)
    vision_states = torch.stack([b["vision_states"] for b in batch], dim=0)
    prompt_mask = torch.stack([b["prompt_mask"] for b in batch], dim=0)
    byt5_text_states = torch.stack(
        [b["byt5_text_states"] for b in batch], dim=0
    )
    byt5_text_mask = torch.stack([b["byt5_text_mask"] for b in batch], dim=0)

    context_frames_list = [b["context_frames_list"] for b in batch]
    prompt = [b["prompt"] for b in batch]

    return {
        "i2v_mask": i2v_mask,
        "latent": latent,
        "prompt_embed": prompt_embed,
        "w2c": w2c,
        "intrinsic": intrinsic,
        "action": action,
        "context_frames_list": context_frames_list,
        "action_for_pe": action_for_pe,
        "image_cond": image_cond,
        "vision_states": vision_states,
        "prompt_mask": prompt_mask,
        "byt5_text_states": byt5_text_states,
        "byt5_text_mask": byt5_text_mask,
        "prompt": prompt,
    }


def cycle(dl):
    while True:
        for data in dl:
            yield data


def latent_collate_function(batch):
    # dir_norm = torch.stack([b["dir_norm"] for b in batch], dim=0)
    # trans_discrete_labels = torch.stack([b["trans_discrete_labels"] for b in batch], dim=0)
    # norm_dirs = torch.stack([b["norm_dirs"] for b in batch], dim=0)
    latent = torch.stack([b["latent"] for b in batch], dim=0)
    prompt_embed = torch.stack([b["prompt_embed"] for b in batch], dim=0)
    w2c = torch.stack([b["w2c"] for b in batch], dim=0)
    intrinsic = torch.stack([b["intrinsic"] for b in batch], dim=0)
    action = torch.stack([b["action"] for b in batch], dim=0)
    action_for_pe = torch.stack([b["action_for_pe"] for b in batch], dim=0)
    # rotate_degree = torch.stack([b["rotate_degree"] for b in batch], dim=0)
    i2v_mask = torch.stack([b["i2v_mask"] for b in batch], dim=0)
    # video_path = [b["video_path"] for b in batch]
    prompt = [b["prompt"] for b in batch]
    return {
        "i2v_mask": i2v_mask,
        # "dir_norm": dir_norm,
        # "rotate_degree": rotate_degree,
        # "trans_discrete_labels": trans_discrete_labels,
        # "norm_dirs": norm_dirs,
        "latent": latent,
        "prompt_embed": prompt_embed,
        "prompt": prompt,
        "w2c": w2c,
        "intrinsic": intrinsic,
        "action": action,
        "action_for_pe": action_for_pe,
    }


def build_hy_camera_dataloader(
    json_path,
    causal,
    window_frames,
    batch_size,
    num_data_workers,
    drop_last,
    drop_first_row,
    seed,
    cfg_rate,
    i2v_rate,
    random_pose_path="",
    neg_prompt_path="",
    neg_byt5_prompt_path="",
) -> tuple[HunyuanImageJsonDataset, StatefulDataLoader]:
    dataset = HunyuanImageJsonDataset(
        json_path,
        causal,
        window_frames,
        batch_size,
        cfg_rate,
        i2v_rate,
        drop_last=drop_last,
        drop_first_row=drop_first_row,
        seed=seed,
        random_pose_path=random_pose_path,
        neg_prompt_path=neg_prompt_path,
        neg_byt5_prompt_path=neg_byt5_prompt_path,
    )

    loader = StatefulDataLoader(
        dataset,
        batch_sampler=dataset.sampler,
        collate_fn=hunyuan_latent_collate_function,
        num_workers=num_data_workers,
        pin_memory=True,
        persistent_workers=num_data_workers > 0,
    )
    return dataset, loader
