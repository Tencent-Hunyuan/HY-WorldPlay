# SPDX-License-Identifier: Apache-2.0
# Adapted from: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/utils.py
"""Utils for model executor."""

from typing import Any

import torch


# TODO(PY): move it elsewhere
def auto_attributes(init_func):
    """Decorator that automatically adds all initialization arguments as object attributes.

    Example:
        @auto_attributes
        def __init__(self, a=1, b=2):
            pass

        # This will automatically set:
        # - self.a = 1 and self.b = 2
        # - self.config.a = 1 and self.config.b = 2
    """

    def wrapper(self, *args, **kwargs):
        # Get the function signature
        import inspect

        signature = inspect.signature(init_func)
        parameters = signature.parameters

        # Get parameter names (excluding 'self')
        param_names = list(parameters.keys())[1:]

        # Bind arguments to parameters
        bound_args = signature.bind(self, *args, **kwargs)
        bound_args.apply_defaults()

        # Create config object if it doesn't exist
        if not hasattr(self, "config"):
            self.config = type("Config", (), {})()

        # Set attributes on self and self.config
        for name in param_names:
            if name in bound_args.arguments:
                value = bound_args.arguments[name]
                setattr(self, name, value)
                setattr(self.config, name, value)

        # Call the original __init__ function
        return init_func(self, *args, **kwargs)

    return wrapper


def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: dict[str, Any] | None,
):
    """Set attributes on a weight tensor.

    This method is used to set attributes on a weight tensor. This method
    will not overwrite existing attributes.

    Args:
        weight: The weight tensor.
        weight_attrs: A dictionary of attributes to set on the weight tensor.
    """
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        assert not hasattr(
            weight, key
        ), f"Overwriting existing tensor attribute: {key}"

        # NOTE(woosuk): During weight loading, we often do something like:
        # narrowed_tensor = param.data.narrow(0, offset, len)
        # narrowed_tensor.copy_(real_weight)
        # expecting narrowed_tensor and param.data to share the same storage.
        # However, on TPUs, narrowed_tensor will lazily propagate to the base
        # tensor, which is param.data, leading to the redundant memory usage.
        # This sometimes causes OOM errors during model loading. To avoid this,
        # we sync the param tensor after its weight loader is called.
        # TODO(woosuk): Remove this hack once we have a better solution.
        from fastvideo.platforms import current_platform

        if current_platform.is_tpu() and key == "weight_loader":
            value = _make_synced_weight_loader(value)
        setattr(weight, key, value)


def _make_synced_weight_loader(original_weight_loader) -> Any:

    def _synced_weight_loader(param, *args, **kwargs):
        original_weight_loader(param, *args, **kwargs)
        torch._sync(param)

    return _synced_weight_loader


def extract_layer_index(layer_name: str) -> int:
    """Extract the layer index from the module name.

    Examples:
    - "encoder.layers.0" -> 0
    - "encoder.layers.1.self_attn" -> 1
    - "2.self_attn" -> 2
    - "model.encoder.layers.0.sub.1" -> ValueError
    """
    subnames = layer_name.split(".")
    int_vals: list[int] = []
    for subname in subnames:
        try:
            int_vals.append(int(subname))
        except ValueError:
            continue
    assert (
        len(int_vals) == 1
    ), f"layer name {layer_name} should only contain one integer"
    return int_vals[0]


def modulate(
    x: torch.Tensor,
    shift: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Modulate by shift and scale.

    Args:
        x (torch.Tensor): input tensor.
        shift (torch.Tensor, optional): shift tensor. Defaults to None.
        scale (torch.Tensor, optional): scale tensor. Defaults to None.

    Returns:
        torch.Tensor: the output tensor after modulate.
    """
    if scale is None and shift is None:
        return x
    elif shift is None:
        return x * (1 + scale.unsqueeze(1))  # type: ignore[union-attr]
    elif scale is None:
        return x + shift.unsqueeze(1)  # type: ignore[union-attr]
    else:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)  # type: ignore[union-attr]


import json
import os
import sys

sys.path.append(os.path.abspath("."))
import torch
import pandas as pd
import numpy as np
import random
from pathlib import Path
from typing import List, Tuple, Dict
from scipy.spatial.transform import Rotation as R_scipy
import math

from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from scipy.spatial.transform import Rotation as R
from fastvideo.distributed import (
    get_local_torch_device,
    maybe_init_distributed_environment_and_model_parallel,
)
from fastvideo.distributed import (
    get_sp_world_size,
    get_world_group,
    get_world_rank,
    get_world_size,
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


# *******************************Add the memory part********************************
def calculate_pose_distance_from_w2c(
    w2c_1: np.ndarray,
    w2c_2: np.ndarray,
    pos_weight: float = 1.0,
    ang_weight: float = 1.0,
) -> float:
    """根据两个 4x4 W2C (World-to-Camera) 矩阵计算它们之间的综合姿态距离。

    该距离量化了两个相机姿态的相似度，类似其 FOV 的重叠程度。

    参数:
        w2c_1 (np.ndarray): 第一个相机的 4x4 World-to-Camera 矩阵。
        w2c_2 (np.ndarray): 第二个相机的 4x4 World-to-Camera 矩阵。
        pos_weight (float): 空间距离的权重。
        ang_weight (float): 角度距离的权重。

    返回:
        float: 两个姿态之间的综合距离。
    """

    def w2c_to_6d_pose(w2c_matrix: np.ndarray) -> np.ndarray:
        """将 4x4 World-to-Camera (W2C) 矩阵转换为 6D 姿态。

        6D 姿态元组为 (x, y, z, pitch, yaw, roll)。
        """
        # 提取旋转矩阵 R 和平移向量 t
        R_cw = w2c_matrix[:3, :3]
        t_cw = w2c_matrix[:3, 3]

        # 计算相机在世界坐标系下的位置 C_world
        # C_world = -R_cw.T @ t_cw
        C_world = -np.dot(R_cw.T, t_cw)

        # 将旋转矩阵转换为欧拉角 (pitch, yaw, roll)
        # 注意: scipy 默认的欧拉角顺序是 ZYX，对应 yaw, pitch, roll
        # 为了与常见的 (pitch, yaw, roll) 顺序匹配，我们手动转换
        r = R_scipy.from_matrix(R_cw)
        pitch, yaw, roll = r.as_euler("yxz", degrees=True)

        return np.array([C_world[0], C_world[1], C_world[2], pitch, yaw, roll])

    # 1. 将两个 W2C 矩阵转换为 6D 姿态
    pose1_6d = w2c_to_6d_pose(w2c_1)
    pose2_6d = w2c_to_6d_pose(w2c_2)

    # 2. 计算空间距离 (欧几里得距离)
    pos1 = pose1_6d[:3]
    pos2 = pose2_6d[:3]
    spatial_distance = np.linalg.norm(pos1 - pos2)

    # 3. 计算角度距离 (考虑圆周特性)
    angles1 = pose1_6d[3:]
    angles2 = pose2_6d[3:]

    angle_diff = np.abs(angles1 - angles2)
    # 修正角度差，确保是最小的圆周距离
    angular_distance_vector = np.minimum(angle_diff, 360 - angle_diff)
    # 使用欧几里得范数作为综合角度距离
    angular_distance = np.linalg.norm(angular_distance_vector)

    # 4. 结合两种距离得到综合姿态距离
    total_distance = (
        pos_weight * spatial_distance + ang_weight * angular_distance
    )

    return total_distance


# TO DO: add a chunk mechanism to chunk the long sequence to the window size
# --- 新的辅助函数：计算复杂片段距离 ---
def calculate_complex_clip_distance(
    w2c_list: List[np.ndarray],
    query_clip_indices: List[int],
    historical_clip_indices: List[int],
    pos_weight: float = 1.0,
    ang_weight: float = 1.0,
) -> float:
    """计算查询片段与历史片段之间的复杂姿态距离。

    该距离是基于查询片段的第二帧和第四帧与历史片段的每一帧的平均距离。
    """
    # 1. 确定查询片段的采样帧索引
    # 从 query_clip_indices 的第二个元素 (索引 1) 开始，每隔两帧选取
    # 例如：如果 query_clip_indices = [10, 11, 12, 13, 14, 15]
    #      采样索引为：11, 13, 15

    # 确保查询片段至少有 2 帧才能进行采样
    if len(query_clip_indices) < 2:
        # 如果预测序列太短，使用第一个和最后一个帧作为样本
        sample_indices = [query_clip_indices[0], query_clip_indices[-1]]
    else:
        # 采样点在 query_clip_indices 中的局部索引：从 1 开始，步长为 2
        # np.arange(1, len(query_clip_indices), 2)
        sample_indices = [
            query_clip_indices[i]
            for i in np.arange(1, len(query_clip_indices), 2)
        ]

    total_avg_distance = 0.0

    # 2. 遍历所有采样帧
    for query_idx in sample_indices:
        query_w2c = w2c_list[query_idx]

        dists_from_query_frame = []

        # 3. 计算该采样帧与历史片段中每一帧的距离
        for hist_idx in historical_clip_indices:
            hist_w2c = w2c_list[hist_idx]
            dist = calculate_pose_distance_from_w2c(
                query_w2c, hist_w2c, pos_weight, ang_weight
            )
            dists_from_query_frame.append(dist)

        # 累加这个采样帧到历史片段的平均距离
        total_avg_distance += np.mean(dists_from_query_frame)

    # 4. 最终距离是所有采样帧平均距离的再次平均
    final_clip_distance = total_avg_distance / len(sample_indices)

    return final_clip_distance


# --- 工具函数 1: 角度转换 (从旋转矩阵到 Pitch/Yaw) ---
def rotation_matrix_to_angles(
    R: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """从相机坐标系的 3x3 旋转矩阵 R 估算 Pitch 和 Yaw 角度。

    假设相机坐标系: X=右, Y=上, Z=后退 (或 NeRF style: X=右, Y=下, Z=前)
    这里我们使用常见的计算机视觉约定：Z轴为向前。

    注意：这里的角度计算直接基于您的 is_inside_fov_3d_hv 函数的约定：
    - Yaw/Azimuth 角度在 XZ 平面 (atan2(x, z))。
    - Pitch/Elevation 角度相对于水平面 (atan2(y, sqrt(x^2 + z^2)))。

    对于 W2C 矩阵 R 的第三列 R[:, 2] (世界坐标系中Z轴在相机坐标系下的方向)，
    通常对应于相机看向的方向（z轴在世界坐标系中的表示）。

    为了简化并匹配您的 is_inside_fov 逻辑，我们直接使用 R 的 Z 轴向量：
    Camera Z-axis direction in World Frame (Forward Vector): fwd = R_w2c_inv @ [0, 0, 1]
    更简单地，C2W 矩阵的 Z 轴向量即为相机在世界坐标系下的前向向量。
    C2W = W2C_inv
    """

    # 1. 估算前向向量（在世界坐标系中）
    # R 是 R_c2w.T，我们想要 R_c2w 的 Z 轴。
    # 对于 R_w2c，它的逆 R_c2w 是 R_w2c.T
    R_c2w = R.T

    # 前向向量是 C2W 矩阵的第三列（Z轴方向）
    fwd = R_c2w[:, 2]  # 形状 (3,)

    x = fwd[0]
    y = fwd[1]
    z = fwd[2]

    # 2. 计算 Yaw (偏航角)
    # Yaw = atan2(x, z) - 假设 Z 是前向，X 是左右
    yaw_rad = torch.atan2(x, z)
    yaw_deg = yaw_rad * (180.0 / math.pi)

    # 3. 计算 Pitch (俯仰角)
    # Pitch = atan2(y, sqrt(x^2 + z^2))
    pitch_rad = torch.atan2(y, torch.sqrt(x**2 + z**2))
    pitch_deg = pitch_rad * (180.0 / math.pi)

    return pitch_deg, yaw_deg


def generate_points_in_sphere(n_points: int, radius: float) -> torch.Tensor:
    """在指定半径的球体内均匀生成点。

    :param n_points: 要生成的点数。
    :param radius: 球体半径。
    :return: 形状为 (n_points, 3) 的张量，表示点的 (x, y, z) 坐标。
    """
    samples_r = torch.rand(n_points)
    samples_phi = torch.rand(n_points)
    samples_u = torch.rand(n_points)

    # 均匀体积采样：r = R * u^(1/3)
    r = radius * torch.pow(samples_r, 1 / 3)
    phi = 2 * math.pi * samples_phi
    # 均匀极角采样：theta = arccos(1 - 2*u)
    theta = torch.acos(1 - 2 * samples_u)

    # 转换为笛卡尔坐标
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)

    points = torch.stack((x, y, z), dim=1)
    return points


def is_inside_fov_3d_hv(
    points: torch.Tensor,
    center: torch.Tensor,
    center_pitch: torch.Tensor,
    center_yaw: torch.Tensor,
    fov_half_h: torch.Tensor,
    fov_half_v: torch.Tensor,
) -> torch.Tensor:
    """检查点是否在由中心坐标、俯仰角和偏航角定义的 3D 视锥体内。

    :param points: (N, 3) 或 (N, B, 3) 采样点坐标。
    :param center: (3) 或 (B, 3) 相机中心坐标。
    :param center_pitch: (1) 或 (B) 中心视线的俯仰角（度）。
    :param center_yaw: (1) 或 (B) 中心视线的偏航角（度）。
    :param fov_half_h: 水平半视场角（度）。
    :param fov_half_v: 垂直半视场角（度）。
    :return: (N) 或 (N, B) 布尔张量，指示点是否在 FOV 内。
    """
    # 确保 center, pitch, yaw 维度可广播
    if points.ndim == 2:  # N, 3
        vectors = points - center[None, :]
        C = 1  # 批次大小为1
    elif points.ndim == 3:  # N, B, 3
        vectors = points - center[None, ...]
        # 确保 center_pitch/yaw 是 (1, B) 形状进行广播
        center_pitch = (
            center_pitch[None, :] if center_pitch.ndim == 1 else center_pitch
        )
        center_yaw = center_yaw[None, :] if center_yaw.ndim == 1 else center_yaw
    else:
        raise ValueError("points 必须是 (N, 3) 或 (N, B, 3) 形状")

    x = vectors[..., 0]
    y = vectors[..., 1]
    z = vectors[..., 2]

    # 计算水平角 (偏航/方位角)，假设 Z 轴为前向
    azimuth = torch.atan2(x, z) * (180 / math.pi)

    # 计算垂直角 (俯仰/仰角)
    elevation = torch.atan2(y, torch.sqrt(x**2 + z**2)) * (180 / math.pi)

    # 计算与中心视线的角度差 (处理角度环绕)
    diff_azimuth = azimuth - center_yaw
    # 将角度差归一化到 [-180, 180] 范围
    diff_azimuth = torch.remainder(diff_azimuth + 180, 360) - 180

    diff_elevation = elevation - center_pitch
    diff_elevation = torch.remainder(diff_elevation + 180, 360) - 180

    # 检查是否在 FOV 限制内
    in_fov_h = diff_azimuth.abs() < fov_half_h
    in_fov_v = diff_elevation.abs() < fov_half_v

    return in_fov_h & in_fov_v


# --- 工具函数 2: 欧氏距离 (基于 W2C 的位置) ---
def calculate_euclidean_distance(
    w2c_matrix_1: torch.Tensor, w2c_matrix_2: torch.Tensor
) -> torch.Tensor:
    """计算两个相机中心位置之间的欧氏距离。"""
    # 相机在世界坐标系中的位置 P_w = -R^T * t
    R1 = w2c_matrix_1[:3, :3]
    t1 = w2c_matrix_1[:3, 3]
    R2 = w2c_matrix_2[:3, :3]
    t2 = w2c_matrix_2[:3, 3]

    P_w1 = -R1.T @ t1
    P_w2 = -R2.T @ t2

    distance = torch.linalg.norm(P_w1 - P_w2)
    return distance


def calculate_fov_overlap_similarity_refine(
    w2c_matrix_curr: torch.Tensor,
    w2c_matrix_hist: torch.Tensor,
    fov_h_deg: float = 105.0,
    fov_v_deg: float = 75.0,
    device=None,
    points_local=None,
) -> float:
    """使用蒙特卡洛采样计算两个 W2C 位姿之间的 FOV 重叠相似性。

    相似性 = (Curr_FOV ∩ Hist_FOV) 的点数 / (Curr_FOV) 的点数。

    :param w2c_matrix_curr: 当前帧的 (4, 4) W2C 矩阵。
    :param w2c_matrix_hist: 历史帧的 (4, 4) W2C 矩阵。 :param num_samples, radius, fov_h_deg, fov_v_deg:
        采样和 FOV 参数。
    :return: 重叠比率 (0.0 到 1.0 之间的浮点数)。
    """
    w2c_matrix_curr = torch.tensor(w2c_matrix_curr, device=device)
    w2c_matrix_hist = torch.tensor(w2c_matrix_hist, device=device)

    # 转换为相对pose
    c2w_matrix_curr = torch.linalg.inv(w2c_matrix_curr)
    c2w_matrix_hist = torch.linalg.inv(w2c_matrix_hist)
    C_inv = w2c_matrix_curr

    w2c_matrix_curr = torch.linalg.inv(C_inv @ c2w_matrix_curr)
    w2c_matrix_hist = torch.linalg.inv(C_inv @ c2w_matrix_hist)

    # --- 1. W2C 矩阵解析为位置和角度 ---

    # P_w = -R^T * t
    R_curr, t_curr = w2c_matrix_curr[:3, :3], w2c_matrix_curr[:3, 3]
    R_hist, t_hist = w2c_matrix_hist[:3, :3], w2c_matrix_hist[:3, 3]

    P_w_curr = -R_curr.T @ t_curr
    P_w_hist = -R_hist.T @ t_hist

    # pitch, yaw 角度 (度)
    pitch_curr, yaw_curr = rotation_matrix_to_angles(R_curr)
    pitch_hist, yaw_hist = rotation_matrix_to_angles(R_hist)

    # FOV 参数
    fov_half_h = torch.tensor(fov_h_deg / 2.0, device=device)
    fov_half_v = torch.tensor(fov_v_deg / 2.0, device=device)

    # 将点平移到当前相机中心 P_w_curr (N, 3)
    # 对应您代码中的 points += pose_conditions[curr_frame, :, :3][None] 的简化版本 (B=1)
    points_world = points_local + P_w_curr[None, :]

    # --- 3. FOV 检查 ---

    # 检查点是否在 Curr FOV 内 (作为分母的基准)
    in_fov_curr = is_inside_fov_3d_hv(
        points_world,
        P_w_curr[None, :],
        pitch_curr[None],
        yaw_curr[None],
        fov_half_h,
        fov_half_v,
    )

    # 检查点是否在 Hist FOV 内, 只根据角度进行计算
    in_fov_hist = is_inside_fov_3d_hv(
        points_world,
        P_w_hist[None, :],
        pitch_hist[None],
        yaw_hist[None],
        fov_half_h,
        fov_half_v,
    )

    # 根据点到点的距离在计算一下
    dist = torch.norm(points_world - P_w_hist.reshape(1, -1), dim=1) < 8.0
    in_fov_hist = in_fov_hist.bool() & dist.reshape(1, -1).bool()
    # --- 4. 计算重叠比率 ---

    # 分子: 交集中的点数
    overlap_count = (in_fov_curr.bool() & in_fov_hist.bool()).sum().float()

    # 分母: Curr FOV 中的总点数
    fov_curr_count = in_fov_curr.sum().float()
    # print(overlap_count, fov_curr_count)

    if fov_curr_count == 0:
        return 0.0  # 避免除以零

    # 重叠比率
    overlap_ratio = overlap_count / fov_curr_count

    return overlap_ratio.item()


def select_aligned_memory_frames_context_per_chunk_w_latent_sink_fov_refine_hunyuan(
    w2c_list: List[np.ndarray],
    current_frame_idx: int,
    memory_frames: int,
    temporal_context_size: int,
    pred_latent_size: int,
    pos_weight: float = 1.0,
    ang_weight: float = 1.0,
    device=None,
    points_local=None,
) -> List[int]:
    """为给定帧选择记忆帧和上下文帧，基于复杂的四帧片段距离计算。

    参数:     w2c_list (List[np.ndarray]): 包含所有N个4x4外参矩阵的列表。     current_frame_idx (int): 当前要处理的帧的索引。
    memory_frames (int): 需要选择的记忆帧总数。     context_size (int): 需要选择的上下文帧总数。     pos_weight (float):
    空间距离的权重。     ang_weight (float): 角度距离的权重。

    返回:     List[int]: 包含选定记忆帧和上下文帧索引的列表。
    """
    if current_frame_idx <= memory_frames:
        return list(range(0, current_frame_idx))

    num_total_frames = len(w2c_list)
    # 检查当前帧是否能构成一个完整的4帧片段
    if current_frame_idx >= num_total_frames or current_frame_idx < 3:
        raise ValueError(
            f"当前帧索引必须在 w2c_list 的有效范围内，且至少为3。{current_frame_idx}, {len(w2c_list)}"
        )

    # 1. 选择上下文帧 (Context Frames)
    start_context_idx = max(0, current_frame_idx - temporal_context_size)
    context_frames_indices = list(range(start_context_idx, current_frame_idx))

    # 2. 计算记忆帧 (Memory Frames) 的候选池
    candidate_distances = []
    query_clip_indices = list(
        range(
            current_frame_idx,
            (
                current_frame_idx + pred_latent_size
                if current_frame_idx + pred_latent_size <= num_total_frames
                else num_total_frames
            ),
        )
    )

    historical_clip_indices = list(
        range(4, current_frame_idx - temporal_context_size, 4)
    )

    # 3. 选取最相似的 `memory_frames` 个帧
    memory_frames_indices = [
        0,
        1,
        2,
        3,
    ]  # add the first latent frame as context
    memory_frames = (
        memory_frames - temporal_context_size - len(memory_frames_indices)
    )

    # 遍历所有历史片段，将每个片段作为记忆帧候选
    # 历史片段的起始索引必须是 4 的倍数，且不能与上下文帧重叠
    for hist_idx in historical_clip_indices:
        total_dist = 0
        hist_w2c_1 = w2c_list[hist_idx]
        hist_w2c_2 = w2c_list[hist_idx + 2]
        for query_idx in query_clip_indices:
            dist_1_for_query_idx = (
                1.0
                - calculate_fov_overlap_similarity_refine(
                    w2c_list[query_idx],
                    hist_w2c_1,
                    fov_h_deg=60.0,
                    fov_v_deg=35.0,
                    device=device,
                    points_local=points_local,
                )
            )
            dist_2_for_query_idx = (
                1.0
                - calculate_fov_overlap_similarity_refine(
                    w2c_list[query_idx],
                    hist_w2c_2,
                    fov_h_deg=60.0,
                    fov_v_deg=35.0,
                    device=device,
                    points_local=points_local,
                )
            )
            dist_for_query_idx = (
                dist_1_for_query_idx + dist_2_for_query_idx
            ) / 2.0
            total_dist += dist_for_query_idx
        # 3. 将多个平均值再次取平均，得到最终的片段距离
        final_clip_distance = total_dist / len(query_clip_indices)
        # 存储 (片段起始帧索引, 平均距离)
        candidate_distances.append((hist_idx, final_clip_distance))

        # 按平均距离从小到大排序
    candidate_distances.sort(key=lambda x: x[1])

    # 遍历排序后的候选片段，直到收集到足够的记忆帧
    for start_idx, _ in candidate_distances:
        if start_idx not in memory_frames_indices:
            memory_frames_indices.extend(range(start_idx, start_idx + 4))

        # 检查是否已达到 memory_size 的要求
        if len(memory_frames_indices) >= memory_frames:
            break

    # 4. 组合并去重，以确保没有重复的帧
    selected_frames_set = set(context_frames_indices)
    selected_frames_set.update(memory_frames_indices)

    final_selected_frames = sorted(list(selected_frames_set))
    # assert len(final_selected_frames) == memory_frames + temporal_context_size
    return final_selected_frames


# --- 新函数：处理所有帧 ---
def process_all_frames_for_memory(
    w2c_list: List[np.ndarray],
    window_size: int,
    memory_frames: int,
    pos_weight: float = 1.0,
    ang_weight: float = 1.0,
) -> Dict[int, List[int]]:
    """遍历超过 window_size 的每一帧，为其选择记忆帧和上下文帧。

    参数:
        w2c_list: 包含所有N个4x4外参矩阵的列表。
        window_size: 初始窗口大小，定义了从哪一帧开始处理。
        memory_frames: 需要选择的记忆帧总数。
        pos_weight: 空间距离的权重。
        ang_weight: 角度距离的权重。

    返回:
        Dict[int, List[int]]: 一个字典，键是帧索引，值是对应的记忆帧和上下文帧列表。
    """
    if window_size >= len(w2c_list) or window_size < 4:
        print("警告: 初始窗口大小必须小于总帧数且至少为4。没有帧需要处理。")
        return {}

    pred_latent_size = window_size - memory_frames  # 预测帧数
    step = 4  # 每次移动4帧

    all_selections = {}

    # 如果w2c_list小于等于window_size - memory_frames，则不处理
    if len(w2c_list) <= window_size - memory_frames:
        print(
            "警告: 总帧数小于等于 window_size - memory_frames，没有帧需要作为memory处理。"
        )
        return all_selections

    # 从 window_size 开始遍历到列表的末尾
    for current_frame_idx in range(
        memory_frames, len(w2c_list) - pred_latent_size + step, step
    ):
        # print(f"正在为帧 {current_frame_idx} 选择记忆和上下文...")

        selected_frames = select_memory_frames(
            w2c_list, current_frame_idx, memory_frames, pos_weight, ang_weight
        )
        all_selections[current_frame_idx] = selected_frames

    return all_selections


# *******************************Add the memory part********************************


def get_normalized_dir_diff(x_inv):
    # 提取方向向量
    dirs = x_inv[:, :3, 3]  # shape: (N, 3)

    # 计算相邻差
    diff = torch.zeros_like(dirs)
    diff[1:] = dirs[1:] - dirs[:-1]

    # 对非零向量归一化
    norms = torch.norm(diff, dim=1, keepdim=True)
    norms[0] = 1.0  # 避免第一行除零
    diff_norm = diff / norms

    # 第一个保持 (0,0,0)
    diff_norm[0] = torch.tensor([0.0, 0.0, 0.0], dtype=diff.dtype)

    return norms, diff_norm


def relative_rotations(c2w_mats):
    """
    c2w_mats: (N, 4, 4) 相机c2w矩阵
    返回 (N-1, 3)，每行为相邻帧绕 x,y,z 的相对旋转角度（度数）
    """
    rots = c2w_mats[:, :3, :3]  # 提取旋转部分
    rel_angles = []

    for i in range(len(rots) - 1):
        R_i = rots[i]
        R_j = rots[i + 1]
        R_rel = R_i.T @ R_j  # 相对旋转矩阵

        # --- 将 R_rel 转为欧拉角 (XYZ顺序) ---
        # 注意：这里假设右手坐标系，欧拉角范围 -180~180°
        sy = torch.sqrt(R_rel[0, 0] ** 2 + R_rel[1, 0] ** 2)

        if sy > 1e-6:  # 一般情况
            x = torch.atan2(R_rel[2, 1], R_rel[2, 2])
            y = torch.atan2(-R_rel[2, 0], sy)
            z = torch.atan2(R_rel[1, 0], R_rel[0, 0])
        else:  # 奇异情况（gimbal lock）
            x = torch.atan2(-R_rel[1, 2], R_rel[1, 1])
            y = torch.atan2(-R_rel[2, 0], sy)
            z = 0.0

        angles = torch.rad2deg(torch.stack([x, y, z]))
        rel_angles.append(angles)

    return torch.stack(rel_angles)
