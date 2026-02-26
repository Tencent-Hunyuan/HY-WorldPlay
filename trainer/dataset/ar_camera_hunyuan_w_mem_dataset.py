import json
import os
import sys

sys.path.append(os.path.abspath('.'))
import torch
import pandas as pd
import numpy as np
import random
from pathlib import Path
from typing import List, Tuple, Dict
from scipy.spatial.transform import Rotation as R_scipy
import math
import torch.multiprocessing as mp

from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from scipy.spatial.transform import (Rotation
                                     as R)
from trainer.distributed import (
        get_local_torch_device,
        )
from trainer.distributed import (get_sp_world_size,
                                   get_world_rank, 
                                   get_world_size)

from trainer.logger import init_logger

logger = init_logger(__name__)


def generate_points_in_sphere(n_points: int, radius: float) -> torch.Tensor:
    """
        Uniformly sample points within a sphere of a specified radius.

        :param n_points: The number of points to generate.
        :param radius: The radius of the sphere.
        :return: A tensor of shape (n_points, 3), representing the (x, y, z) coordinates of the points.
    """
    samples_r = torch.rand(n_points)
    samples_phi = torch.rand(n_points)
    samples_u = torch.rand(n_points)

    r = radius * torch.pow(samples_r, 1 / 3)
    phi = 2 * math.pi * samples_phi
    theta = torch.acos(1 - 2 * samples_u)

    # transfer the coordinates from spherical to cartesian
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)

    points = torch.stack((x, y, z), dim=1)
    return points


def rotation_matrix_to_angles(R: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        Estimate the Pitch and Yaw angles from a 3x3 rotation matrix R in the camera coordinate system.

        Assumed Camera Coordinate System: X=Right, Y=Up, Z=Backward 
        (or NeRF style: X=Right, Y=Down, Z=Forward).
        Here we adopt the common Computer Vision convention: Z-axis is Forward.

        Note: The angle calculations here are directly based on the conventions of your `is_inside_fov_3d_hv` function:
        - Yaw/Azimuth angle is in the XZ plane (atan2(x, z)).
        - Pitch/Elevation angle is relative to the horizontal plane (atan2(y, sqrt(x^2 + z^2))).

        For the third column R[:, 2] of the W2C matrix R (the direction of the World Z-axis in the Camera frame),
        this typically corresponds to the direction the camera is looking 
        (the representation of the world Z-axis in the camera frame).

        To simplify and match your `is_inside_fov` logic, we directly use the camera's Z-axis vector:
        Camera Z-axis direction in World Frame (Forward Vector): fwd = R_w2c_inv @ [0, 0, 1]
        More simply, the Z-axis vector of the C2W matrix is the camera's forward vector in the world frame.
        C2W = W2C_inv
    """

    R_c2w = R.T
    fwd = R_c2w[:, 2]  

    x = fwd[0]
    y = fwd[1]
    z = fwd[2]

    # compute yaw and pitch
    yaw_rad = torch.atan2(x, z)
    yaw_deg = yaw_rad * (180.0 / math.pi)
    pitch_rad = torch.atan2(y, torch.sqrt(x ** 2 + z ** 2))
    pitch_deg = pitch_rad * (180.0 / math.pi)

    return pitch_deg, yaw_deg

def is_inside_fov_3d_hv(points: torch.Tensor, center: torch.Tensor,
                        center_pitch: torch.Tensor, center_yaw: torch.Tensor,
                        fov_half_h: torch.Tensor, fov_half_v: torch.Tensor) -> torch.Tensor:
    """
        Check whether points are inside a 3D view frustum defined by a center coordinate, pitch angle, and yaw angle.

        :param points: Tensor of shape (N, 3) or (N, B, 3) representing the coordinates of the sampled points.
        :param center: Tensor of shape (3) or (B, 3) representing the camera center coordinates.
        :param center_pitch: Tensor of shape (1) or (B) representing the pitch angle of center view direction.
        :param center_yaw: Tensor of shape (1) or (B) representing the yaw angle of the center view direction.
        :param fov_half_h: The horizontal half field-of-view angle (in degrees).
        :param fov_half_v: The vertical half field-of-view angle (in degrees).
        :return: Boolean tensor of shape (N) or (N, B), indicating whether each point is inside the FOV.
    """
    if points.ndim == 2:  # N, 3
        vectors = points - center[None, :]
        C = 1  
    elif points.ndim == 3:  # N, B, 3
        vectors = points - center[None, ...]
        center_pitch = center_pitch[None, :] if center_pitch.ndim == 1 else center_pitch
        center_yaw = center_yaw[None, :] if center_yaw.ndim == 1 else center_yaw
    else:
        raise ValueError("points' shape should be (N, 3) or (N, B, 3)")

    x = vectors[..., 0]
    y = vectors[..., 1]
    z = vectors[..., 2]

    # Calculate the horizontal angle (yaw/azimuth), assuming the Z-axis is forward.
    azimuth = torch.atan2(x, z) * (180 / math.pi)

    # Calculate the vertical angle (pitch/elevation).
    elevation = torch.atan2(y, torch.sqrt(x ** 2 + z ** 2)) * (180 / math.pi)

    # Calculate the angular difference from the center view direction (handling angle wrapping).
    diff_azimuth = (azimuth - center_yaw)
    diff_azimuth = torch.remainder(diff_azimuth + 180, 360) - 180

    diff_elevation = (elevation - center_pitch)
    diff_elevation = torch.remainder(diff_elevation + 180, 360) - 180

    # Check if within FOV
    in_fov_h = diff_azimuth.abs() < fov_half_h
    in_fov_v = diff_elevation.abs() < fov_half_v

    return in_fov_h & in_fov_v


def calculate_fov_overlap_similarity(
        w2c_matrix_curr: torch.Tensor,
        w2c_matrix_hist: torch.Tensor,
        fov_h_deg: float = 105.0,
        fov_v_deg: float = 75.0,
        device=None,
        points_local=None,
) -> float:
    """
        Calculate the Field-of-View (FOV) overlap similarity between two W2C poses using Monte Carlo sampling.

        Similarity = (Number of points in Curr_FOV ∩ Hist_FOV) / (Number of points in Curr_FOV).

        :param w2c_matrix_curr: The (4, 4) W2C matrix for the current frame.
        :param w2c_matrix_hist: The (4, 4) W2C matrix for the historical frame.
        :param num_samples, radius, fov_h_deg, fov_v_deg: Sampling and FOV parameters.
        :return: The overlap ratio (a float between 0.0 and 1.0).
    """
    w2c_matrix_curr = torch.tensor(w2c_matrix_curr, device=device)
    w2c_matrix_hist = torch.tensor(w2c_matrix_hist, device=device)

    c2w_matrix_curr = torch.linalg.inv(w2c_matrix_curr)
    c2w_matrix_hist = torch.linalg.inv(w2c_matrix_hist)
    C_inv = w2c_matrix_curr

    w2c_matrix_curr = torch.linalg.inv(C_inv @ c2w_matrix_curr)
    w2c_matrix_hist = torch.linalg.inv(C_inv @ c2w_matrix_hist)

    R_curr, t_curr = w2c_matrix_curr[:3, :3], w2c_matrix_curr[:3, 3]
    R_hist, t_hist = w2c_matrix_hist[:3, :3], w2c_matrix_hist[:3, 3]
    P_w_curr = -R_curr.T @ t_curr
    P_w_hist = -R_hist.T @ t_hist

    # pitch, yaw
    pitch_curr, yaw_curr = rotation_matrix_to_angles(R_curr)
    pitch_hist, yaw_hist = rotation_matrix_to_angles(R_hist)

    fov_half_h = torch.tensor(fov_h_deg / 2.0, device=device)
    fov_half_v = torch.tensor(fov_v_deg / 2.0, device=device)

    # move to P_w_curr (N, 3)
    points_world = points_local + P_w_curr[None, :]

    in_fov_curr = is_inside_fov_3d_hv(
        points_world, P_w_curr[None, :],
        pitch_curr[None], yaw_curr[None],
        fov_half_h, fov_half_v
    )

    # compute based on angle
    in_fov_hist = is_inside_fov_3d_hv(
        points_world, P_w_hist[None, :],
        pitch_hist[None], yaw_hist[None],
        fov_half_h, fov_half_v
    )

    # compute based on distance
    dist = torch.norm(points_world - P_w_hist.reshape(1, -1), dim=1) < 8.0
    in_fov_hist = in_fov_hist.bool() & dist.reshape(1, -1).bool()

    overlap_count = (in_fov_curr.bool() & in_fov_hist.bool()).sum().float()
    fov_curr_count = in_fov_curr.sum().float()

    if fov_curr_count == 0:
        return 0.0  

    overlap_ratio = overlap_count / fov_curr_count

    return overlap_ratio.item()


def select_aligned_memory_frames(
                                 w2c_list: List[np.ndarray], current_frame_idx: int,
                                 memory_frames: int, temporal_context_size: int,
                                 pred_latent_size: int, pos_weight: float = 1.0,
                                 ang_weight: float = 1.0, device=None,
                                 points_local=None) -> List[int]:
    """
        Selects memory and context frames for a given frame based on a four-frame segment distance calculation.

        :param w2c_list: List of all N 4x4 World-to-Camera (W2C) extrinsic matrices (np.ndarray).
        :param current_frame_idx: The index of the current frame to be processed.
        :param memory_frames: The total number of memory frames to select.
        :param context_size: The total number of context frames to select.
        :param pos_weight: The weight applied to the spatial (position) distance component.
        :param ang_weight: The weight applied to the angular distance component.

        :return: List[int]: A list containing the indices of the selected memory frames and context frames.
    """
    if current_frame_idx <= memory_frames:
        return list(range(0, current_frame_idx))

    num_total_frames = len(w2c_list)
    if current_frame_idx >= num_total_frames or current_frame_idx < 3:
        raise ValueError(
            f"The current frame index must be within the valid range of w2c_list and must be at least 3."
            f"{current_frame_idx}, {len(w2c_list)}"
            )

    start_context_idx = max(0, current_frame_idx - temporal_context_size)
    context_frames_indices = list(range(start_context_idx, current_frame_idx))

    candidate_distances = []
    query_clip_indices = list(
        range(
            current_frame_idx,
            current_frame_idx + pred_latent_size 
            if current_frame_idx + pred_latent_size <= num_total_frames 
            else num_total_frames
        )
    )

    historical_clip_indices = list(range(4, current_frame_idx - temporal_context_size, 4))

    memory_frames_indices = [0,1,2,3]  # add the first chunk as context
    memory_frames = memory_frames - temporal_context_size

    for hist_idx in historical_clip_indices:
        total_dist = 0
        hist_w2c_1 = w2c_list[hist_idx]
        hist_w2c_2 = w2c_list[hist_idx + 2]
        for query_idx in query_clip_indices:
            dist_1_for_query_idx = 1.0 - calculate_fov_overlap_similarity(w2c_list[query_idx], hist_w2c_1,
                                                                          fov_h_deg=60.0, fov_v_deg=35.0,
                                                                          device=device, points_local=points_local)
            dist_2_for_query_idx = 1.0 - calculate_fov_overlap_similarity(w2c_list[query_idx], hist_w2c_2,
                                                                          fov_h_deg=60.0, fov_v_deg=35.0,
                                                                          device=device, points_local=points_local)
            dist_for_query_idx = (dist_1_for_query_idx + dist_2_for_query_idx) / 2.0
            total_dist += dist_for_query_idx

        final_clip_distance = total_dist / len(query_clip_indices)
        candidate_distances.append((hist_idx, final_clip_distance))

    candidate_distances.sort(key=lambda x: x[1])

    for start_idx, _ in candidate_distances:
        # check the memory frame number
        if len(memory_frames_indices) >= memory_frames:
            break

        if start_idx not in memory_frames_indices:
            memory_frames_indices.extend(range(start_idx, start_idx + 4))

    # exclude the repeated frames
    selected_frames_set = set(context_frames_indices)
    selected_frames_set.update(memory_frames_indices)

    final_selected_frames = sorted(list(selected_frames_set))

    return final_selected_frames


class DP_SP_BatchSampler(Sampler[list[int]]):
    """
    A simple sequential batch sampler that yields batches of indices.
    """

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
            global_indices = global_indices[:num_global_batches *
                                             self.num_sp_groups *
                                             self.batch_size]
        else:
            if self.dataset_size % (self.num_sp_groups * self.batch_size) != 0:
                # add more indices to make it divisible by (batch_size * num_sp_groups)
                padding_size = self.num_sp_groups * self.batch_size - (
                        self.dataset_size % (self.num_sp_groups * self.batch_size))
                logger.info("Padding the dataset from %d to %d",
                            self.dataset_size, self.dataset_size + padding_size)
                global_indices = torch.cat(
                    [global_indices, global_indices[:padding_size]])

        # shard the indices to each sp group
        ith_sp_group = self.global_rank // self.sp_world_size
        sp_group_local_indices = global_indices[ith_sp_group::self.
        num_sp_groups]
        self.sp_group_local_indices = sp_group_local_indices
        logger.info("Dataset size for each sp group: %d",
                    len(sp_group_local_indices))

    def __iter__(self):
        indices = self.sp_group_local_indices
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield batch_indices.tolist()

    def __len__(self):
        return len(self.sp_group_local_indices) // self.batch_size

class CameraJsonWMemDataset(Dataset):
    def __init__(self, json_path, causal, window_frames, batch_size, cfg_rate, i2v_rate, drop_last, drop_first_row,
                 seed, device, shared_state):
        self.json_data = json.load(open(json_path, 'r'))
        self.all_length = len(self.json_data)
        self.causal = causal
        self.window_frames = window_frames
        self.memory_frames = 20
        self.cfg_rate = cfg_rate
        self.rng = random.Random(seed)
        self.i2v_rate = i2v_rate
        self.device = device
        self.shared_state = shared_state

        self.sampler = DP_SP_BatchSampler(
            batch_size=batch_size,
            dataset_size=self.all_length,
            num_sp_groups=get_world_size() // get_sp_world_size(),
            sp_world_size=get_sp_world_size(),
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

        self.points_local = generate_points_in_sphere(50000, 8.0).to(device)

        self.neg_prompt_pt = torch.load(
            "./model_ckpts/hunyuan_neg_prompt.pt",
            map_location="cpu",
            weights_only=True,
        )

        self.neg_byt5_pt = torch.load(
            "./model_ckpts/hunyuan_neg_byt5_prompt.pt",
            map_location="cpu",
            weights_only=True,
        )

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

    def perturb_pose(self, w2c_list, current_frame_idx, generate_length):
        total_latent_num = w2c_list.shape[0]
        context_w2c = w2c_list[:current_frame_idx]

        pertubed_left = self.rng.random() < 0.5
        if pertubed_left:
            generate_w2c =  w2c_list[current_frame_idx - 4: current_frame_idx - 4 + generate_length]
        else:
            if total_latent_num >= current_frame_idx + 4 + generate_length:
                generate_w2c = w2c_list[current_frame_idx + 4: current_frame_idx + 4 + generate_length]
            else:
                generate_w2c = w2c_list[current_frame_idx: current_frame_idx + generate_length]

        return np.concatenate((context_w2c, generate_w2c), axis=0)

    def update_max_frames(self, training_step):
        if training_step < 500:
            self.shared_state["max_frames"] = 32
        elif training_step < 1000:
            self.shared_state["max_frames"] = 64
        elif training_step < 2000:
            self.shared_state["max_frames"] = 96
        elif training_step < 3000:
            self.shared_state["max_frames"] = 128
        else:
            self.shared_state["max_frames"] = 160

    def __getitem__(self, idx):
        while True:
            try:
                json_data = self.json_data[idx]
                latent_pt_path = json_data['latent_path']
                pose_path = json_data['pose_path']

                latent_pt = torch.load(
                    os.path.join(latent_pt_path),
                    map_location="cpu",
                    weights_only=True,
                )
                latent = latent_pt['latent'][0]
                latent_length = latent.shape[1]

                # if self.causal:
                if latent_length < self.window_frames:
                    idx = self.rng.randint(0, self.all_length - 1)
                    continue
                else:
                    max_frames = int(self.shared_state["max_frames"]) // 4 * 4
                    max_length = min(max_frames, latent_length // 4 * 4)

                latent = latent[:, :max_length, ...]  # mostly larger than window_frames, need chunk

                prompt_embed = latent_pt['prompt_embeds'][0]
                prompt_mask = latent_pt['prompt_mask'][0]

                image_cond = latent_pt['image_cond'][0]
                vision_states = latent_pt['vision_states'][0]
                byt5_text_states = latent_pt['byt5_text_states'][0]
                byt5_text_mask = latent_pt['byt5_text_mask'][0]

                if self.rng.random() < self.cfg_rate:
                    prompt_embed = self.neg_prompt_pt['negative_prompt_embeds'][0]
                    prompt_mask = self.neg_prompt_pt['negative_prompt_mask'][0]
                    byt5_text_states = self.neg_byt5_pt['byt5_text_states'][0]
                    byt5_text_mask = self.neg_byt5_pt['byt5_text_mask'][0]

                pose_json = json.load(open(pose_path, 'r'))
                pose_keys = list(pose_json.keys())
                intrinsic_list = []
                w2c_list = []
                for i in range(latent.shape[1]):
                    t_key = pose_keys[0] if i == 0 else pose_keys[4 * (i - 1) + 4]
                    intrinsic = np.array(pose_json[t_key]['intrinsic'])
                    w2c = np.array(pose_json[t_key]['w2c'])

                    intrinsic[0, 0] /= intrinsic[0, 2] * 2
                    intrinsic[1, 1] /= intrinsic[1, 2] * 2
                    intrinsic[0, 2] = 0.5
                    intrinsic[1, 2] = 0.5
                    w2c_list.append(w2c)
                    intrinsic_list.append(intrinsic)

                w2c_list = np.array(w2c_list)
                w2c_list = self.camera_center_normalization(w2c_list)
                intrinsic_list = torch.tensor(np.array(intrinsic_list))

                if 'latent_dataset_w_action' in latent_pt_path:    # prepare for dataset with action labels
                    trans_one_hot = np.zeros((intrinsic_list.shape[0], 4), dtype=np.int32)
                    rotate_one_hot = np.zeros((intrinsic_list.shape[0], 4), dtype=np.int32)
                    action_json = json.load(open(json_data["action_path"], 'r'))
                    action_keys = list(action_json.keys())
                    for action_idx in range(1, trans_one_hot.shape[0]):
                        t_key = action_keys[4 * (action_idx - 1) + 4]
                        t_move_action = action_json[t_key]["move_action"]
                        t_view_action = action_json[t_key]["view_action"]
                        if "W" in t_move_action and "S" not in t_move_action:
                            trans_one_hot[action_idx, 0] = 1
                        if "S" in t_move_action and "W" not in t_move_action:
                            trans_one_hot[action_idx, 1] = 1
                        if "D" in t_move_action and "A" not in t_move_action:
                            trans_one_hot[action_idx, 2] = 1
                        if "A" in t_move_action and "D" not in t_move_action:
                            trans_one_hot[action_idx, 3] = 1

                        if t_view_action == "LR":
                            rotate_one_hot[action_idx, 0] = 1
                        elif t_view_action == "LL":
                            rotate_one_hot[action_idx, 1] = 1
                        elif t_view_action == "LU":
                            rotate_one_hot[action_idx, 2] = 1
                        elif t_view_action == "LD":
                            rotate_one_hot[action_idx, 3] = 1

                    trans_one_label = self.one_hot_to_one_dimension(trans_one_hot)
                    rotate_one_label = self.one_hot_to_one_dimension(rotate_one_hot)
                    action_for_pe = trans_one_label * 9 + rotate_one_label

                else:    # prepare action labels on the fly
                    c2ws = np.linalg.inv(w2c_list)
                    C_inv = np.linalg.inv(c2ws[:-1])
                    relative_c2w = np.zeros_like(c2ws)
                    relative_c2w[0, ...] = c2ws[0, ...]
                    relative_c2w[1:, ...] = C_inv @ c2ws[1:, ...]
                    trans_one_hot = np.zeros((relative_c2w.shape[0], 4), dtype=np.int32)
                    rotate_one_hot = np.zeros((relative_c2w.shape[0], 4), dtype=np.int32)

                    move_norm_valid = 0.01
                    for i in range(1, relative_c2w.shape[0]):
                        move_dirs = relative_c2w[i, :3, 3]
                        move_norms = np.linalg.norm(move_dirs)
                        # compute translation angles
                        if move_norms > move_norm_valid:
                            move_norm_dirs = move_dirs / move_norms
                            angles_rad = np.arccos(move_norm_dirs.clip(-1.0, 1.0))
                            trans_angles_deg = angles_rad * (180.0 / torch.pi)

                            if trans_angles_deg[2] < 60:
                                trans_one_hot[i, 0] = 1 
                            elif trans_angles_deg[2] > 120:
                                trans_one_hot[i, 1] = 1

                            if trans_angles_deg[0] < 60:
                                trans_one_hot[i, 2] = 1 
                            elif trans_angles_deg[0] > 120:
                                trans_one_hot[i, 3] = 1 
                        else:
                            trans_angles_deg = torch.zeros(3)

                        R_rel = relative_c2w[i, :3, :3]
                        r = R.from_matrix(R_rel)
                        rot_angles_deg = r.as_euler('xyz', degrees=True)

                        # compute rotation angles
                        if rot_angles_deg[1] > 5e-2:
                            rotate_one_hot[i, 0] = 1
                        elif rot_angles_deg[1] < -5e-2:
                            rotate_one_hot[i, 1] = 1

                        if rot_angles_deg[0] > 5e-2:
                            rotate_one_hot[i, 2] = 1
                        elif rot_angles_deg[0] < -5e-2:
                            rotate_one_hot[i, 3] = 1

                    trans_one_hot = torch.tensor(trans_one_hot)
                    rotate_one_hot = torch.tensor(rotate_one_hot)

                    trans_one_label = self.one_hot_to_one_dimension(trans_one_hot)
                    rotate_one_label = self.one_hot_to_one_dimension(rotate_one_hot)
                    action_for_pe = trans_one_label * 9 + rotate_one_label

                select_window_out_flag = 0  # whether to select the latents with length > window_frames
                select_prob = self.rng.random()

                if select_prob < 0.8:
                    select_window_out_flag = 1  # mean to select frames outside the window
                    max_index = latent.shape[1] - (self.window_frames - self.memory_frames)

                    start_chunk_id = (self.window_frames) // 4
                    end_chunk_id = max_index // 4
                    current_frame_idx = self.rng.randint(start_chunk_id, end_chunk_id) * 4  # include the left and right

                    # -------------------- for ar, only search the memory for the current chunk
                    selected_history_frame_id = select_aligned_memory_frames(w2c_list, 
                                                                            current_frame_idx, 
                                                                            memory_frames=self.memory_frames, 
                                                                            temporal_context_size=12, 
                                                                            pred_latent_size=4, 
                                                                            points_local=self.points_local, 
                                                                            device=self.device)   # align the training objective: refine the fov selection
                    selected_history_frame_id.extend(range(current_frame_idx, current_frame_idx + 4))
                    latent = latent[:, selected_history_frame_id]
                    reset_w2c_list = w2c_list[selected_history_frame_id]
                    w2c_list = reset_w2c_list
                    reset_intrinsic_list = intrinsic_list[selected_history_frame_id]
                    intrinsic_list = reset_intrinsic_list
                    reset_action_for_pe = action_for_pe[selected_history_frame_id]
                    action_for_pe = reset_action_for_pe

                else:
                    pred_latent_size = self.window_frames
                    latent = latent[:, :pred_latent_size, ...]
                    w2c_list = w2c_list[:pred_latent_size]
                    intrinsic_list = intrinsic_list[:pred_latent_size]
                    action_for_pe = action_for_pe[:pred_latent_size]

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
                    "select_window_out_flag": select_window_out_flag,  # select frames outside the window or not
                    "video_path": json_data["pose_path"],
                    "max_length": max_frames,

                    "image_cond": image_cond,
                    "vision_states": vision_states,
                    "prompt_mask": prompt_mask,
                    "byt5_text_states": byt5_text_states,
                    "byt5_text_mask": byt5_text_mask,
                }
                break
            except Exception as e:
                print('error:', e, latent_pt_path, flush=True)
                idx = self.rng.randint(0, self.all_length - 1)
        return batch


def cycle(dl):
    while True:
        for data in dl:
            yield data


def latent_collate_function(batch):
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
    byt5_text_states = torch.stack([b["byt5_text_states"] for b in batch], dim=0)
    byt5_text_mask = torch.stack([b["byt5_text_mask"] for b in batch], dim=0)


    context_frames_list = [b["context_frames_list"] for b in batch]
    select_window_out_flag = [b["select_window_out_flag"] for b in batch]
    video_path = [b["video_path"] for b in batch]
    max_length = [b["max_length"] for b in batch]

    return {
        "i2v_mask": i2v_mask,
        "latent": latent,
        "prompt_embed": prompt_embed,
        "w2c": w2c,
        "intrinsic": intrinsic,
        "action": action,
        "video_path": video_path,
        "context_frames_list": context_frames_list,
        "select_window_out_flag": select_window_out_flag,
        "action_for_pe": action_for_pe,
        "max_length": max_length,
        "image_cond": image_cond,
        "vision_states": vision_states,
        "prompt_mask": prompt_mask,
        "byt5_text_states": byt5_text_states,
        "byt5_text_mask": byt5_text_mask,
    }


def build_ar_camera_hunyuan_w_mem_dataloader(
        json_path,
        causal,
        window_frames,
        # memory_frames,
        batch_size,
        num_data_workers,
        drop_last,
        drop_first_row,
        seed,
        cfg_rate,
        i2v_rate, ) -> tuple[CameraJsonWMemDataset, StatefulDataLoader]:
    manager = mp.Manager()
    shared_state = manager.dict()
    shared_state["max_frames"] = window_frames

    dataset = CameraJsonWMemDataset(json_path, causal, window_frames, batch_size, cfg_rate, i2v_rate,
                                    drop_last=drop_last, drop_first_row=drop_first_row, seed=seed, 
                                    device=get_local_torch_device(), shared_state=shared_state)

    loader = StatefulDataLoader(
        dataset,
        batch_sampler=dataset.sampler,
        collate_fn=latent_collate_function,
        num_workers=num_data_workers,
        pin_memory=True,
        persistent_workers=num_data_workers > 0,
    )
    return dataset, loader
