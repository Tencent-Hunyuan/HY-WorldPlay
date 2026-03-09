from pathlib import Path

import numpy as np
import torch
import moviepy.editor as mpy

from reward_function.HunyuanWorldMirror.src.models.models.rasterization import (
    GaussianSplatRenderer,
)
from reward_function.HunyuanWorldMirror.src.models.utils.sh_utils import (
    RGB2SH,
    SH2RGB,
)
from reward_function.HunyuanWorldMirror.src.utils.gs_effects import GSEffects
from reward_function.HunyuanWorldMirror.src.utils.color_map import (
    apply_color_map_to_image,
)
from tqdm import tqdm


def rotation_matrix_to_quaternion(R):
    """Convert rotation matrix to quaternion."""
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]

    q = torch.zeros(R.shape[:-2] + (4,), device=R.device, dtype=R.dtype)

    # Case where trace > 0
    mask1 = trace > 0
    s = torch.sqrt(trace[mask1] + 1.0) * 2  # s=4*qw
    q[mask1, 0] = 0.25 * s  # qw
    q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s  # qx
    q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s  # qy
    q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s  # qz

    # Case where R[0,0] > R[1,1] and R[0,0] > R[2,2]
    mask2 = (
        (~mask1) & (R[..., 0, 0] > R[..., 1, 1]) & (R[..., 0, 0] > R[..., 2, 2])
    )
    s = (
        torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2
    )  # s=4*qx
    q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s  # qw
    q[mask2, 1] = 0.25 * s  # qx
    q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s  # qy
    q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s  # qz

    # Case where R[1,1] > R[2,2]
    mask3 = (~mask1) & (~mask2) & (R[..., 1, 1] > R[..., 2, 2])
    s = (
        torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2
    )  # s=4*qy
    q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s  # qw
    q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s  # qx
    q[mask3, 2] = 0.25 * s  # qy
    q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s  # qz

    # Remaining case
    mask4 = (~mask1) & (~mask2) & (~mask3)
    s = (
        torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2
    )  # s=4*qz
    q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s  # qw
    q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s  # qx
    q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s  # qy
    q[mask4, 3] = 0.25 * s  # qz

    return q


def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Normalize quaternion
    norm = torch.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    R = torch.zeros(q.shape[:-1] + (3, 3), device=q.device, dtype=q.dtype)

    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - w * z)
    R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y)
    R[..., 2, 1] = 2 * (y * z + w * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)

    return R


def slerp_quaternions(q1, q2, t):
    """Spherical linear interpolation between quaternions."""
    # Compute dot product
    dot = (q1 * q2).sum(dim=-1, keepdim=True)

    # If dot product is negative, slerp won't take the shorter path.
    # Note that q and -q represent the same rotation, so we can flip one.
    mask = dot < 0
    q2 = torch.where(mask, -q2, q2)
    dot = torch.where(mask, -dot, dot)

    # If the inputs are too close for comfort, linearly interpolate
    # and normalize the result.
    DOT_THRESHOLD = 0.9995
    mask_linear = dot > DOT_THRESHOLD

    result = torch.zeros_like(q1)

    # Linear interpolation for close quaternions
    if mask_linear.any():
        result_linear = q1 + t * (q2 - q1)
        norm = torch.norm(result_linear, dim=-1, keepdim=True)
        result_linear = result_linear / norm
        result = torch.where(mask_linear, result_linear, result)

    # Spherical interpolation for distant quaternions
    mask_slerp = ~mask_linear
    if mask_slerp.any():
        theta_0 = torch.acos(torch.abs(dot))
        sin_theta_0 = torch.sin(theta_0)

        theta = theta_0 * t
        sin_theta = torch.sin(theta)

        s0 = torch.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0

        result_slerp = (s0 * q1) + (s1 * q2)
        result = torch.where(mask_slerp, result_slerp, result)

    return result


def render_interpolated_video(
    gs_renderer: GaussianSplatRenderer,
    splats: dict,
    camtoworlds: torch.Tensor,
    intrinsics: torch.Tensor,
    hw: tuple[int, int],
    out_path: Path,
    interp_per_pair: int = 20,
    loop_reverse: bool = True,
    effects: GSEffects = None,
    effect_type: int = 2,
    save_mode: str = "split",
) -> None:
    # camtoworlds: [B, S, 4, 4], intrinsics: [B, S, 3, 3]
    b, s, _, _ = camtoworlds.shape
    h, w = hw

    # Build interpolated trajectory
    def build_interpolated_traj(index, nums):
        exts, ints = [], []
        tmp_camtoworlds = camtoworlds[:, index]
        tmp_intrinsics = intrinsics[:, index]
        for i in range(len(index) - 1):
            exts.append(tmp_camtoworlds[:, i : i + 1])
            ints.append(tmp_intrinsics[:, i : i + 1])
            # Extract rotation and translation
            R0, t0 = tmp_camtoworlds[:, i, :3, :3], tmp_camtoworlds[:, i, :3, 3]
            R1, t1 = (
                tmp_camtoworlds[:, i + 1, :3, :3],
                tmp_camtoworlds[:, i + 1, :3, 3],
            )

            # Convert rotations to quaternions
            q0 = rotation_matrix_to_quaternion(R0)
            q1 = rotation_matrix_to_quaternion(R1)

            # Interpolate using smooth quaternion slerp
            for j in range(1, nums + 1):
                alpha = j / (nums + 1)

                # Linear interpolation for translation
                t_interp = (1 - alpha) * t0 + alpha * t1

                # Spherical interpolation for rotation
                q_interp = slerp_quaternions(q0, q1, alpha)
                R_interp = quaternion_to_rotation_matrix(q_interp)

                # Create interpolated extrinsic matrix
                ext = torch.eye(
                    4, device=R_interp.device, dtype=R_interp.dtype
                )[None].repeat(b, 1, 1)
                ext[:, :3, :3] = R_interp
                ext[:, :3, 3] = t_interp

                # Linear interpolation for intrinsics
                K0 = tmp_intrinsics[:, i]
                K1 = tmp_intrinsics[:, i + 1]
                K = (1 - alpha) * K0 + alpha * K1

                exts.append(ext[:, None])
                ints.append(K[:, None])

        exts = torch.cat(exts, dim=1)[:1]
        ints = torch.cat(ints, dim=1)[:1]
        return exts, ints

    # Build wobble trajectory
    def build_wobble_traj(nums, delta):
        assert s == 1
        t = torch.linspace(
            0, 1, nums, dtype=torch.float32, device=camtoworlds.device
        )
        t = (torch.cos(torch.pi * (t + 1)) + 1) / 2
        tf = torch.eye(4, dtype=torch.float32, device=camtoworlds.device)
        radius = delta * 0.15
        tf = tf.broadcast_to((*radius.shape, t.shape[0], 4, 4)).clone()
        radius = radius[..., None]
        radius = radius * t
        tf[..., 0, 3] = torch.sin(2 * torch.pi * t) * radius
        tf[..., 1, 3] = -torch.cos(2 * torch.pi * t) * radius
        exts = camtoworlds @ tf
        ints = intrinsics.repeat(1, exts.shape[1], 1, 1)
        return exts, ints

    if s > 1:
        all_ext, all_int = build_interpolated_traj(
            [i for i in range(s)], interp_per_pair
        )
    else:
        all_ext, all_int = build_wobble_traj(
            interp_per_pair * 12,
            splats["means"][0].median(dim=0).values.norm(dim=-1)[None],
        )

    rendered_rgbs, rendered_depths = [], []
    chunk = 40 if effects is None else 1
    t = 0
    t_skip = 0
    if effects is not None:
        try:
            pruned_splats = gs_renderer.prune_gs(splats, gs_renderer.voxel_size)
        except:
            pruned_splats = splats
        # indices = [x for x in range(0, all_ext.shape[1], 2)][:4]
        # add_ext, add_int = build_interpolated_traj(indices, 150)
        # add_ext = torch.flip(add_ext, dims=[1])
        # add_int = torch.flip(add_int, dims=[1])
        add_ext = all_ext[:, :1, :, :].repeat(1, 320, 1, 1)
        add_int = all_int[:, :1, :, :].repeat(1, 320, 1, 1)
        shift = pruned_splats["means"][0].median(dim=0).values
        scale_factor = (
            (pruned_splats["means"][0] - shift)
            .abs()
            .quantile(0.95, dim=0)
            .max()
        )
        all_ext[0, :, :3, -1] = (all_ext[0, :, :3, -1] - shift) / scale_factor
        add_ext[0, :, :3, -1] = (add_ext[0, :, :3, -1] - shift) / scale_factor
        flag = None
        try:
            raw_splats = gs_renderer.rasterizer.runner.splats
        except:
            pass
        for st in range(0, add_ext.shape[1]):
            ed = min(st + 1, add_ext.shape[1])
            assert gs_renderer.sh_degree == 0
            if flag is not None and (flag < 0.99).any():
                break
            sample_gsplat = {
                "means": (pruned_splats["means"][0] - shift) / scale_factor,
                "quats": pruned_splats["quats"][0],
                "scales": pruned_splats["scales"][0] / scale_factor,
                "opacities": pruned_splats["opacities"][0],
                "colors": SH2RGB(pruned_splats["sh"][0].reshape(-1, 3)),
            }
            effects_splats, flag = effects.apply_effect(
                sample_gsplat, t, effect_type=effect_type
            )
            t += 0.04
            effects_splats["sh"] = RGB2SH(effects_splats["colors"]).reshape(
                -1, 1, 3
            )
            try:
                gs_renderer.rasterizer.runner.splats
                effects_splats["sh0"] = effects_splats["sh"][:, :1, :]
                effects_splats["shN"] = effects_splats["sh"][:, 1:, :]
                effects_splats["scales"] = effects_splats["scales"].log()
                effects_splats["opacities"] = torch.logit(
                    torch.clamp(effects_splats["opacities"], 1e-6, 1 - 1e-6)
                )
                gs_renderer.rasterizer.runner.splats = effects_splats
                colors, depths, _ = gs_renderer.rasterizer.rasterize_batches(
                    None,
                    None,
                    None,
                    None,
                    None,
                    add_ext[:, st:ed].to(torch.float32),
                    add_int[:, st:ed].to(torch.float32),
                    width=w,
                    height=h,
                    sh_degree=gs_renderer.sh_degree,
                )
            except:
                colors, depths, _ = gs_renderer.rasterizer.rasterize_batches(
                    effects_splats["means"][None],
                    effects_splats["quats"][None],
                    effects_splats["scales"][None],
                    effects_splats["opacities"][None],
                    effects_splats["sh"][None],
                    add_ext[:, st:ed].to(torch.float32),
                    add_int[:, st:ed].to(torch.float32),
                    width=w,
                    height=h,
                    sh_degree=(
                        gs_renderer.sh_degree if "sh" in pruned_splats else None
                    ),
                )

            if st > add_ext.shape[1] * 0.14:
                t_skip = t if t_skip == 0 else t_skip
                # break
                rendered_rgbs.append(colors)
                rendered_depths.append(depths)
            # if (flag == 0).all():
            #     break
    t_st = t
    t_ed = 0
    loop_dir = 1
    ignore_scale = False
    for st in tqdm(range(0, all_ext.shape[1], chunk)):
        ed = min(st + chunk, all_ext.shape[1])
        if effects is not None:
            try:
                sample_gsplat = {
                    "means": (pruned_splats["means"][0] - shift) / scale_factor,
                    "quats": pruned_splats["quats"][0],
                    "scales": pruned_splats["scales"][0] / scale_factor,
                    "opacities": pruned_splats["opacities"][0],
                    "colors": SH2RGB(pruned_splats["sh"][0].reshape(-1, 3)),
                }
            except:
                sample_gsplat = {
                    "means": (pruned_splats["means"][0] - shift) / scale_factor,
                    "quats": pruned_splats["quats"][0],
                    "scales": pruned_splats["scales"][0] / scale_factor,
                    "opacities": pruned_splats["opacities"][0],
                    "colors": SH2RGB(pruned_splats["sh"][0].reshape(-1, 3)),
                }
            effects_splats, flag = effects.apply_effect(
                sample_gsplat,
                t,
                effect_type=effect_type,
                ignore_scale=ignore_scale,
            )
            if loop_dir < 0:
                t -= 0.04
            else:
                t += 0.04
            if flag.mean() < 0.01 and t_ed == 0:
                t_ed = t
            effects_splats["sh"] = RGB2SH(effects_splats["colors"]).reshape(
                -1, 1, 3
            )
            effects_splats["sh0"] = effects_splats["sh"][:, :1, :]
            effects_splats["shN"] = effects_splats["sh"][:, 1:, :]
            try:
                gs_renderer.rasterizer.runner.splats
                effects_splats["sh0"] = effects_splats["sh"][:, :1, :]
                effects_splats["shN"] = effects_splats["sh"][:, 1:, :]
                effects_splats["scales"] = effects_splats["scales"].log()
                effects_splats["opacities"] = torch.logit(
                    torch.clamp(effects_splats["opacities"], 1e-6, 1 - 1e-6)
                )
                gs_renderer.rasterizer.runner.splats = effects_splats
                colors, depths, _ = gs_renderer.rasterizer.rasterize_batches(
                    None,
                    None,
                    None,
                    None,
                    None,
                    all_ext[:, st:ed].to(torch.float32),
                    all_int[:, st:ed].to(torch.float32),
                    width=w,
                    height=h,
                    sh_degree=gs_renderer.sh_degree,
                )
            except:
                colors, depths, _ = gs_renderer.rasterizer.rasterize_batches(
                    effects_splats["means"][None],
                    effects_splats["quats"][None],
                    effects_splats["scales"][None],
                    effects_splats["opacities"][None],
                    effects_splats["sh"][None],
                    all_ext[:, st:ed].to(torch.float32),
                    all_int[:, st:ed].to(torch.float32),
                    width=w,
                    height=h,
                    sh_degree=(
                        gs_renderer.sh_degree if "sh" in pruned_splats else None
                    ),
                )

            if (
                t
                > (all_ext.shape[1]) * 0.04
                + t_st
                - (t_ed - t_st) * 2
                - 15 * 0.04
                or t < t_st
            ):
                # ignore_scale = True
                loop_dir *= -1
                t = t_ed if loop_dir == -1 else t
        else:
            colors, depths, _ = gs_renderer.rasterizer.rasterize_batches(
                splats["means"][:1],
                splats["quats"][:1],
                splats["scales"][:1],
                splats["opacities"][:1],
                splats["sh"][:1] if "sh" in splats else splats["colors"][:1],
                all_ext[:, st:ed].to(torch.float32),
                all_int[:, st:ed].to(torch.float32),
                width=w,
                height=h,
                sh_degree=gs_renderer.sh_degree if "sh" in splats else None,
            )
        rendered_rgbs.append(colors)
        rendered_depths.append(depths)

    rgbs = torch.cat(rendered_rgbs, dim=1)[0]  # [N, H, W, 3]
    depths = torch.cat(rendered_depths, dim=1)[0, ..., 0]  # [N, H, W]

    def depth_vis(d: torch.Tensor) -> torch.Tensor:
        valid = d > 0
        if valid.any():
            near = d[valid].float().quantile(0.01).log()
        else:
            near = torch.tensor(0.0, device=d.device)
        far = d.flatten().float().quantile(0.99).log()
        x = d.float().clamp(min=1e-9).log()
        x = 1.0 - (x - near) / (far - near + 1e-9)
        return apply_color_map_to_image(x, "turbo")

    frames = []
    rgb_frames = []
    depth_frames = []

    for rgb, dep in zip(rgbs, depths):
        rgb_img = rgb.permute(2, 0, 1)  # [3, H, W]
        depth_img = depth_vis(dep)  # [3, H, W]

        if save_mode == "both":
            combined = torch.cat([rgb_img, depth_img], dim=1)  # [3, 2*H, W]
            frames.append(combined)
        elif save_mode == "split":
            rgb_frames.append(rgb_img)
            depth_frames.append(depth_img)
        else:
            raise ValueError("save_mode must be 'both' or 'split'")

    def _make_video(frames, path):
        video = torch.stack(frames).clamp(0, 1)  # [N, 3, H, W]
        video = video.permute(0, 2, 3, 1)  # [N, H, W, 3] for moviepy
        video = (video * 255).to(torch.uint8).cpu().numpy()
        if loop_reverse and video.shape[0] > 1:
            video = np.concatenate([video, video[::-1][1:-1]], axis=0)
        clip = mpy.ImageSequenceClip(list(video), fps=30)
        clip.write_videofile(str(path), logger=None)

    # Save videos
    if save_mode == "both":
        _make_video(frames, f"{out_path}.mp4")
    elif save_mode == "split":
        _make_video(rgb_frames, f"{out_path}_rgb.mp4")
        _make_video(depth_frames, f"{out_path}_depth.mp4")

    print(f"Video saved to {out_path} (mode: {save_mode})")

    if effects is not None:
        try:
            gs_renderer.rasterizer.runner.splats = raw_splats
        except:
            pass
    torch.cuda.empty_cache()
