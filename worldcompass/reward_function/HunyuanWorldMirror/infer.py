import argparse
import glob
from pathlib import Path
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image
import onnxruntime

from reward_function.HunyuanWorldMirror.src.models.models.worldmirror import (
    WorldMirror,
)
from reward_function.HunyuanWorldMirror.src.utils.inference_utils import (
    prepare_images_to_tensor,
)
from reward_function.HunyuanWorldMirror.src.utils.video_utils import (
    video_to_image_frames,
)
from reward_function.HunyuanWorldMirror.src.models.utils.geometry import (
    depth_to_world_coords_points,
)
from reward_function.HunyuanWorldMirror.src.models.utils.geometry import (
    create_pixel_coordinate_grid,
)

from reward_function.HunyuanWorldMirror.src.utils.save_utils import (
    save_depth_png,
    save_depth_npy,
    save_normal_png,
)
from reward_function.HunyuanWorldMirror.src.utils.save_utils import (
    save_scene_ply,
    save_gs_ply,
    save_points_ply,
)
from reward_function.HunyuanWorldMirror.src.utils.render_utils import (
    render_interpolated_video,
)

from reward_function.HunyuanWorldMirror.src.utils.build_pycolmap_recon import (
    build_pycolmap_reconstruction,
)
from reward_function.HunyuanWorldMirror.src.models.utils.camera_utils import (
    vector_to_camera_matrices,
)

# Import mask computation utilities
from reward_function.HunyuanWorldMirror.src.utils.geometry import (
    depth_edge,
    normals_edge,
)
from reward_function.HunyuanWorldMirror.src.utils.visual_util import (
    segment_sky,
    download_file_from_url,
)


def create_filter_mask(
    pts3d_conf: np.ndarray,
    depth_preds: np.ndarray,
    normal_preds: np.ndarray,
    sky_mask: np.ndarray,
    confidence_percentile: float = 10.0,
    edge_normal_threshold: float = 5.0,
    edge_depth_threshold: float = 0.03,
    apply_confidence_mask: bool = True,
    apply_edge_mask: bool = True,
    apply_sky_mask: bool = False,
) -> np.ndarray:
    """Create comprehensive filter mask based on confidence, edges, and sky segmentation. This
    follows the same logic as app.py for consistent mask computation.

    Args:
        pts3d_conf: Point confidence scores [S, H, W]
        depth_preds: Depth predictions [S, H, W, 1]
        normal_preds: Normal predictions [S, H, W, 3]
        sky_mask: Sky segmentation mask [S, H, W]
        confidence_percentile: Percentile threshold for confidence filtering (0-100)
        edge_normal_threshold: Normal angle threshold in degrees for edge detection
        edge_depth_threshold: Relative depth threshold for edge detection
        apply_confidence_mask: Whether to apply confidence-based filtering
        apply_edge_mask: Whether to apply edge-based filtering
        apply_sky_mask: Whether to apply sky mask filtering

    Returns:
        final_mask: Boolean mask array [S, H, W] for filtering points
    """
    S, H, W = pts3d_conf.shape[:3]
    final_mask_list = []

    for i in range(S):
        final_mask = None

        if apply_confidence_mask:
            # Compute confidence mask based on the pointmap confidence
            confidences = pts3d_conf[i, :, :]  # [H, W]
            percentile_threshold = np.quantile(
                confidences, confidence_percentile / 100.0
            )
            conf_mask = confidences >= percentile_threshold
            if final_mask is None:
                final_mask = conf_mask
            else:
                final_mask = final_mask & conf_mask

        if apply_edge_mask:
            # Compute edge mask based on the normalmap
            normal_pred = normal_preds[i]  # [H, W, 3]
            normal_edges = normals_edge(
                normal_pred, tol=edge_normal_threshold, mask=final_mask
            )
            # Compute depth mask based on the depthmap
            depth_pred = depth_preds[i, :, :, 0]  # [H, W]
            depth_edges = depth_edge(
                depth_pred, rtol=edge_depth_threshold, mask=final_mask
            )
            edge_mask = ~(depth_edges & normal_edges)
            if final_mask is None:
                final_mask = edge_mask
            else:
                final_mask = final_mask & edge_mask

        if apply_sky_mask:
            # Apply sky mask filtering (sky_mask is already inverted: True = non-sky)
            sky_mask_frame = sky_mask[i]  # [H, W]
            if final_mask is None:
                final_mask = sky_mask_frame
            else:
                final_mask = final_mask & sky_mask_frame

        final_mask_list.append(final_mask)

    # Stack all frame masks
    if final_mask_list[0] is not None:
        final_mask = np.stack(final_mask_list, axis=0)  # [S, H, W]
    else:
        final_mask = np.ones(pts3d_conf.shape[:3], dtype=bool)  # [S, H, W]

    return final_mask


def main():
    parser = argparse.ArgumentParser(
        description="HunyuanWorld-Mirror inference"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="examples/realistic/Ireland_Landscape",
        help="Input can be: a directory of images; a single video file; or a directory containing multiple video files (.mp4/.avi/.mov/.webm/.gif). If directory has multiple videos, frames from all clips are extracted (using --fps) and merged in filename order.",
    )
    parser.add_argument("--output_path", type=str, default="inference_output")
    parser.add_argument(
        "--fps",
        type=int,
        default=1,
        help="Frames per second for video extraction",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=518,
        help="Target size for image resizing",
    )
    parser.add_argument(
        "--write_txt",
        action="store_true",
        help="Also write human-readable COLMAP txt (slow, huge)",
    )
    # Mask filtering parameters
    parser.add_argument(
        "--confidence_percentile",
        type=float,
        default=10.0,
        help="Confidence percentile threshold for filtering (0-100, filters bottom X percent)",
    )
    parser.add_argument(
        "--edge_normal_threshold",
        type=float,
        default=5.0,
        help="Normal angle threshold in degrees for edge detection",
    )
    parser.add_argument(
        "--edge_depth_threshold",
        type=float,
        default=0.03,
        help="Relative depth threshold for edge detection",
    )
    parser.add_argument(
        "--apply_confidence_mask",
        action="store_true",
        default=True,
        help="Apply confidence-based filtering",
    )
    parser.add_argument(
        "--apply_edge_mask",
        action="store_true",
        default=True,
        help="Apply edge-based filtering",
    )
    parser.add_argument(
        "--apply_sky_mask",
        action="store_true",
        default=False,
        help="Apply sky mask filtering",
    )
    # Save flags
    parser.add_argument(
        "--save_pointmap",
        action="store_true",
        default=True,
        help="Save points PLY",
    )
    parser.add_argument(
        "--save_depth", action="store_true", default=True, help="Save depth PNG"
    )
    parser.add_argument(
        "--save_normal",
        action="store_true",
        default=True,
        help="Save normal PNG",
    )
    parser.add_argument(
        "--save_gs",
        action="store_true",
        default=True,
        help="Save Gaussians PLY",
    )
    parser.add_argument(
        "--save_rendered",
        action="store_true",
        default=True,
        help="Save rendered video",
    )
    parser.add_argument(
        "--save_colmap",
        action="store_true",
        default=True,
        help="Save COLMAP sparse",
    )
    # Conditioning flags
    parser.add_argument(
        "--cond_pose",
        action="store_true",
        help="Use camera pose conditioning if available",
    )
    parser.add_argument(
        "--cond_intrinsics",
        action="store_true",
        help="Use intrinsics conditioning if available",
    )
    parser.add_argument(
        "--cond_depth",
        action="store_true",
        help="Use depth conditioning if available",
    )
    args = parser.parse_args()

    # Print inference parameters
    print(f"🔧 Configuration:")
    print(f"  - FPS: {args.fps}")
    print(f"  - Target size: {args.target_size}px")
    print(f"  - Mask Filtering:")
    print(
        f"    - Confidence mask: {'✅' if args.apply_confidence_mask else '❌'} (percentile: {args.confidence_percentile}%)"
    )
    print(
        f"    - Edge mask: {'✅' if args.apply_edge_mask else '❌'} (normal: {args.edge_normal_threshold}°, depth: {args.edge_depth_threshold})"
    )
    print(f"    - Sky mask: {'✅' if args.apply_sky_mask else '❌'}")
    print(f"  - Conditioning:")
    print(f"    - Pose: {'✅' if args.cond_pose else '❌'}")
    print(f"    - Intrinsics: {'✅' if args.cond_intrinsics else '❌'}")
    print(f"    - Depth: {'✅' if args.cond_depth else '❌'}")

    # 1) Init model - This requires internet access or the huggingface hub cache to be pre-downloaded
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WorldMirror.from_pretrained("tencent/HunyuanWorld-Mirror").to(
        device
    )
    model.eval()

    input_path = Path(args.input_path)

    # Create output directory with filename-based subdirectory
    if input_path.is_file():
        subdir_name = input_path.stem
    elif input_path.is_dir():
        subdir_name = input_path.name
    else:
        raise ValueError(
            f"❌ Invalid input path: {input_path} (must be directory or video file)"
        )

    outdir = Path(args.output_path) / subdir_name
    outdir.mkdir(parents=True, exist_ok=True)

    # Determine input type and get image paths
    video_exts = [".mp4", ".avi", ".mov", ".webm", ".gif"]

    if input_path.is_file() and input_path.suffix.lower() in video_exts:
        # Case 1: Single video file - extract frames
        print(f"📹 Processing video: {input_path}")
        input_frames_dir = outdir / "input_frames"
        input_frames_dir.mkdir(exist_ok=True)

        img_paths = video_to_image_frames(
            str(input_path), str(input_frames_dir), fps=args.fps
        )
        if not img_paths:
            raise RuntimeError("❌ Failed to extract frames from video")

        img_paths = sorted(img_paths)
        print(f"✅ Extracted {len(img_paths)} frames to {input_frames_dir}")

    elif input_path.is_dir():
        # Case 2: Directory of images
        print(f"📁 Processing directory: {input_path}")
        img_paths = []
        for ext in ["*.jpeg", "*.jpg", "*.png", "*.webp"]:
            img_paths.extend(glob.glob(os.path.join(str(input_path), ext)))
        if len(img_paths) == 0:
            raise FileNotFoundError(f"❌ No image files found in {input_path}")
        img_paths = sorted(img_paths)
        print(f"✅ Loaded {len(img_paths)} images from {input_path}")

    else:
        raise ValueError(f"❌ Invalid input path: {input_path}")

    # 3) Load and preprocess images
    views = {}
    imgs = prepare_images_to_tensor(
        img_paths, target_size=args.target_size, resize_strategy="crop"
    ).to(
        device
    )  # [1,S,3,H,W], in [0,1]
    views["img"] = imgs
    B, S, C, H, W = imgs.shape
    cond_flags = [0, 0, 0]
    print(f"📸 Loaded {S} images with shape {imgs.shape}")

    # 4) Inference
    print("\n🚀 Starting inference pipeline...")
    start_time = time.time()
    use_amp = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if use_amp:
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float32
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=bool(use_amp), dtype=amp_dtype):
            predictions = model(
                views=views, cond_flags=cond_flags
            )  # Multi-modal inference with priors
    print(f"🕒 Inference time: {time.time() - start_time:.3f} seconds")

    # 4.5) Sky mask segmentation (if needed)
    sky_mask = None
    if args.apply_sky_mask:
        print("\n🌤️  Computing sky masks...")
        if not os.path.exists("skyseg.onnx"):
            print("Downloading skyseg.onnx...")
            download_file_from_url(
                "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx",
                "skyseg.onnx",
            )
        skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
        sky_mask_list = []
        for i, img_path in enumerate(img_paths):
            sky_mask_frame = segment_sky(img_path, skyseg_session)
            # Resize mask to match H×W if needed
            if sky_mask_frame.shape[0] != H or sky_mask_frame.shape[1] != W:
                sky_mask_frame = cv2.resize(sky_mask_frame, (W, H))
            sky_mask_list.append(sky_mask_frame)
        sky_mask = np.stack(sky_mask_list, axis=0)  # [S, H, W]
        sky_mask = sky_mask > 0  # Binary mask: True = non-sky, False = sky
        print(f"✅ Sky masks computed for {S} frames")
    else:
        # Create dummy sky mask (all True = keep all points)
        sky_mask = np.ones((S, H, W), dtype=bool)

    # 5) Save results
    print("\n📤 Saving results...")
    images_dir = outdir / "images"  # original resolution images
    images_dir.mkdir(exist_ok=True)
    images_resized_dir = outdir / "images_resized"  # resized images
    images_resized_dir.mkdir(exist_ok=True)
    if args.save_depth:
        depth_dir = outdir / "depth"
        depth_dir.mkdir(exist_ok=True)
    if args.save_normal:
        normal_dir = outdir / "normal"
        normal_dir.mkdir(exist_ok=True)
    if args.save_colmap:
        sparse_dir = outdir / "sparse" / "0"
        sparse_dir.mkdir(parents=True, exist_ok=True)

    # save images
    processed_image_names = []
    for i in range(S):
        im = (
            (imgs[0, i].permute(1, 2, 0).clamp(0, 1) * 255)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        fname = f"image_{i + 1:04d}.png"
        Image.fromarray(im).save(str(images_resized_dir / fname))
        pil_img = Image.open(img_paths[i]).convert("RGB")
        processed_height, processed_width = (
            imgs[0, i].shape[1],
            imgs[0, i].shape[2],
        )
        processed_aspect_ratio = processed_width / processed_height
        orig_width, orig_height = pil_img.size
        new_height = int(orig_width / processed_aspect_ratio)
        new_width = orig_width
        pil_img = pil_img.resize(
            (orig_width, new_height), Image.Resampling.BICUBIC
        )
        pil_img.save(str(images_dir / fname))

        processed_image_names.append(fname)

    # save pointmap with filtering
    if "pts3d" in predictions and args.save_pointmap:
        print("Computing filter mask for pointmap...")

        # Prepare data for mask computation
        pts3d_conf_np = (
            predictions["pts3d_conf"][0].detach().cpu().numpy()
        )  # [S, H, W]
        depth_preds_np = (
            predictions["depth"][0].detach().cpu().numpy()
        )  # [S, H, W, 1]
        normal_preds_np = (
            predictions["normals"][0].detach().cpu().numpy()
        )  # [S, H, W, 3]

        # Compute comprehensive filter mask
        final_mask = create_filter_mask(
            pts3d_conf=pts3d_conf_np,
            depth_preds=depth_preds_np,
            normal_preds=normal_preds_np,
            sky_mask=sky_mask,
            confidence_percentile=args.confidence_percentile,
            edge_normal_threshold=args.edge_normal_threshold,
            edge_depth_threshold=args.edge_depth_threshold,
            apply_confidence_mask=args.apply_confidence_mask,
            apply_edge_mask=args.apply_edge_mask,
            apply_sky_mask=args.apply_sky_mask,
        )  # [S, H, W]

        # Collect points and colors
        pts_list = []
        pts_colors_list = []

        for i in range(S):
            pts = predictions["pts3d"][0, i]  # [H,W,3]
            img_colors = imgs[0, i].permute(1, 2, 0)  # [H, W, 3]
            img_colors = (img_colors * 255).to(torch.uint8)

            pts_list.append(pts.reshape(-1, 3))
            pts_colors_list.append(img_colors.reshape(-1, 3))

        all_pts = torch.cat(pts_list, dim=0)
        all_colors = torch.cat(pts_colors_list, dim=0)

        # Apply filter mask
        final_mask_flat = final_mask.reshape(-1)  # Flatten to [S*H*W]
        final_mask_torch = torch.from_numpy(final_mask_flat).to(all_pts.device)

        filtered_pts = all_pts[final_mask_torch]
        filtered_colors = all_colors[final_mask_torch]

        save_scene_ply(
            outdir / "pts_from_pointmap.ply", filtered_pts, filtered_colors
        )
        print(
            f"  - Saved {len(filtered_pts)} filtered points to {outdir / 'pts_from_pointmap.ply'}"
        )

    # save depthmap
    if "depth" in predictions and args.save_depth:
        for i in range(S):
            # Save both PNG (for visualization) and NPY (for actual depth values)
            save_depth_png(
                depth_dir / f"depth_{i:04d}.png",
                predictions["depth"][0, i, :, :, 0],
            )
            save_depth_npy(
                depth_dir / f"depth_{i:04d}.npy",
                predictions["depth"][0, i, :, :, 0],
            )
        print(
            f"  - Saved {S} depth maps to {depth_dir} (both PNG and NPY formats)"
        )

    # save normalmap
    if "normals" in predictions and args.save_normal:
        for i in range(S):
            save_normal_png(
                normal_dir / f"normal_{i:04d}.png", predictions["normals"][0, i]
            )
        print(f"  - Saved {S} normal maps to {normal_dir}")

    # Save Gaussians PLY and render video
    if "splats" in predictions and args.save_gs:
        # Get Gaussian parameters (already filtered by GaussianSplatRenderer)
        means = predictions["splats"]["means"][0].reshape(-1, 3)
        scales = predictions["splats"]["scales"][0].reshape(-1, 3)
        quats = predictions["splats"]["quats"][0].reshape(-1, 4)
        colors = (
            predictions["splats"]["sh"][0]
            if "sh" in predictions["splats"]
            else predictions["splats"]["colors"][0]
        ).reshape(-1, 3)
        opacities = predictions["splats"]["opacities"][0].reshape(-1)

        # Save Gaussian PLY
        ply_path = outdir / "gaussians.ply"
        save_gs_ply(
            ply_path,
            means,
            scales,
            quats,
            colors,
            opacities,
        )

        # Render video using the same filtered splats from predictions
        num_views = S
        if args.save_rendered:
            e4x4 = predictions["camera_poses"]
            k3x3 = predictions["camera_intrs"]
            render_interpolated_video(
                model.gs_renderer,
                predictions["splats"],
                e4x4,
                k3x3,
                (H, W),
                outdir / "rendered",
                interp_per_pair=15,
                loop_reverse=num_views == 1,
            )
            print(f"  - Saved rendered.mp4 to {outdir}")
        else:
            print(f"⚠️  Not set --save_rendered flag, skipping video rendering")

    # Build and export COLMAP reconstruction (images + sparse)
    if args.save_colmap:
        print("Computing filter mask for COLMAP reconstruction...")

        final_width, final_height = new_width, new_height
        print(f"colmap_width: {final_width}, colmap_height: {final_height}")

        # Prepare data for mask computation (reuse from pointmap if not already computed)
        if not ("pts3d" in predictions and args.save_pointmap):
            pts3d_conf_np = (
                predictions["pts3d_conf"][0].detach().cpu().numpy()
            )  # [S, H, W]
            depth_preds_np = (
                predictions["depth"][0].detach().cpu().numpy()
            )  # [S, H, W, 1]
            normal_preds_np = (
                predictions["normals"][0].detach().cpu().numpy()
            )  # [S, H, W, 3]

            # Compute comprehensive filter mask
            final_mask = create_filter_mask(
                pts3d_conf=pts3d_conf_np,
                depth_preds=depth_preds_np,
                normal_preds=normal_preds_np,
                sky_mask=sky_mask,
                confidence_percentile=args.confidence_percentile,
                edge_normal_threshold=args.edge_normal_threshold,
                edge_depth_threshold=args.edge_depth_threshold,
                apply_confidence_mask=args.apply_confidence_mask,
                apply_edge_mask=args.apply_edge_mask,
                apply_sky_mask=args.apply_sky_mask,
            )  # [S, H, W]

        # Prepare extrinsics/intrinsics (camera-from-world) using resized image size
        e3x4, intr = vector_to_camera_matrices(
            predictions["camera_params"], image_hw=(final_height, final_width)
        )
        _, intr_resize = vector_to_camera_matrices(
            predictions["camera_params"], image_hw=(H, W)
        )
        extrinsics = e3x4[0]  # [S,3,4]
        intrinsics = intr[0]  # [S,3,3]
        intrinsics_resize = intr_resize[0]  # [S,3,3]

        points_list = []
        colors_list = []
        xyf_list = []

        # Precompute pixel coordinate grid (XYF) like demo_colmap
        xyf_grid = create_pixel_coordinate_grid(
            num_frames=S, height=H, width=W
        )  # [S,H,W,3] float32
        xyf_grid = xyf_grid.astype(np.int32)

        # Calculate scaling factors to map from processed to resized coordinates
        scale_x = final_width / W
        scale_y = final_height / H

        # Use the SAME coordinate transformation as GaussianSplatRenderer.prepare_splats
        # to ensure consistency between Gaussian PLY and depth-based sparse points
        for i in range(S):
            d = predictions["depth"][0, i, :, :, 0]
            w2c = extrinsics[i][:3, :4]  # [3, 4] camera-to-world
            w2c = torch.cat(
                [w2c, torch.tensor([[0, 0, 0, 1]], device=w2c.device)], dim=0
            )  # [4,4]
            c2w = torch.linalg.inv(w2c)[:3, :4]  # [4,4]
            K = intrinsics_resize[i]
            pts_i, _, mask = depth_to_world_coords_points(
                d[None], c2w[None], K[None]
            )

            img_colors = (imgs[0, i].permute(1, 2, 0) * 255).to(torch.uint8)

            # Apply filter mask from mask computation
            filter_mask_frame = torch.from_numpy(final_mask[i]).to(
                mask.device
            )  # [H, W]
            valid = (
                mask[0] & filter_mask_frame
            )  # Combine depth validity with filter mask

            if valid.sum().item() == 0:
                continue
            xyf_np = xyf_grid[i][valid.cpu().numpy()]  # [N,3] int32
            xyf_list.append(torch.from_numpy(xyf_np).to(valid.device))
            points_list.append(pts_i[0][valid])
            colors_list.append(img_colors[valid])

        all_pts = torch.cat(points_list, dim=0)
        all_cols = torch.cat(colors_list, dim=0)
        all_xyf = torch.cat(xyf_list, dim=0)

        # Convert to numpy
        extrinsics = extrinsics.detach().cpu().numpy()
        intrinsics = intrinsics.detach().cpu().numpy()
        f_pts = all_pts.detach().cpu().to(torch.float32).numpy()
        f_cols = all_cols.detach().cpu().to(torch.uint8).numpy()
        f_xyf = all_xyf.detach().cpu().to(torch.int32).numpy()

        # Scale 2D coordinates from processed image to resized image resolution (if still valid)
        f_xyf[:, 0] = (f_xyf[:, 0] * scale_x).astype(np.int32)  # x coordinates
        f_xyf[:, 1] = (f_xyf[:, 1] * scale_y).astype(np.int32)  # y coordinates

        # Build reconstruction using pycolmap (PINHOLE) with resized image size
        # Standard COLMAP reconstruction with 2D-3D correspondences
        image_size = np.array([final_width, final_height])
        reconstruction = build_pycolmap_reconstruction(
            points=f_pts,
            pixel_coords=f_xyf,
            point_colors=f_cols,
            poses=extrinsics,
            intrinsics=intrinsics,
            image_size=image_size,
            shared_camera_model=False,
            camera_model="SIMPLE_PINHOLE",
        )

        # Update image names to match saved files
        for pyimageid in reconstruction.images:
            reconstruction.images[pyimageid].name = processed_image_names[
                pyimageid - 1
            ]

        # Write BIN
        reconstruction.write(str(sparse_dir))
        # Save points3D.ply
        save_points_ply(sparse_dir / "points3D.ply", f_pts, f_cols)

        print(f"  - Saved COLMAP BIN and points3D.ply to {sparse_dir}")


if __name__ == "__main__":
    main()
