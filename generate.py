# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

import os

if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import loguru
import torch
import argparse
import einops
import imageio
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, VideoClip

from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline
from hyvideo.commons.parallel_states import initialize_parallel_state
from hyvideo.commons.infer_state import initialize_infer_state
from generate_custom_trajectory import generate_camera_trajectory_local

parallel_dims = initialize_parallel_state(sp=int(os.environ.get('WORLD_SIZE', '1')))
torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))

mapping = {
            (0,0,0,0): 0,
            (1,0,0,0): 1,
            (0,1,0,0): 2,
            (0,0,1,0): 3,
            (0,0,0,1): 4,
            (1,0,1,0): 5,
            (1,0,0,1): 6,
            (0,1,1,0): 7,
            (0,1,0,1): 8,
        }

def one_hot_to_one_dimension(one_hot):
    y = torch.tensor([mapping[tuple(row.tolist())] for row in one_hot])
    return y

def parse_pose_string(pose_string):
    """
    Parse pose string to motions list.
    Format: "w-3, right-0.5, d-4"
    - w: forward movement
    - s: backward movement
    - a: left movement
    - d: right movement
    - up: pitch up rotation
    - down: pitch down rotation
    - left: yaw left rotation
    - right: yaw right rotation
    - number after dash: duration in latents

    Args:
        pose_string: str, comma-separated pose commands

    Returns:
        list of dict: motions for generate_camera_trajectory_local
    """
    # Movement amount per frame
    forward_speed = 0.08  # units per frame
    yaw_speed = np.deg2rad(3)  # radians per frame
    pitch_speed = np.deg2rad(3)  # radians per frame

    motions = []
    commands = [cmd.strip() for cmd in pose_string.split(',')]

    for cmd in commands:
        if not cmd:
            continue

        parts = cmd.split('-')
        if len(parts) != 2:
            raise ValueError(f"Invalid pose command: {cmd}. Expected format: 'action-duration'")

        action = parts[0].strip()
        try:
            duration = float(parts[1].strip())
        except ValueError:
            raise ValueError(f"Invalid duration in command: {cmd}")

        num_frames = int(duration)

        # Parse action and create motion dict
        if action == 'w':
            # Forward
            for _ in range(num_frames):
                motions.append({"forward": forward_speed})
        elif action == 's':
            # Backward
            for _ in range(num_frames):
                motions.append({"forward": -forward_speed})
        elif action == 'a':
            # Left
            for _ in range(num_frames):
                motions.append({"right": -forward_speed})
        elif action == 'd':
            # Right
            for _ in range(num_frames):
                motions.append({"right": forward_speed})
        elif action == 'up':
            # Pitch up
            for _ in range(num_frames):
                motions.append({"pitch": pitch_speed})
        elif action == 'down':
            # Pitch down
            for _ in range(num_frames):
                motions.append({"pitch": -pitch_speed})
        elif action == 'left':
            # Yaw left
            for _ in range(num_frames):
                motions.append({"yaw": -yaw_speed})
        elif action == 'right':
            # Yaw right
            for _ in range(num_frames):
                motions.append({"yaw": yaw_speed})
        else:
            raise ValueError(f"Unknown action: {action}. Supported actions: w, s, a, d, up, down, left, right")

    return motions

def pose_string_to_json(pose_string):
    """
    Convert pose string to pose JSON format.

    Args:
        pose_string: str, comma-separated pose commands

    Returns:
        dict: pose JSON with extrinsic and intrinsic parameters
    """
    motions = parse_pose_string(pose_string)
    poses = generate_camera_trajectory_local(motions)

    # Default intrinsic matrix (from generate_custom_trajectory.py)
    intrinsic = [
        [969.6969696969696, 0.0, 960.0],
        [0.0, 969.6969696969696, 540.0],
        [0.0, 0.0, 1.0]
    ]

    pose_json = {}
    for i, p in enumerate(poses):
        pose_json[str(i)] = {"extrinsic": p.tolist(), "K": intrinsic}

    return pose_json

def pose_to_input(pose_data, latent_num, tps=False):
    """
    Convert pose data to input tensors.

    Args:
        pose_data: str or dict
            - If str ending with '.json': path to JSON file
            - If str: pose string (e.g., "w-3, right-0.5, d-4")
            - If dict: pose JSON data
        latent_num: int, number of latents
        tps: bool, third person mode

    Returns:
        tuple: (w2c_list, intrinsic_list, action_one_label)
    """
    # Handle different input types
    if isinstance(pose_data, str):
        if pose_data.endswith('.json'):
            # Load from JSON file
            pose_json = json.load(open(pose_data, 'r'))
        else:
            # Parse pose string
            pose_json = pose_string_to_json(pose_data)
    elif isinstance(pose_data, dict):
        pose_json = pose_data
    else:
        raise ValueError(f"Invalid pose_data type: {type(pose_data)}. Expected str or dict.")

    pose_keys = list(pose_json.keys())
    latent_num_from_pose = len(pose_keys)
    assert latent_num_from_pose == latent_num, (f"pose corresponds to {latent_num_from_pose * 4 - 3} frames, num_frames "
                                                f"must be set to {latent_num_from_pose * 4 - 3} to ensure alignment.")

    intrinsic_list = []
    w2c_list = []
    for i in range(latent_num):
        t_key = pose_keys[i]
        c2w = np.array(pose_json[t_key]["extrinsic"])
        w2c = np.linalg.inv(c2w)
        w2c_list.append(w2c)
        intrinsic = np.array(pose_json[t_key]["K"])
        intrinsic[0, 0] /= intrinsic[0, 2] * 2
        intrinsic[1, 1] /= intrinsic[1, 2] * 2
        intrinsic[0, 2] = 0.5
        intrinsic[1, 2] = 0.5
        intrinsic_list.append(intrinsic)

    w2c_list = np.array(w2c_list)
    intrinsic_list = torch.tensor(np.array(intrinsic_list))

    c2ws = np.linalg.inv(w2c_list)
    C_inv = np.linalg.inv(c2ws[:-1])
    relative_c2w = np.zeros_like(c2ws)
    relative_c2w[0, ...] = c2ws[0, ...]
    relative_c2w[1:, ...] = C_inv @ c2ws[1:, ...]
    trans_one_hot = np.zeros((relative_c2w.shape[0], 4), dtype=np.int32)
    rotate_one_hot = np.zeros((relative_c2w.shape[0], 4), dtype=np.int32)

    move_norm_valid = 0.0001
    for i in range(1, relative_c2w.shape[0]):
        move_dirs = relative_c2w[i, :3, 3]  # direction vector
        move_norms = np.linalg.norm(move_dirs)
        if move_norms > move_norm_valid:  # threshold for movement
            move_norm_dirs = move_dirs / move_norms
            angles_rad = np.arccos(move_norm_dirs.clip(-1.0, 1.0))  
            trans_angles_deg = angles_rad * (180.0 / torch.pi)  # convert to degrees
        else:
            trans_angles_deg = torch.zeros(3)

        R_rel = relative_c2w[i, :3, :3]
        r = R.from_matrix(R_rel)
        rot_angles_deg = r.as_euler('xyz', degrees=True)

        # Determine movement and rotation actions
        if move_norms > move_norm_valid:  # threshold for movement
            if (not tps) or (tps == True and abs(rot_angles_deg[1]) < 5e-2 and abs(rot_angles_deg[0]) < 5e-2):
                if trans_angles_deg[2] < 60:
                    trans_one_hot[i, 0] = 1  # forward
                elif trans_angles_deg[2] > 120:
                    trans_one_hot[i, 1] = 1  # backward

                if trans_angles_deg[0] < 60:
                    trans_one_hot[i, 2] = 1  # right
                elif trans_angles_deg[0] > 120:
                    trans_one_hot[i, 3] = 1  # left

        if rot_angles_deg[1] > 5e-2:
            rotate_one_hot[i, 0] = 1  # right
        elif rot_angles_deg[1] < -5e-2:
            rotate_one_hot[i, 1] = 1  # left

        if rot_angles_deg[0] > 5e-2:
            rotate_one_hot[i, 2] = 1  # up
        elif rot_angles_deg[0] < -5e-2:
            rotate_one_hot[i, 3] = 1  # down
    trans_one_hot = torch.tensor(trans_one_hot)
    rotate_one_hot = torch.tensor(rotate_one_hot)

    trans_one_label = one_hot_to_one_dimension(trans_one_hot)
    rotate_one_label = one_hot_to_one_dimension(rotate_one_hot)
    action_one_label = trans_one_label * 9 + rotate_one_label

    return torch.as_tensor(w2c_list), torch.as_tensor(intrinsic_list), action_one_label

def save_video(video, path):
    if video.ndim == 5:
        assert video.shape[0] == 1
        video = video[0]
    vid = (video * 255).clamp(0, 255).to(torch.uint8)
    vid = einops.rearrange(vid, 'c f h w -> f h w c')
    imageio.mimwrite(path, vid, fps=24)

def rank0_log(message, level):
    if int(os.environ.get('RANK', '0')) == 0:
        loguru.logger.log(level, message)

def str_to_bool(value):
    """Convert string to boolean, supporting true/false, 1/0, yes/no.
    If value is None (when flag is provided without value), returns True."""
    if value is None:
        return True  # When --flag is provided without value, enable it
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.lower().strip()
        if value in ('true', '1', 'yes', 'on'):
            return True
        elif value in ('false', '0', 'no', 'off'):
            return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")

def camera_center_normalization(w2c):
    c2w = np.linalg.inv(w2c)
    C0_inv = np.linalg.inv(c2w[0])
    c2w_aligned = np.array([C0_inv @ C for C in c2w])
    return np.linalg.inv(c2w_aligned)

def parse_pose_string_to_actions(pose_string, fps=6):
    """
    Parse pose string to action timeline with durations.

    Args:
        pose_string: str, comma-separated pose commands (e.g., "w-3, right-0.5, d-4")
        fps: int, frames per second

    Returns:
        list of dict: Each dict contains {"second": int, "actions": {"forward": 0/1/-1, "left": 0/1/-1, "yaw": 0/1/-1, "pitch": 0/1/-1}}
    """
    actions_timeline = []
    commands = [cmd.strip() for cmd in pose_string.split(',')]

    current_second = 0

    for cmd in commands:
        if not cmd:
            continue

        parts = cmd.split('-')
        if len(parts) != 2:
            raise ValueError(f"Invalid pose command: {cmd}. Expected format: 'action-duration'")

        action = parts[0].strip()
        try:
            duration = float(parts[1].strip())
        except ValueError:
            raise ValueError(f"Invalid duration in command: {cmd}")

        num_seconds = int(np.ceil(duration))

        # Map action to action values
        action_values = {"forward": 0, "left": 0, "yaw": 0, "pitch": 0}

        if action == 'w':
            action_values["forward"] = 1
        elif action == 's':
            action_values["forward"] = -1
        elif action == 'a':
            action_values["left"] = 1
        elif action == 'd':
            action_values["left"] = -1
        elif action == 'up':
            action_values["pitch"] = 1
        elif action == 'down':
            action_values["pitch"] = -1
        elif action == 'left':
            action_values["yaw"] = -1
        elif action == 'right':
            action_values["yaw"] = 1
        else:
            raise ValueError(f"Unknown action: {action}")

        # Add this action for its duration
        for _ in range(num_seconds):
            actions_timeline.append({
                "second": current_second,
                "actions": action_values.copy()
            })
            current_second += 1

    return actions_timeline

def draw_rounded_rectangle(draw, xy, radius, fill=None, outline=None, width=1):
    """Draw a rounded rectangle."""
    x1, y1, x2, y2 = xy
    diameter = radius * 2

    # Draw four corners (circles)
    draw.ellipse([x1, y1, x1 + diameter, y1 + diameter], fill=fill, outline=outline, width=width)
    draw.ellipse([x2 - diameter, y1, x2, y1 + diameter], fill=fill, outline=outline, width=width)
    draw.ellipse([x1, y2 - diameter, x1 + diameter, y2], fill=fill, outline=outline, width=width)
    draw.ellipse([x2 - diameter, y2 - diameter, x2, y2], fill=fill, outline=outline, width=width)

    # Draw two rectangles to fill the middle
    draw.rectangle([x1 + radius, y1, x2 - radius, y2], fill=fill)
    draw.rectangle([x1, y1 + radius, x2, y2 - radius], fill=fill)

    # Draw border if outline is specified
    if outline:
        # Top and bottom lines
        draw.line([x1 + radius, y1, x2 - radius, y1], fill=outline, width=width)
        draw.line([x1 + radius, y2, x2 - radius, y2], fill=outline, width=width)
        # Left and right lines
        draw.line([x1, y1 + radius, x1, y2 - radius], fill=outline, width=width)
        draw.line([x2, y1 + radius, x2, y2 - radius], fill=outline, width=width)

def create_wasd_keyboard(actions, key_size=70, key_spacing=6, corner_radius=14):
    """Create WASD keyboard overlay."""
    keyboard_width = 3 * key_size + 2 * key_spacing
    keyboard_height = 2 * key_size + key_spacing

    img = Image.new('RGBA', (keyboard_width, keyboard_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    bg_normal = (0, 0, 0, 128)
    bg_active = (30, 120, 255, 220)
    text_color = (255, 255, 255, 255)
    font_size = 28

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", font_size)
        except (IOError, OSError):
            font = ImageFont.load_default()

    def draw_key(x, y, label, is_active):
        bg_color = bg_active if is_active else bg_normal
        draw_rounded_rectangle(draw, [x, y, x + key_size, y + key_size], radius=corner_radius, fill=bg_color)
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = x + (key_size - text_width) // 2
        text_y = y + (key_size - text_height) // 2
        draw.text((text_x, text_y), label, fill=text_color, font=font)

    forward_val = actions.get("forward", 0)
    left_val = actions.get("left", 0)

    w_active = forward_val > 0
    s_active = forward_val < 0
    a_active = left_val > 0
    d_active = left_val < 0

    wasd_keys = [
        ("W", 1, 0, w_active),
        ("A", 0, 1, a_active),
        ("S", 1, 1, s_active),
        ("D", 2, 1, d_active),
    ]

    for label, col, row, is_active in wasd_keys:
        x = col * (key_size + key_spacing)
        y = row * (key_size + key_spacing)
        draw_key(x, y, label, is_active)

    return img

def create_arrow_keyboard(actions, key_size=70, key_spacing=6, corner_radius=14):
    """Create arrow keys keyboard overlay with triangle symbols."""
    keyboard_width = 3 * key_size + 2 * key_spacing
    keyboard_height = 2 * key_size + key_spacing

    img = Image.new('RGBA', (keyboard_width, keyboard_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    bg_normal = (0, 0, 0, 128)
    bg_active = (30, 120, 255, 220)
    text_color = (255, 255, 255, 255)

    def draw_key_with_triangle(x, y, direction, is_active):
        bg_color = bg_active if is_active else bg_normal
        draw_rounded_rectangle(draw, [x, y, x + key_size, y + key_size], radius=corner_radius, fill=bg_color)
        cx = x + key_size // 2
        cy = y + key_size // 2
        size = key_size // 8
        if direction == "up":
            points = [(cx, cy - size), (cx - size, cy + size // 2), (cx + size, cy + size // 2)]
        elif direction == "down":
            points = [(cx, cy + size), (cx - size, cy - size // 2), (cx + size, cy - size // 2)]
        elif direction == "left":
            points = [(cx - size, cy), (cx + size // 2, cy - size), (cx + size // 2, cy + size)]
        elif direction == "right":
            points = [(cx + size, cy), (cx - size // 2, cy - size), (cx - size // 2, cy + size)]
        draw.polygon(points, fill=text_color)

    yaw_val = actions.get("yaw", 0)
    pitch_val = actions.get("pitch", 0)

    up_active = pitch_val > 0
    down_active = pitch_val < 0
    left_active = yaw_val < 0
    right_active = yaw_val > 0

    arrow_keys = [
        ("up", 1, 0, up_active),
        ("left", 0, 1, left_active),
        ("down", 1, 1, down_active),
        ("right", 2, 1, right_active),
    ]

    for direction, col, row, is_active in arrow_keys:
        kx = col * (key_size + key_spacing)
        ky = row * (key_size + key_spacing)
        draw_key_with_triangle(kx, ky, direction, is_active)

    return img

def blend_overlay(base_frame, overlay, position):
    """Blend an RGBA overlay onto a RGB frame."""
    x, y = position
    overlay_array = np.array(overlay)

    oh, ow = overlay_array.shape[:2]
    bh, bw = base_frame.shape[:2]

    x = max(0, min(x, bw - 1))
    y = max(0, min(y, bh - 1))

    if x + ow > bw:
        ow = bw - x
        overlay_array = overlay_array[:, :ow]
    if y + oh > bh:
        oh = bh - y
        overlay_array = overlay_array[:oh, :]

    if ow <= 0 or oh <= 0:
        return base_frame

    overlay_rgb = overlay_array[:, :, :3].astype(np.float32)
    overlay_alpha = overlay_array[:, :, 3:4].astype(np.float32) / 255.0

    base_region = base_frame[y:y+oh, x:x+ow].astype(np.float32)

    blended = (overlay_rgb * overlay_alpha + base_region * (1 - overlay_alpha)).astype(np.uint8)

    result = base_frame.copy()
    result[y:y+oh, x:x+ow] = blended

    return result

def add_keyboard_overlay_to_video(video_path, output_path, actions_timeline):
    """Add keyboard overlay to an existing video."""
    try:
        video = VideoFileClip(video_path)
        width, height = video.size
        fps = video.fps
        duration = video.duration

        key_size = 70
        key_spacing = 6
        corner_radius = 14
        margin = 40

        keyboard_width = 3 * key_size + 2 * key_spacing
        keyboard_height = 2 * key_size + key_spacing

        wasd_pos = (margin, height - margin - keyboard_height)
        arrow_pos = (width - margin - keyboard_width, height - margin - keyboard_height)

        wasd_cache = {}
        arrow_cache = {}

        def get_actions_at_time(t):
            second_idx = int(t)
            if 0 <= second_idx < len(actions_timeline):
                return actions_timeline[second_idx]["actions"]
            return {"forward": 0, "left": 0, "yaw": 0, "pitch": 0}

        def make_frame(t):
            frame = video.get_frame(t)
            actions = get_actions_at_time(t)

            def sign(x):
                if x > 0:
                    return 1
                elif x < 0:
                    return -1
                return 0

            wasd_key = (sign(actions.get("forward", 0)), sign(actions.get("left", 0)))
            arrow_key = (sign(actions.get("yaw", 0)), sign(actions.get("pitch", 0)))

            if wasd_key not in wasd_cache:
                wasd_cache[wasd_key] = create_wasd_keyboard(actions, key_size, key_spacing, corner_radius)
            wasd_img = wasd_cache[wasd_key]

            if arrow_key not in arrow_cache:
                arrow_cache[arrow_key] = create_arrow_keyboard(actions, key_size, key_spacing, corner_radius)
            arrow_img = arrow_cache[arrow_key]

            frame = blend_overlay(frame, wasd_img, wasd_pos)
            frame = blend_overlay(frame, arrow_img, arrow_pos)

            return frame

        output_clip = VideoClip(make_frame, duration=duration)
        output_clip = output_clip.set_fps(fps)

        if video.audio is not None:
            output_clip = output_clip.set_audio(video.audio)

        output_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            logger=None
        )

        video.close()
        output_clip.close()

        return True

    except Exception as e:
        import traceback
        print(f"Error adding keyboard overlay: {e}")
        traceback.print_exc()
        return False

def generate_video(args):
    assert ((args.video_length - 1) // 4 + 1) % 4 == 0, "number of latents must be divisible by 4"
    initialize_infer_state(args)

    task = 'i2v' if args.image_path else 't2v'
    
    enable_sr = args.sr
    
    # Build transformer_version based on flags
    transformer_version = f'{args.resolution}_{task}'
    assert transformer_version == "480p_i2v"

    if args.dtype == 'bf16':
        transformer_dtype = torch.bfloat16
    elif args.dtype == 'fp32':
        transformer_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}. Must be 'bf16' or 'fp32'")
    
    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=args.model_path,
        transformer_version=transformer_version,
        enable_offloading=args.offloading,
        enable_group_offloading=args.group_offloading,
        create_sr_pipeline=enable_sr,
        force_sparse_attn=False,
        transformer_dtype=transformer_dtype,
        action_ckpt=args.action_ckpt,
    )

    extra_kwargs = {}
    if task == 'i2v':
        extra_kwargs['reference_image'] = args.image_path

    enable_rewrite = args.rewrite
    if not args.rewrite:
        rank0_log("Warning: Prompt rewriting is disabled. This may affect the quality of generated videos.", "WARNING")

    viewmats, Ks, action = pose_to_input(args.pose, (args.video_length - 1) // 4 + 1)

    if task == 'i2v':
        extra_kwargs['reference_image'] = args.image_path

    out = pipe(
        enable_sr=enable_sr,
        prompt=args.prompt,
        aspect_ratio=args.aspect_ratio,
        num_inference_steps=args.num_inference_steps,
        sr_num_inference_steps=None,
        video_length=args.video_length,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        output_type="pt",
        prompt_rewrite=enable_rewrite,
        return_pre_sr_video=args.save_pre_sr_video,
        viewmats=viewmats.unsqueeze(0),
        Ks=Ks.unsqueeze(0),
        action=action.unsqueeze(0),
        few_step=args.few_step,
        chunk_latent_frames=4 if args.model_type == "ar" else 16,
        model_type=args.model_type,
        user_height=args.height,
        user_width=args.width,
        **extra_kwargs,
    )

    # save video
    if int(os.environ.get('RANK', '0')) == 0:
        output_path = args.output_path
        os.makedirs(output_path, exist_ok=True)

        save_video_path = os.path.join(output_path, "gen.mp4")
        save_video_sr_path = os.path.join(output_path, "gen_sr.mp4")

        # Determine which video to process for UI overlay
        video_to_process = None
        final_video_path = None

        if enable_sr and hasattr(out, 'sr_videos'):
            save_video(out.sr_videos, save_video_sr_path)
            print(f"Saved SR video to: {save_video_sr_path}")
            video_to_process = save_video_sr_path
            final_video_path = save_video_sr_path

            if args.save_pre_sr_video:
                save_video(out.videos, save_video_path)
                print(f"Saved original video (before SR) to: {save_video_path}")
        else:
            save_video(out.videos, save_video_path)
            print(f"Saved video to: {save_video_path}")
            video_to_process = save_video_path
            final_video_path = save_video_path

        # Add keyboard overlay if --with-ui is enabled and pose is a string
        if args.with_ui and isinstance(args.pose, str) and not args.pose.endswith('.json'):
            print(f"Adding keyboard overlay to video...")
            try:
                actions_timeline = parse_pose_string_to_actions(args.pose)

                # Create temporary output path for video with UI
                video_with_ui_path = os.path.join(output_path, "gen_with_ui_temp.mp4")

                if add_keyboard_overlay_to_video(video_to_process, video_with_ui_path, actions_timeline):
                    # Replace original video with UI version
                    os.replace(video_with_ui_path, final_video_path)
                    print(f"Successfully added keyboard overlay to: {final_video_path}")
                else:
                    print(f"Failed to add keyboard overlay, keeping original video")
                    if os.path.exists(video_with_ui_path):
                        os.remove(video_with_ui_path)
            except Exception as e:
                print(f"Error processing keyboard overlay: {e}")
                import traceback
                traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Generate video using HunyuanWorld-1.5')

    parser.add_argument("--pose", type=str,
                        default="./assets/pose/test_forward_32_latents.json",
                        help="Path to pose JSON file or pose string (e.g., 'w-3, right-0.5, d-4')")
    parser.add_argument(
        '--prompt', type=str, required=True,
        help='Text prompt for video generation'
    )
    parser.add_argument(
        '--negative_prompt', type=str, default='',
        help='Negative prompt for video generation (default: empty string)'
    )
    parser.add_argument(
        '--resolution', type=str, required=True, choices=['480p', '720p'],
        help='Video resolution (480p or 720p)'
    )
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='Path to pretrained model'
    )
    parser.add_argument(
        '--action_ckpt', type=str, required=True,
        help='Path to pretrained action model'
    )
    parser.add_argument(
        '--aspect_ratio', type=str, default='16:9',
        help='Aspect ratio (default: 16:9)'
    )
    parser.add_argument(
        '--num_inference_steps', type=int, default=50,
        help='Number of inference steps (default: 50)'
    )
    parser.add_argument(
        '--video_length', type=int, default=127,
        help='Number of frames to generate (default: 127)'
    )
    parser.add_argument(
        '--sr', type=str_to_bool, nargs='?', const=True, default=True,
        help='Enable super resolution (default: true). '
             'Use --sr or --sr true/1 to enable, --sr false/0 to disable'
    )
    parser.add_argument(
        '--save_pre_sr_video', type=str_to_bool, nargs='?', const=True, default=False,
        help='Save original video before super resolution (default: false). '
             'Use --save_pre_sr_video or --save_pre_sr_video true/1 to enable, '
             '--save_pre_sr_video false/0 to disable'
    )
    parser.add_argument(
        '--rewrite', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable prompt rewriting (default: true). '
             'Use --rewrite or --rewrite true/1 to enable, --rewrite false/0 to disable'
    )
    parser.add_argument(
        '--offloading', type=str_to_bool, nargs='?', const=True, default=True,
        help='Enable offloading (default: true). '
             'Use --offloading or --offloading true/1 to enable, '
             '--offloading false/0 to disable'
    )
    parser.add_argument(
        '--group_offloading', type=str_to_bool, nargs='?', const=True, default=None,
        help='Enable group offloading (default: None, automatically enabled if offloading is enabled). '
             'Use --group_offloading or --group_offloading true/1 to enable, '
             '--group_offloading false/0 to disable'
    )
    parser.add_argument(
        '--dtype', type=str, default='bf16', choices=['bf16', 'fp32'],
        help='Data type for transformer (default: bf16). '
             'bf16: faster, lower memory; fp32: better quality, slower, higher memory'
    )
    parser.add_argument(
        '--seed', type=int, default=123,
        help='Random seed (default: 123)'
    )
    parser.add_argument(
        '--image_path', type=str, default=None,
        help='Path to reference image for i2v (if provided, uses i2v mode)'
    )
    parser.add_argument(
        '--output_path', type=str, default=None,
        help='Output file path for generated video (if not provided, saves to ./outputs/output.mp4)'
    )
    parser.add_argument(
        '--enable_torch_compile', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable torch compile for transformer (default: false). '
             'Use --enable_torch_compile or --enable_torch_compile true/1 to enable, '
             '--enable_torch_compile false/0 to disable'
    )
    parser.add_argument(
        '--few_step', type=str_to_bool, nargs='?', const=False, default=False,
        help='Enable super resolution (default: true). '
             'Use --few_step or --few_step true/1 to enable, --few_step false/0 to disable'
    )
    parser.add_argument(
        '--model_type', type=str, required=True, choices=['bi', 'ar'],
        help='inference bidirectional or autoregressive model. '
    )
    parser.add_argument(
        '--height', type=int, default=None,
        help='height for generation (recommended to set as 480)'
    )
    parser.add_argument(
        '--width', type=int, default=None,
        help='width for generation (recommended to set as 832)'
    )
    parser.add_argument(
        '--with-ui', type=str_to_bool, nargs='?', const=True, default=False,
        help='Add keyboard overlay to generated video (default: false). '
             'Only works with pose string input, not JSON files. '
             'Use --with-ui or --with-ui true/1 to enable, --with-ui false/0 to disable'
    )

    parser.add_argument(
        '--use_sageattn', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable sageattn (default: false). '
             'Use --use_sageattn or --use_sageattn true/1 to enable, '
             '--use_sageattn false/0 to disable'
    )
    parser.add_argument(
        '--sage_blocks_range', type=str, default="0-53",
        help='Sageattn blocks range (e.g., 0-5 or 0,1,2,3,4,5)'
    )
    parser.add_argument(
        '--use_vae_parallel', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable vae parallel (default: false). '
             'Use --use_vae_parallel or --use_vae_parallel true/1 to enable, '
             '--use_vae_parallel false/0 to disable'
    )
    # fp8 gemm related
    parser.add_argument(
        '--use_fp8_gemm', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable fp8 gemm for transformer (default: false). '
             'Use --use_fp8_gemm or --use_fp8_gemm true/1 to enable, '
             '--use_fp8_gemm false/0 to disable'
    )
    parser.add_argument(
        '--quant_type', type=str, default="fp8-per-block",
        help='Quantization type for fp8 gemm (e.g., fp8-per-tensor-weight-only, fp8-per-tensor, fp8-per-block)'
    )
    parser.add_argument(
        '--include_patterns', type=str, default="double_blocks",
        help='Include patterns for fp8 gemm (default: double_blocks)'
    )

    args = parser.parse_args()
    
    assert args.image_path is not None

    generate_video(args)


if __name__ == '__main__':
    main()
