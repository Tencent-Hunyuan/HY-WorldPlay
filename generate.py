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

from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline
from hyvideo.commons.parallel_states import initialize_parallel_state
from hyvideo.commons.infer_state import initialize_infer_state

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

def pose_to_input(pose_json_path, latent_chunk_num, tps=False):
    pose_json = json.load(open(pose_json_path, 'r'))
    pose_keys = list(pose_json.keys())
    intrinsic_list = []
    w2c_list = []
    for i in range(latent_chunk_num):
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

    return torch.tensor(w2c_list), torch.tensor(intrinsic_list), action_one_label

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

    viewmats, Ks, action = pose_to_input(args.pose_json_path, (args.video_length - 1) // 4 + 1)

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
        if enable_sr and hasattr(out, 'sr_videos'):
            save_video(out.sr_videos, save_video_sr_path)
            print(f"Saved SR video to: {save_video_sr_path}")

            if args.save_pre_sr_video:
                save_video(out.videos, save_video_path)
                print(f"Saved original video (before SR) to: {save_video_path}")
        else:
            save_video(out.videos, save_video_path)
            print(f"Saved video to: {save_video_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate video using HunyuanWorld-1.5')

    parser.add_argument("--pose_json_path", type=str,
                        default="./assets/pose/test_forward_32_latents.json",
                        help="Path to the action pose file")
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

    args = parser.parse_args()
    
    assert args.image_path is not None

    generate_video(args)


if __name__ == '__main__':
    main()
