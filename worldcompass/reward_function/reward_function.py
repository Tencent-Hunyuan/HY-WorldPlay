import torch
import cv2
import numpy as np
import os
import uuid
import tempfile
import logging
from PIL import Image
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from hpsv3 import HPSv3RewardInferencer

from depth_anything_3.api import DepthAnything3
DEPTH_ANYTHING_3_AVAILABLE = True

from reward_function.HunyuanWorldMirror import WorldMirror
from reward_function.HunyuanWorldMirror.src.utils.inference_utils import (
    extract_load_and_preprocess_images,
)

logger = logging.getLogger(__name__)


class CompassReward:
    """视频一致性奖励计算器."""

    def __init__(self, device=None, camera_estimator="dav3", cache_dir=None):
        """初始化视频一致性奖励计算器.

        Args:
            device: 计算设备，默认自动选择
            camera_estimator: 相机估计模型选择，可选 "dav3" 或 "worldmirror"
            cache_dir: 模型缓存目录，默认使用HuggingFace默认缓存
        """
        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = (
            torch.bfloat16
            if torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )
        self.camera_estimator = camera_estimator.lower()

        # 初始化相机估计模型
        if self.camera_estimator == "worldmirror":
            worldmirror_model_path = "tencent/HunyuanWorld-Mirror"
            logger.info(
                f"Loading WorldMirror model from: {worldmirror_model_path}"
            )
            self.worldmirror_model = WorldMirror.from_pretrained(
                worldmirror_model_path, cache_dir=cache_dir
            ).to(self.device)
            self.worldmirror_model.eval()
            self.worldmirror_model.enable_gs = False
            self.dav3_model = None
        elif self.camera_estimator == "dav3":
            if not DEPTH_ANYTHING_3_AVAILABLE:
                raise ImportError(
                    "DepthAnything3 is not available. Please install the depth_anything_3 package "
                    "or use --camera-estimator worldmirror instead. "
                    "To install depth_anything_3, follow the instructions in the project documentation."
                )
            da3_model_path = "depth-anything/DA3-GIANT-1.1"
            logger.info(f"Loading DepthAnything3 model from: {da3_model_path}")
            self.dav3_model = DepthAnything3.from_pretrained(
                da3_model_path, cache_dir=cache_dir
            ).to(self.device)
            self.worldmirror_model = None
        else:
            raise ValueError(
                f"Unsupported camera_estimator: {camera_estimator}. Choose 'dav3' or 'worldmirror'"
            )

        self.hpsv3_model = HPSv3RewardInferencer(device=self.device)

        # 初始化RoMa模型
        # 为每个实例创建唯一的临时目录
        self.temp_dir = tempfile.mkdtemp(prefix="video_consistency_")
        self.instance_id = str(uuid.uuid4())[:8]  # 短的唯一标识符

    def __del__(self):
        """析构函数，清理临时目录."""
        self.cleanup()

    def cleanup(self):
        """清理临时目录和文件."""
        import shutil

        if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"已清理临时目录: {self.temp_dir}")

    def __enter__(self):
        """上下文管理器入口."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，自动清理."""
        self.cleanup()

    def concat_images_vertically(self, list1, list2, video_path):
        concat_paths = []
        for idx, (img_path1, img_path2) in enumerate(zip(list1, list2)):
            img1 = Image.open(img_path1)
            img2 = Image.open(img_path2)
            w = max(img1.width, img2.width)
            h = img1.height + img2.height
            new_img = Image.new("RGB", (w, h))
            new_img.paste(img1, (0, 0))
            new_img.paste(img2, (0, img1.height))
            save_path = video_path.replace(".mp4", f"_{idx}.jpg")
            new_img.save(save_path)
            concat_paths.append(save_path)
        return concat_paths

    def extract_frames_from_video(
        self, video_path, interval=4, max_frames=None
    ):
        """按每k帧取一帧的方式从视频文件中提取帧."""
        cap = cv2.VideoCapture(video_path)
        frame_paths = []

        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        # 获取视频总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_count = 0
        for frame_idx in range(0, total_frames, interval):
            if max_frames is not None and frame_count >= max_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # 转换BGR到RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 使用唯一的临时文件名
                frame_path = os.path.join(
                    self.temp_dir,
                    f"temp_frame_{self.instance_id}_{frame_count}.jpg",
                )
                Image.fromarray(frame_rgb).save(frame_path)
                frame_paths.append(frame_path)
                frame_count += 1

        cap.release()
        return frame_paths

    def process_video_with_dav3(
        self, video_path, interval=4, max_frames=None, last_frames=16
    ):
        """处理视频并返回DAv3预测结果."""
        # 按每k帧取一帧的方式提取视频帧
        frame_paths = self.extract_frames_from_video(
            video_path, interval, max_frames
        )

        first_frame = frame_paths[0]
        images = frame_paths[-last_frames:]

        with torch.no_grad():
            predictions = self.dav3_model.inference(images)

        return first_frame, images, predictions

    def process_video_with_worldmirror(
        self,
        video_path,
        interval=4,
        max_frames=None,
        last_frames=16,
        fps=1,
        target_size=518,
    ):
        """处理视频并返回WorldMirror预测结果."""

        # 按每k帧取一帧的方式提取视频帧
        frame_paths = self.extract_frames_from_video(
            video_path, interval, max_frames
        )

        first_frame = frame_paths[0]
        images = frame_paths[-last_frames:]

        # 创建临时目录保存图片序列
        temp_image_dir = os.path.join(
            self.temp_dir, f"worldmirror_input_{self.instance_id}"
        )
        os.makedirs(temp_image_dir, exist_ok=True)

        # 复制图片到临时目录（WorldMirror需要目录输入）
        for idx, img_path in enumerate(images):
            import shutil

            dst_path = os.path.join(temp_image_dir, f"{idx:05d}.jpg")
            shutil.copy(img_path, dst_path)

        # 加载和预处理图像
        inputs = {}
        inputs["img"] = extract_load_and_preprocess_images(
            Path(temp_image_dir), fps=fps, target_size=target_size
        ).to(
            self.device
        )  # [1, N, 3, H, W], in [0,1]

        # 不使用先验信息
        cond_flags = [0, 0, 0]  # [camera_pose, depth, intrinsics]

        # WorldMirror推理
        with torch.no_grad():
            predictions = self.worldmirror_model(
                views=inputs, cond_flags=cond_flags
            )

        camera_poses = predictions["camera_poses"][0]  # [S, 4, 4]

        w2c_poses = torch.inverse(camera_poses)  # [S, 4, 4]

        class WorldMirrorPredictions:
            def __init__(self, extrinsics):
                self.extrinsics = extrinsics  # [S, 3, 4] format to match DAv3

        # WorldMirror输出是 [S, 4, 4]，转换为 [S, 3, 4] 以匹配DAv3格式
        extrinsics = w2c_poses[:, :3, :].cpu().numpy()  # [S, 3, 4]

        wm_predictions = WorldMirrorPredictions(extrinsics)

        return first_frame, images, wm_predictions

    def process_video(
        self, video_path, interval=4, max_frames=None, last_frames=16
    ):
        """统一的视频处理接口，根据camera_estimator选择处理方法."""
        if self.camera_estimator == "worldmirror":
            return self.process_video_with_worldmirror(
                video_path, interval, max_frames, last_frames
            )
        elif self.camera_estimator == "dav3":
            return self.process_video_with_dav3(
                video_path, interval, max_frames, last_frames
            )
        else:
            raise ValueError(
                f"Unsupported camera_estimator: {self.camera_estimator}. Choose 'worldmirror' or 'dav3'"
            )

    def _camera_pose_to_discrete_action(
        self, camera_poses, move_norm_valid, rot_threshold
    ):
        """将相机姿态转换为离散动作，参考camera_dataset的实现.

        Args:
            camera_poses: 相机姿态矩阵 [N, 4, 4]，假设是w2c格式

        Returns:
            torch.Tensor: 动作标签 [N]
        """
        N = camera_poses.shape[0]

        # 转换为c2w格式
        c2ws = torch.inverse(camera_poses)  # [N, 4, 4]

        # 计算相对变换
        C_inv = torch.inverse(c2ws[:-1])  # [N-1, 4, 4]
        relative_c2w = torch.zeros_like(c2ws)
        relative_c2w[0] = c2ws[0]
        relative_c2w[1:] = torch.bmm(C_inv, c2ws[1:])

        # 初始化动作
        trans_one_hot = torch.zeros(
            (N, 4), dtype=torch.int32, device=camera_poses.device
        )
        rotate_one_hot = torch.zeros(
            (N, 4), dtype=torch.int32, device=camera_poses.device
        )

        for i in range(1, N):
            # 平移部分
            move_dirs = relative_c2w[i, :3, 3]  # [3]
            move_norms = torch.norm(move_dirs)

            if move_norms > move_norm_valid:  # 认为有移动
                move_norm_dirs = move_dirs / move_norms
                # 防止数值误差
                move_norm_dirs = torch.clamp(move_norm_dirs, -1.0, 1.0)
                angles_rad = torch.acos(move_norm_dirs)
                trans_angles_deg = angles_rad * (180.0 / torch.pi)

                # 前后判断 (z轴)
                if trans_angles_deg[2] < 60:
                    trans_one_hot[i, 0] = 1  # 前
                elif trans_angles_deg[2] > 120:
                    trans_one_hot[i, 1] = 1  # 后

                # 左右判断 (x轴)
                if trans_angles_deg[0] < 60:
                    trans_one_hot[i, 2] = 1  # 右
                elif trans_angles_deg[0] > 120:
                    trans_one_hot[i, 3] = 1  # 左

            # 旋转部分 - 转换为欧拉角
            R_rel = relative_c2w[i, :3, :3]

            # 使用与camera_dataset相同的欧拉角转换方法
            sy = torch.sqrt(R_rel[0, 0] ** 2 + R_rel[1, 0] ** 2)
            if sy > 1e-6:
                x = torch.atan2(R_rel[2, 1], R_rel[2, 2])
                y = torch.atan2(-R_rel[2, 0], sy)
                z = torch.atan2(R_rel[1, 0], R_rel[0, 0])
            else:
                x = torch.atan2(-R_rel[1, 2], R_rel[1, 1])
                y = torch.atan2(-R_rel[2, 0], sy)
                z = torch.tensor(0.0, device=camera_poses.device)

            rot_angles_deg = torch.stack([x, y, z]) * (180.0 / torch.pi)

            # 左右旋转 (y轴)
            if rot_angles_deg[1] > rot_threshold:
                rotate_one_hot[i, 0] = 1  # 右
            elif rot_angles_deg[1] < -rot_threshold:
                rotate_one_hot[i, 1] = 1  # 左

            # 上下旋转 (x轴)
            if rot_angles_deg[0] > rot_threshold:
                rotate_one_hot[i, 2] = 1  # 上
            elif rot_angles_deg[0] < -rot_threshold:
                rotate_one_hot[i, 3] = 1  # 下

        # 参考camera_dataset.py的映射方式
        mapping = {
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

        # 分别计算平移和旋转的动作标签
        trans_one_label = torch.zeros(
            N, dtype=torch.long, device=camera_poses.device
        )
        rotate_one_label = torch.zeros(
            N, dtype=torch.long, device=camera_poses.device
        )

        for i in range(N):
            # 平移动作映射
            trans_tuple = tuple(trans_one_hot[i].tolist())
            if trans_tuple in mapping:
                trans_one_label[i] = mapping[trans_tuple]

            # 旋转动作映射
            rotate_tuple = tuple(rotate_one_hot[i].tolist())
            if rotate_tuple in mapping:
                rotate_one_label[i] = mapping[rotate_tuple]

        # 与camera_dataset.py保持一致的最终动作编码
        action_for_pe = trans_one_label * 9 + rotate_one_label

        return action_for_pe

    @torch.no_grad()
    def score_video(
        self,
        video_path,
        caption,
        gt_camera_pose,
        gt_action,
        interval=1,
        latent_num=64,
    ):
        """计算视频分数.

        Args:
            video_path: 视频文件路径
            interval: 间隔帧数 (现在设置为1，每相邻帧都对应一个动作)
            latent_num: 隐变量数量
        """

        first_frame, images, predictions = self.process_video(
            video_path, interval, last_frames=(latent_num * 4) - 3
        )

        pose_pred = torch.tensor(predictions.extrinsics).unsqueeze(0)
        frame_num = pose_pred.shape[1]
        pose_pred = torch.cat(
            [
                pose_pred,
                torch.tensor([0, 0, 0, 1], device=pose_pred.device)
                .view(1, 1, 1, 4)
                .expand(1, frame_num, 1, 4),
            ],
            dim=2,
        )

        # 扩展gt_action: 原来每4帧记录一个动作，现在扩展为每帧都有对应的动作
        # gt_action shape: [batch, latent_num] -> 需要扩展为 [batch, latent_num*4-3]
        expanded_gt_action = gt_action.repeat_interleave(4, dim=1)[
            :, 4:
        ]  # 重复4次并截取到匹配frame数

        # 1. 拿到每一帧的预测pose
        num_frames = len(images)
        chunk_size = 4  # 每个chunk包含4帧
        num_chunks = num_frames // chunk_size  # 计算chunk数量

        first_hps_quality_score = self.hpsv3_model.reward(
            [first_frame],
            ["A high-quality, ultra-detailed and well-structured image."],
        )
        if first_hps_quality_score.ndim == 2:
            first_hps_quality_score = first_hps_quality_score[:, 0].cpu()

        action_acc_list = []
        hps_acc_list = []
        hps_quality_acc_list = []
        actions_summary = []

        for chunk_idx in range(num_chunks):
            chunk_end = ((chunk_idx + 1) * chunk_size) + 1

            # 当前chunk范围的pose和action
            chunk_images = images[:chunk_end]
            img_paths = chunk_images[-chunk_size:]

            # -- Action acc
            pred_camera_pose = pose_pred[0, :chunk_end, :, :].to(torch.float32)
            pred_camera_pose = pred_camera_pose[
                -(chunk_size + 1) :
            ]  # 取最后chunk_size+1帧来计算chunk_size个动作

            pred_action_001 = self._camera_pose_to_discrete_action(
                pred_camera_pose, move_norm_valid=0.002, rot_threshold=0.2
            )[-chunk_size:].cpu()
            pred_action_002 = self._camera_pose_to_discrete_action(
                pred_camera_pose, move_norm_valid=0.005, rot_threshold=0.2
            )[-chunk_size:].cpu()
            pred_action_003 = self._camera_pose_to_discrete_action(
                pred_camera_pose, move_norm_valid=0.01, rot_threshold=0.2
            )[-chunk_size:].cpu()
            chunk_gt_action = expanded_gt_action[0, :chunk_end].cpu()
            chunk_gt_action = chunk_gt_action[-chunk_size:].cpu()

            Action_acc_001 = torch.mean(
                (pred_action_001 == chunk_gt_action).float()
            ).item()
            Action_acc_002 = torch.mean(
                (pred_action_002 == chunk_gt_action).float()
            ).item()
            Action_acc_003 = torch.mean(
                (pred_action_003 == chunk_gt_action).float()
            ).item()

            Action_acc = max(Action_acc_001, Action_acc_002, Action_acc_003)
            # -- HPSv3 acc

            # concat_img_paths = self.concat_images_vertically([first_frame], [img_paths[-1]], video_path)

            hps_score = self.hpsv3_model.reward(
                [img_paths[-1]], [caption] * len([img_paths[-1]])
            )
            if hps_score.ndim == 2:
                hps_score = hps_score[:, 0]

            hps_quality_score = self.hpsv3_model.reward(
                [img_paths[-1]],
                ["A high-quality, ultra-detailed and well-structured image."]
                * len([img_paths[-1]]),
            )
            if hps_quality_score.ndim == 2:
                hps_quality_score = hps_quality_score[:, 0]

            action_acc_list.append(Action_acc)
            hps_acc_list.append(hps_score.mean().item())
            hps_quality_acc_list.append(hps_quality_score.mean().item())
            actions_summary += pred_action_001.cpu().tolist()

        hps_drift_score_list = -1 * torch.abs(
            torch.tensor(hps_quality_acc_list) - first_hps_quality_score
        )
        hps_drift_score_list = hps_drift_score_list[-4:]

        return {
            "action_acc": action_acc_list,
            "hps_acc": hps_acc_list,
            "hps_quality_acc": hps_quality_acc_list,
            "hps_drift_score": hps_drift_score_list,
        }

    @torch.no_grad()
    def reward(
        self,
        video_path,
        gt_camera_pose=None,
        gt_action=None,
        caption=None,
        interval=1,
        update_latent_num=4,
    ):
        """计算视频一致性奖励.

        Args:
            video_path: 视频文件路径
            max_frames: 最大提取帧数
            top_percent: 取相对距离误差最大的百分比

        Returns:
            float: 相对距离误差最大前20%的平均值（负值，越小越好）
        """

        expected_frame_num = (update_latent_num * 4) + 1
        first_frame, images, predictions = self.process_video(
            video_path, interval, last_frames=expected_frame_num
        )
        actual_frame_num = len(images)

        if actual_frame_num != expected_frame_num:
            expanded_gt_action = gt_action.repeat_interleave(4, dim=1)[
                0, 4:
            ].cpu()
        else:
            expanded_gt_action = gt_action.repeat_interleave(4, dim=1)[0].cpu()

        camera_pose = torch.tensor(predictions.extrinsics).unsqueeze(0)
        camera_pose = torch.cat(
            [
                camera_pose,
                torch.tensor([0, 0, 0, 1], device=camera_pose.device)
                .view(1, 1, 1, 4)
                .expand(1, actual_frame_num, 1, 4),
            ],
            dim=2,
        )[0]

        pred_action_001 = self._camera_pose_to_discrete_action(
            camera_pose, move_norm_valid=0.002, rot_threshold=0.2
        )[-len(expanded_gt_action) :].cpu()
        pred_action_002 = self._camera_pose_to_discrete_action(
            camera_pose, move_norm_valid=0.005, rot_threshold=0.2
        )[-len(expanded_gt_action) :].cpu()
        pred_action_003 = self._camera_pose_to_discrete_action(
            camera_pose, move_norm_valid=0.01, rot_threshold=0.2
        )[-len(expanded_gt_action) :].cpu()

        Action_acc_001 = torch.mean(
            (pred_action_001 == expanded_gt_action).float()
        )
        Action_acc_002 = torch.mean(
            (pred_action_002 == expanded_gt_action).float()
        )
        Action_acc_003 = torch.mean(
            (pred_action_003 == expanded_gt_action).float()
        )

        action_accs = torch.stack(
            [Action_acc_001, Action_acc_002, Action_acc_003]
        )
        pred_actions = [pred_action_001, pred_action_002, pred_action_003]
        best_idx = torch.argmax(action_accs).item()
        Action_acc = action_accs[best_idx]
        pred_action = pred_actions[best_idx]

        pred_trans_one_label = pred_action // 9
        pred_rotate_one_label = pred_action % 9
        gt_trans_one_label = (expanded_gt_action // 9).to(torch.long)
        gt_rotate_one_label = (expanded_gt_action % 9).to(torch.long)

        Fine_Action_acc = (
            torch.mean(
                (
                    pred_trans_one_label
                    == gt_trans_one_label.cpu().to(torch.long)
                ).float()
            )
            + torch.mean(
                (
                    pred_rotate_one_label
                    == gt_rotate_one_label.cpu().to(torch.long)
                ).float()
            )
        ) / 2

        hps_images = images[3::4]

        hps_score = self.hpsv3_model.reward(
            hps_images, [caption] * len(hps_images)
        )
        if hps_score.ndim == 2:
            hps_score = hps_score[:, 0]

        hps_quality_score = self.hpsv3_model.reward(
            hps_images,
            ["A high-quality, ultra-detailed and well-structured image."]
            * len(hps_images),
        )
        if hps_quality_score.ndim == 2:
            hps_quality_score = hps_quality_score[:, 0].cpu()

        first_hps_quality_score = self.hpsv3_model.reward(
            [first_frame],
            ["A high-quality, ultra-detailed and well-structured image."],
        )
        if first_hps_quality_score.ndim == 2:
            first_hps_quality_score = first_hps_quality_score[:, 0].cpu()

        hpsv3_quality_drift_score = -1 * torch.abs(
            hps_quality_score - first_hps_quality_score
        )

        for img_path in images:
            if os.path.exists(img_path):
                os.remove(img_path)

        if os.path.exists(first_frame):
            os.remove(first_frame)

        Geometry_acc = torch.tensor(0.0)
        trans_error = torch.tensor(0.0)
        rot_error = torch.tensor(0.0)
        Camera_acc = torch.tensor(0.0)

        return {
            "action_acc": Action_acc.item(),
            "fine_action_acc": Fine_Action_acc.item(),
            "geometry_acc": Geometry_acc.item(),
            "camera_acc": Camera_acc.item(),
            "hpsv3_acc": hps_score.mean().item(),
            "hpsv3_quality_acc": hps_quality_score.mean().item(),
            "hpsv3_quality_drift_score": hpsv3_quality_drift_score.mean().item(),
            "trans_error": trans_error if trans_error is not None else None,
            "rot_error": rot_error if rot_error is not None else None,
            "pred_trans_one_label": (
                pred_trans_one_label.tolist()
                if pred_trans_one_label is not None
                else None
            ),
            "gt_trans_one_label": (
                gt_trans_one_label.tolist()
                if gt_trans_one_label is not None
                else None
            ),
            "pred_rotate_one_label": (
                pred_rotate_one_label.tolist()
                if pred_rotate_one_label is not None
                else None
            ),
            "gt_rotate_one_label": (
                gt_rotate_one_label.tolist()
                if gt_rotate_one_label is not None
                else None
            ),
        }


if __name__ == "__main__":
    # 测试视频路径列表
    video_paths = [
        "generated_videos/ar_grpo_no_same_noise_32/step_1/0_0_chunk_2_VQ_-0.91_MQ_-0.09.mp4",  # 好
        "generated_videos/ar_grpo_no_same_noise_32/step_1/0_1_chunk_2_VQ_-0.88_MQ_0.17.mp4",  # 好
        "generated_videos/ar_grpo_no_same_noise_32/step_1/0_5_chunk_2_VQ_-1.17_MQ_-0.19.mp4",  # 不好
        "generated_videos/ar_grpo_no_same_noise_32/step_1/0_11_chunk_2_VQ_-1.09_MQ_-0.46.mp4",  # 不好
    ]

    # 使用上下文管理器确保资源自动清理
    with VideoConsistencyReward() as reward_calculator:
        # 为每个视频计算奖励值
        print(f"\n{'=' * 60}")
        print("视频一致性奖励计算结果")
        print(f"{'=' * 60}")

        results = []

        for video_idx, video_path in enumerate(video_paths):
            print(f"\n正在处理视频 {video_idx + 1}: {video_path}")

            # 计算奖励值（相对距离误差最大前20%的平均值）
            reward_value = reward_calculator.reward(
                video_path, max_frames=20, top_percent=0.2
            )

            result = {
                "video_idx": video_idx + 1,
                "video_path": video_path,
                "reward": reward_value,
            }
            results.append(result)

            print(
                f"奖励值（相对距离误差最大前20%的平均值）: {reward_value:.4f}"
            )

        # 总结结果
        print(f"\n{'=' * 60}")
        print("总结")
        print(f"{'=' * 60}")

        for result in results:
            print(
                f"视频 {result['video_idx']}: {result['reward']:.4f} - {result['video_path']}"
            )

        print("\n处理完成！")

    # 临时目录会在这里自动清理
