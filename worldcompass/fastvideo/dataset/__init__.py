# SPDX-License-Identifier: Apache-2.0
from torchvision import transforms
from torchvision.transforms import Lambda

from fastvideo.dataset.camera_dataset import build_hy_camera_dataloader
from fastvideo.dataset.preprocessing_datasets import VideoCaptionMergedDataset
from fastvideo.dataset.transform import (
    CenterCropResizeVideo,
    Normalize255,
    TemporalRandomCrop,
)
from fastvideo.dataset.validation_dataset import ValidationDataset


def getdataset(args) -> VideoCaptionMergedDataset:
    if args.do_temporal_sample:
        temporal_sample = TemporalRandomCrop(args.num_frames)  # 16 x
    else:
        temporal_sample = None
    norm_fun = Lambda(lambda x: 2.0 * x - 1.0)
    resize_topcrop = [
        CenterCropResizeVideo((args.max_height, args.max_width), top_crop=True),
    ]
    resize = [
        CenterCropResizeVideo((args.max_height, args.max_width)),
    ]
    transform = transforms.Compose(
        [
            # Normalize255(),
            *resize,
        ]
    )
    transform_topcrop = transforms.Compose(
        [
            Normalize255(),
            *resize_topcrop,
            norm_fun,
        ]
    )
    return VideoCaptionMergedDataset(
        data_merge_path=args.data_merge_path,
        args=args,
        transform=transform,
        temporal_sample=temporal_sample,
        transform_topcrop=transform_topcrop,
        seed=args.seed,
    )


__all__ = [
    "ValidationDataset",
    "build_camera_dataloader",
    "build_ar_dataloader",
    "build_hy_camera_dataloader",
    "VideoCaptionMergedDataset",
]
