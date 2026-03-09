import torch
from PIL import Image
from torchvision import transforms

import glob
import os
from reward_function.HunyuanWorldMirror.src.utils.video_utils import (
    video_to_image_frames,
)

IMAGE_EXTS = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.webp"]
VIDEO_EXTS = [".mp4", ".avi", ".mov", ".webm", ".gif"]


def load_and_preprocess_images(
    image_file_paths, preprocessing_mode="crop", output_size=518
):
    """Transform raw image files into model-ready tensor batches with standardized dimensions.

    This utility function handles the complete pipeline from file paths to batched tensors,
    ensuring compatibility with neural network requirements while preserving image quality.

    Args:
        image_file_paths (list): Collection of file system paths pointing to image files
        preprocessing_mode (str, optional): Image transformation strategy:
                             - "crop" (default): Resize width to 518px, center-crop height if oversized
                             - "pad": Scale largest dimension to 518px, pad smaller dimension to square
        output_size (int, optional): Target dimension for model input (default: 518)

    Returns:
        torch.Tensor: Processed image batch with shape (1, N, 3, H, W) ready for model inference

    Raises:
        ValueError: When input validation fails (empty list or invalid mode)

    Implementation Details:
        - Automatic alpha channel handling: RGBA images composited onto white backgrounds
        - Dimension normalization: All outputs divisible by 14 for patch-based processing
        - Batch consistency: Different-sized images padded to uniform dimensions
        - Memory optimization: Efficient tensor operations with minimal data copying
        - Quality preservation: Bicubic resampling maintains visual fidelity
    """
    # Input validation and parameter setup
    if len(image_file_paths) == 0:
        raise ValueError("At least 1 image is required")

    if preprocessing_mode not in ["crop", "pad"]:
        raise ValueError("preprocessing_mode must be either 'crop' or 'pad'")

    processed_image_list = []
    image_dimension_set = set()
    tensor_converter = transforms.ToTensor()
    model_target_size = output_size

    # Individual image processing pipeline
    for image_file_path in image_file_paths:
        # File system to memory conversion
        loaded_image = Image.open(image_file_path)

        # Transparency handling for RGBA images
        if loaded_image.mode == "RGBA":
            # Generate white canvas matching image dimensions
            white_background = Image.new(
                "RGBA", loaded_image.size, (255, 255, 255, 255)
            )
            # Blend transparent pixels with white background
            loaded_image = Image.alpha_composite(white_background, loaded_image)

        # Format standardization to RGB
        loaded_image = loaded_image.convert("RGB")

        original_width, original_height = loaded_image.size

        # Dimension calculation based on preprocessing strategy
        if preprocessing_mode == "pad":
            # Proportional scaling to fit largest dimension within target
            if original_width >= original_height:
                scaled_width = model_target_size
                scaled_height = (
                    round(
                        original_height * (scaled_width / original_width) / 14
                    )
                    * 14
                )  # Patch compatibility
            else:
                scaled_height = model_target_size
                scaled_width = (
                    round(
                        original_width * (scaled_height / original_height) / 14
                    )
                    * 14
                )  # Patch compatibility
        else:  # preprocessing_mode == "crop"
            # Width normalization with proportional height adjustment
            scaled_width = model_target_size
            scaled_height = (
                round(original_height * (scaled_width / original_width) / 14)
                * 14
            )

        # High-quality image resizing
        loaded_image = loaded_image.resize(
            (scaled_width, scaled_height), Image.Resampling.BICUBIC
        )
        image_tensor = tensor_converter(
            loaded_image
        )  # Normalize to [0, 1] range

        # Height trimming for crop mode (center-based)
        if preprocessing_mode == "crop" and scaled_height > model_target_size:
            crop_start_y = (scaled_height - model_target_size) // 2
            image_tensor = image_tensor[
                :, crop_start_y : crop_start_y + model_target_size, :
            ]

        # Square padding for pad mode (centered)
        if preprocessing_mode == "pad":
            height_padding_needed = model_target_size - image_tensor.shape[1]
            width_padding_needed = model_target_size - image_tensor.shape[2]

            if height_padding_needed > 0 or width_padding_needed > 0:
                padding_top = height_padding_needed // 2
                padding_bottom = height_padding_needed - padding_top
                padding_left = width_padding_needed // 2
                padding_right = width_padding_needed - padding_left

                # White padding application (value=1.0 for normalized images)
                image_tensor = torch.nn.functional.pad(
                    image_tensor,
                    (padding_left, padding_right, padding_top, padding_bottom),
                    mode="constant",
                    value=1.0,
                )

        image_dimension_set.add((image_tensor.shape[1], image_tensor.shape[2]))
        processed_image_list.append(image_tensor)

    # Cross-image dimension harmonization
    if len(image_dimension_set) > 1:
        print(
            f"Warning: Found images with different shapes: {image_dimension_set}"
        )
        # Calculate maximum dimensions across the batch
        maximum_height = max(dimension[0] for dimension in image_dimension_set)
        maximum_width = max(dimension[1] for dimension in image_dimension_set)

        # Uniform padding to achieve batch consistency
        uniformly_sized_images = []
        for image_tensor in processed_image_list:
            height_padding_needed = maximum_height - image_tensor.shape[1]
            width_padding_needed = maximum_width - image_tensor.shape[2]

            if height_padding_needed > 0 or width_padding_needed > 0:
                padding_top = height_padding_needed // 2
                padding_bottom = height_padding_needed - padding_top
                padding_left = width_padding_needed // 2
                padding_right = width_padding_needed - padding_left

                image_tensor = torch.nn.functional.pad(
                    image_tensor,
                    (padding_left, padding_right, padding_top, padding_bottom),
                    mode="constant",
                    value=1.0,
                )
            uniformly_sized_images.append(image_tensor)
        processed_image_list = uniformly_sized_images

    # Batch tensor construction
    batched_images = torch.stack(
        processed_image_list
    )  # Concatenate along batch dimension

    # Single image batch dimension handling
    if len(image_file_paths) == 1:
        # Ensure proper 4D tensor structure (batch, channels, height, width)
        if batched_images.dim() == 3:
            batched_images = batched_images.unsqueeze(0)

    return batched_images.unsqueeze(0)


def _handle_alpha_channel(img_data):
    """Process RGBA images by blending with white background."""
    if img_data.mode == "RGBA":
        white_bg = Image.new("RGBA", img_data.size, (255, 255, 255, 255))
        img_data = Image.alpha_composite(white_bg, img_data)
    return img_data.convert("RGB")


def _calculate_resize_dims(
    orig_w, orig_h, max_dim, resize_strategy, patch_size=14
):
    """Calculate new dimensions based on resize strategy."""
    if resize_strategy == "pad":
        if orig_w >= orig_h:
            new_w = max_dim
            new_h = round(orig_h * (new_w / orig_w) / patch_size) * patch_size
        else:
            new_h = max_dim
            new_w = round(orig_w * (new_h / orig_h) / patch_size) * patch_size
    else:  # crop strategy
        new_w = max_dim
        new_h = round(orig_h * (new_w / orig_w) / patch_size) * patch_size
    return new_w, new_h


def _apply_padding(tensor_img, target_dim):
    """Apply padding to make tensor square."""
    h_pad = target_dim - tensor_img.shape[1]
    w_pad = target_dim - tensor_img.shape[2]

    if h_pad > 0 or w_pad > 0:
        pad_top, pad_bottom = h_pad // 2, h_pad - h_pad // 2
        pad_left, pad_right = w_pad // 2, w_pad - w_pad // 2
        return torch.nn.functional.pad(
            tensor_img,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=1.0,
        )
    return tensor_img


def prepare_images_to_tensor(
    file_paths, resize_strategy="crop", target_size=518
):
    """Process image files into uniform tensor batch for model input.

    Args:
        file_paths (list): Paths to image files
        resize_strategy (str): "crop" or "pad" processing mode
        target_size (int): Target size for processing

    Returns:
        torch.Tensor: Processed image batch (1, N, 3, H, W)
    """
    if not file_paths:
        raise ValueError("At least 1 image is required")

    if resize_strategy not in ["crop", "pad"]:
        raise ValueError("Strategy must be 'crop' or 'pad'")

    tensor_list = []
    dimension_set = set()
    converter = transforms.ToTensor()

    # Process each image file
    for file_path in file_paths:
        img_data = Image.open(file_path)
        img_data = _handle_alpha_channel(img_data)

        orig_w, orig_h = img_data.size
        new_w, new_h = _calculate_resize_dims(
            orig_w, orig_h, target_size, resize_strategy
        )

        # Resize and convert to tensor
        img_data = img_data.resize((new_w, new_h), Image.Resampling.BICUBIC)
        tensor_img = converter(img_data)

        # Apply center crop for crop strategy
        if resize_strategy == "crop" and new_h > target_size:
            crop_start = (new_h - target_size) // 2
            tensor_img = tensor_img[:, crop_start : crop_start + target_size, :]

        # Apply padding for pad strategy
        if resize_strategy == "pad":
            tensor_img = _apply_padding(tensor_img, target_size)

        dimension_set.add((tensor_img.shape[1], tensor_img.shape[2]))
        tensor_list.append(tensor_img)

    # Handle mixed dimensions
    if len(dimension_set) > 1:
        print(f"Warning: Mixed image dimensions found: {dimension_set}")
        max_h = max(dims[0] for dims in dimension_set)
        max_w = max(dims[1] for dims in dimension_set)

        tensor_list = [
            (
                _apply_padding(img, max(max_h, max_w))
                if img.shape[1] != max_h or img.shape[2] != max_w
                else img
            )
            for img in tensor_list
        ]

    batch_tensor = torch.stack(tensor_list)

    # Ensure proper batch dimensions
    if batch_tensor.dim() == 3:
        batch_tensor = batch_tensor.unsqueeze(0)

    return batch_tensor.unsqueeze(0)


def extract_load_and_preprocess_images(
    image_folder_or_video_path, fps=1, target_size=518, mode="crop"
):
    # Support multiple image formats
    if (
        image_folder_or_video_path.is_file()
        and image_folder_or_video_path.suffix.lower() in VIDEO_EXTS
    ):
        frame_paths = video_to_image_frames(
            str(image_folder_or_video_path), fps=fps
        )
        img_paths = sorted(frame_paths)
    else:
        img_paths = []
        for ext in IMAGE_EXTS:
            img_paths.extend(
                glob.glob(os.path.join(str(image_folder_or_video_path), ext))
            )
        img_paths = sorted(img_paths)
    images = prepare_images_to_tensor(
        img_paths, resize_strategy=mode, target_size=target_size
    )
    return images
