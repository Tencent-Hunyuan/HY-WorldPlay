"""Utility functions for cropping and resizing data while maintaining proper cameras.

References: DUSt3R
"""

import cv2
import numpy as np
import PIL.Image

try:
    lanczos = PIL.Image.Resampling.LANCZOS
    bicubic = PIL.Image.Resampling.BICUBIC
except AttributeError:
    lanczos = PIL.Image.LANCZOS
    bicubic = PIL.Image.BICUBIC

from reward_function.HunyuanWorldMirror.src.utils.geometry import (
    colmap_to_opencv_intrinsics,
    opencv_to_colmap_intrinsics,
)


class ImageList:
    """Convenience class to apply the same operation to a whole set of images.

    This class wraps a list of PIL.Image objects and provides methods to perform operations on all
    images simultaneously.
    """

    def __init__(self, images):
        if not isinstance(images, (tuple, list, set)):
            images = [images]
        self.images = []
        for image in images:
            if not isinstance(image, PIL.Image.Image):
                image = PIL.Image.fromarray(image)
            self.images.append(image)

    def __len__(self):
        """Return the number of images in the list."""
        return len(self.images)

    def to_pil(self):
        """Convert ImageList back to PIL Image(s).

        Returns:
            PIL.Image.Image or tuple: Single PIL Image if list contains one image,
                                      or tuple of PIL Images if multiple images
        """
        return tuple(self.images) if len(self.images) > 1 else self.images[0]

    @property
    def size(self):
        """Get the size of images in the list.

        Returns:
            tuple: (width, height) of the images

        Raises:
            AssertionError: If images have different sizes
        """
        sizes = [im.size for im in self.images]
        assert all(
            sizes[0] == s for s in sizes
        ), "All images must have the same size"
        return sizes[0]

    def resize(self, *args, **kwargs):
        """Resize all images with the same parameters.

        Args:
            *args, **kwargs: Arguments passed to PIL.Image.resize()

        Returns:
            ImageList: New ImageList containing resized images
        """
        return ImageList(self._dispatch("resize", *args, **kwargs))

    def crop(self, *args, **kwargs):
        """Crop all images with the same parameters.

        Args:
            *args, **kwargs: Arguments passed to PIL.Image.crop()

        Returns:
            ImageList: New ImageList containing cropped images
        """
        return ImageList(self._dispatch("crop", *args, **kwargs))

    def _dispatch(self, func, *args, **kwargs):
        """Apply a PIL.Image method to all images in the list.

        Args:
            func (str): Name of the PIL.Image method to call
            *args, **kwargs: Arguments to pass to the method

        Returns:
            list: List of results from applying the method to each image
        """
        return [getattr(im, func)(*args, **kwargs) for im in self.images]


def rescale_image_and_other_optional_info(
    image,
    output_resolution,
    depthmap=None,
    camera_intrinsics=None,
    force=True,
    additional_quantities_to_be_resized_with_nearest=None,
):
    """Rescale the image and depthmap to the output resolution. If the image is larger than the
    output resolution, it is rescaled with lanczos interpolation. If force is false and the image is
    smaller than the output resolution, it is not rescaled. If force is true and the image is
    smaller than the output resolution, it is rescaled with bicubic interpolation. Depth and other
    quantities are rescaled with nearest interpolation.

    Args:
        image (PIL.Image.Image or np.ndarray): The input image to be rescaled.
        output_resolution (tuple): The desired output resolution as a tuple (width, height).
        depthmap (np.ndarray, optional): The depth map associated with the image. Defaults to None.
        camera_intrinsics (np.ndarray, optional): The camera intrinsics matrix. Defaults to None.
        force (bool, optional): If True, force rescaling even if the image is smaller than the output resolution. Defaults to True.
        additional_quantities_to_be_resized_with_nearest (list of np.ndarray, optional): Additional quantities to be rescaled using nearest interpolation. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - The rescaled image (PIL.Image.Image)
            - The rescaled depthmap (numpy.ndarray or None)
            - The updated camera intrinsics (numpy.ndarray or None)
            - The list of rescaled additional quantities (list of numpy.ndarray or None)
    """
    image = ImageList(image)
    input_resolution = np.array(image.size)  # (W, H)
    output_resolution = np.array(output_resolution)
    if depthmap is not None:
        assert tuple(depthmap.shape[:2]) == image.size[::-1]
    if additional_quantities_to_be_resized_with_nearest is not None:
        assert all(
            tuple(additional_quantity.shape[:2]) == image.size[::-1]
            for additional_quantity in additional_quantities_to_be_resized_with_nearest
        )

    # Define output resolution
    assert output_resolution.shape == (2,)
    scale_final = max(output_resolution / image.size) + 1e-8
    if (
        scale_final >= 1 and not force
    ):  # image is already smaller than what is asked
        output = (
            image.to_pil(),
            depthmap,
            camera_intrinsics,
            additional_quantities_to_be_resized_with_nearest,
        )
        return output
    output_resolution = np.floor(input_resolution * scale_final).astype(int)

    # First rescale the image so that it contains the crop
    image = image.resize(
        tuple(output_resolution),
        resample=lanczos if scale_final < 1 else bicubic,
    )
    if depthmap is not None:
        depthmap = cv2.resize(
            depthmap,
            output_resolution,
            fx=scale_final,
            fy=scale_final,
            interpolation=cv2.INTER_NEAREST,
        )
    if additional_quantities_to_be_resized_with_nearest is not None:
        resized_additional_quantities = []
        for quantity in additional_quantities_to_be_resized_with_nearest:
            resized_additional_quantities.append(
                cv2.resize(
                    quantity,
                    output_resolution,
                    fx=scale_final,
                    fy=scale_final,
                    interpolation=cv2.INTER_NEAREST,
                )
            )
        additional_quantities_to_be_resized_with_nearest = (
            resized_additional_quantities
        )

    # No offset here; simple rescaling
    if camera_intrinsics is not None:
        camera_intrinsics = camera_matrix_of_crop(
            camera_intrinsics,
            input_resolution,
            output_resolution,
            scaling=scale_final,
        )

    # Return
    return (
        image.to_pil(),
        depthmap,
        camera_intrinsics,
        additional_quantities_to_be_resized_with_nearest,
    )


def camera_matrix_of_crop(
    input_camera_matrix,
    input_resolution,
    output_resolution,
    scaling=1,
    offset_factor=0.5,
    offset=None,
):
    """Calculate the camera matrix for a cropped image.

    Args:
        input_camera_matrix (numpy.ndarray): Original camera intrinsics matrix
        input_resolution (tuple or numpy.ndarray): Original image resolution as (width, height)
        output_resolution (tuple or numpy.ndarray): Target image resolution as (width, height)
        scaling (float, optional): Scaling factor for the image. Defaults to 1.
        offset_factor (float, optional): Factor to determine crop offset. Defaults to 0.5 (centered).
        offset (tuple or numpy.ndarray, optional): Explicit offset to use. If None, calculated from offset_factor.

    Returns:
        numpy.ndarray: Updated camera matrix for the cropped image
    """
    # Margins to offset the origin
    margins = np.asarray(input_resolution) * scaling - output_resolution
    assert np.all(margins >= 0.0)
    if offset is None:
        offset = offset_factor * margins

    # Generate new camera parameters
    output_camera_matrix_colmap = opencv_to_colmap_intrinsics(
        input_camera_matrix
    )
    output_camera_matrix_colmap[:2, :] *= scaling
    output_camera_matrix_colmap[:2, 2] -= offset
    output_camera_matrix = colmap_to_opencv_intrinsics(
        output_camera_matrix_colmap
    )

    return output_camera_matrix


def crop_image_and_other_optional_info(
    image,
    crop_bbox,
    depthmap=None,
    camera_intrinsics=None,
    additional_quantities=None,
):
    """Return a crop of the input view and associated data.

    Args:
        image (PIL.Image.Image or numpy.ndarray): The input image to be cropped
        crop_bbox (tuple): Crop bounding box as (left, top, right, bottom)
        depthmap (numpy.ndarray, optional): Depth map associated with the image
        camera_intrinsics (numpy.ndarray, optional): Camera intrinsics matrix
        additional_quantities (list of numpy.ndarray, optional): Additional data arrays to crop

    Returns:
        tuple: A tuple containing:
            - The cropped image
            - The cropped depth map (if provided or None)
            - Updated camera intrinsics (if provided or None)
            - List of cropped additional quantities (if provided or None)
    """
    image = ImageList(image)
    left, top, right, bottom = crop_bbox

    image = image.crop((left, top, right, bottom))
    if depthmap is not None:
        depthmap = depthmap[top:bottom, left:right]
    if additional_quantities is not None:
        additional_quantities = [
            quantity[top:bottom, left:right]
            for quantity in additional_quantities
        ]

    if camera_intrinsics is not None:
        camera_intrinsics = camera_intrinsics.copy()
        camera_intrinsics[0, 2] -= left
        camera_intrinsics[1, 2] -= top

    return (image.to_pil(), depthmap, camera_intrinsics, additional_quantities)


def crop_image_depthmap(image, depthmap, camera_intrinsics, crop_bbox):
    """Return a crop of the input view."""
    image = ImageList(image)
    l, t, r, b = crop_bbox

    image = image.crop((l, t, r, b))
    if depthmap is not None:
        depthmap = depthmap[t:b, l:r]

    camera_intrinsics = camera_intrinsics.copy()
    camera_intrinsics[0, 2] -= l
    camera_intrinsics[1, 2] -= t

    return image.to_pil(), depthmap, camera_intrinsics


def crop_image_normalmap(
    image, normalmap, camera_intrinsics, valid_mask, crop_bbox
):
    """Return a crop of the input view."""
    image = ImageList(image)
    l, t, r, b = crop_bbox

    image = image.crop((l, t, r, b))
    normalmap = normalmap[t:b, l:r]
    valid_mask = valid_mask[t:b, l:r]

    camera_intrinsics = camera_intrinsics.copy()
    camera_intrinsics[0, 2] -= l
    camera_intrinsics[1, 2] -= t

    return image.to_pil(), normalmap, camera_intrinsics, valid_mask


def rescale_image_depthmap(
    image, depthmap, camera_intrinsics, output_resolution, force=True
):
    """Jointly rescale a (image, depthmap) so that (out_width, out_height) >= output_res."""
    image = ImageList(image)
    input_resolution = np.array(image.size)  # (W,H)
    output_resolution = np.array(output_resolution)
    if depthmap is not None:
        # can also use this with masks instead of depthmaps
        assert tuple(depthmap.shape[:2]) == image.size[::-1]

    # define output resolution
    assert output_resolution.shape == (2,)
    scale_final = max(output_resolution / image.size) + 1e-8
    if (
        scale_final >= 1 and not force
    ):  # image is already smaller than what is asked
        return (image.to_pil(), depthmap, camera_intrinsics)
    output_resolution = np.floor(input_resolution * scale_final).astype(int)

    # first rescale the image so that it contains the crop
    image = image.resize(
        output_resolution, resample=lanczos if scale_final < 1 else bicubic
    )
    if depthmap is not None:
        depthmap = cv2.resize(
            depthmap,
            output_resolution,
            fx=scale_final,
            fy=scale_final,
            interpolation=cv2.INTER_NEAREST,
        )

    # no offset here; simple rescaling
    camera_intrinsics = camera_matrix_of_crop(
        camera_intrinsics,
        input_resolution,
        output_resolution,
        scaling=scale_final,
    )

    return image.to_pil(), depthmap, camera_intrinsics


def rescale_image_normalmap(
    image,
    normalmap,
    camera_intrinsics,
    normal_mask,
    output_resolution,
    force=True,
):
    """Jointly rescale a (image, normalmap, normal_mask) so that (out_width, out_height) >=
    output_res."""
    image = ImageList(image)
    input_resolution = np.array(image.size)  # (W,H)
    output_resolution = np.array(output_resolution)
    if normalmap is not None:
        assert tuple(normalmap.shape[:2]) == image.size[::-1]
    if normal_mask is not None:
        assert tuple(normal_mask.shape[:2]) == image.size[::-1]

    # define output resolution
    assert output_resolution.shape == (2,)
    scale_final = max(output_resolution / image.size) + 1e-8
    if (
        scale_final >= 1 and not force
    ):  # image is already smaller than what is asked
        return (image.to_pil(), normalmap, camera_intrinsics, normal_mask)
    output_resolution = np.floor(input_resolution * scale_final).astype(int)

    # first rescale the image so that it contains the crop
    image = image.resize(
        output_resolution, resample=lanczos if scale_final < 1 else bicubic
    )
    if normalmap is not None:
        normalmap = cv2.resize(
            normalmap,
            output_resolution,
            fx=scale_final,
            fy=scale_final,
            interpolation=cv2.INTER_NEAREST,
        )
    if normal_mask is not None:
        # Convert bool to uint8 for cv2.resize, then back to bool
        normal_mask_uint8 = normal_mask.astype(np.uint8)
        normal_mask_uint8 = cv2.resize(
            normal_mask_uint8,
            output_resolution,
            fx=scale_final,
            fy=scale_final,
            interpolation=cv2.INTER_NEAREST,
        )
        normal_mask = normal_mask_uint8.astype(bool)

    # no offset here; simple rescaling
    camera_intrinsics = camera_matrix_of_crop(
        camera_intrinsics,
        input_resolution,
        output_resolution,
        scaling=scale_final,
    )

    return image.to_pil(), normalmap, camera_intrinsics, normal_mask


def bbox_from_intrinsics_in_out(
    input_camera_matrix, output_camera_matrix, output_resolution
):
    """Calculate the bounding box for cropping based on input and output camera intrinsics.

    Args:
        input_camera_matrix (numpy.ndarray): Original camera intrinsics matrix
        output_camera_matrix (numpy.ndarray): Target camera intrinsics matrix
        output_resolution (tuple): Target resolution as (width, height)

    Returns:
        tuple: Crop bounding box as (left, top, right, bottom)
    """
    out_width, out_height = output_resolution
    left, top = np.int32(
        np.round(input_camera_matrix[:2, 2] - output_camera_matrix[:2, 2])
    )
    crop_bbox = (left, top, left + out_width, top + out_height)
    return crop_bbox


def crop_resize_if_necessary(
    image,
    resolution,
    depthmap=None,
    intrinsics=None,
    additional_quantities=None,
):
    """First downsample image using LANCZOS and then crop if necessary to achieve target resolution.

    This function performs high-quality downsampling followed by cropping to achieve the
    desired output resolution while maintaining proper camera intrinsics.

    Args:
        image (PIL.Image.Image or numpy.ndarray): The input image to be processed
        resolution (tuple): Target resolution as (width, height)
        depthmap (numpy.ndarray, optional): Depth map associated with the image
        intrinsics (numpy.ndarray, optional): Camera intrinsics matrix
        additional_quantities (list of numpy.ndarray, optional): Additional data arrays to process

    Returns:
        tuple: A tuple containing the processed image and any provided additional data
               (depthmap, intrinsics, additional_quantities) that have been similarly processed
    """
    # Convert image to PIL.Image.Image if necessary
    if not isinstance(image, PIL.Image.Image):
        image = PIL.Image.fromarray(image)

    # Get width and height of image
    original_width, original_height = image.size

    # High-quality Lanczos down-scaling
    target_rescale_resolution = np.array(resolution)
    image, depthmap, intrinsics, additional_quantities = (
        rescale_image_and_other_optional_info(
            image=image,
            output_resolution=target_rescale_resolution,
            depthmap=depthmap,
            camera_intrinsics=intrinsics,
            additional_quantities_to_be_resized_with_nearest=additional_quantities,
        )
    )

    # Actual cropping (if necessary)
    if intrinsics is not None:
        new_intrinsics = camera_matrix_of_crop(
            input_camera_matrix=intrinsics,
            input_resolution=image.size,
            output_resolution=resolution,
            offset_factor=0.5,
        )
        crop_bbox = bbox_from_intrinsics_in_out(
            input_camera_matrix=intrinsics,
            output_camera_matrix=new_intrinsics,
            output_resolution=resolution,
        )
    else:
        # Create a centered crop if no intrinsics are available
        w, h = image.size
        target_w, target_h = resolution
        left = (w - target_w) // 2
        top = (h - target_h) // 2
        crop_bbox = (left, top, left + target_w, top + target_h)

    image, depthmap, new_intrinsics, additional_quantities = (
        crop_image_and_other_optional_info(
            image=image,
            crop_bbox=crop_bbox,
            depthmap=depthmap,
            camera_intrinsics=intrinsics,
            additional_quantities=additional_quantities,
        )
    )

    # Return the output
    output = (image,)
    if depthmap is not None:
        output += (depthmap,)
    if new_intrinsics is not None:
        output += (new_intrinsics,)
    if additional_quantities is not None:
        output += (additional_quantities,)
    return output
