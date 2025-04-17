# coding=utf-8
# Copyright 2025 The OpenBMB Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Image processor class for MiniCPM-O."""

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
    convert_to_rgb,
    get_resize_output_image_size,
    pad,
    resize,
    to_channel_dimension_format,
)
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_flat_list_of_images,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from ...utils import TensorType, logging


logger = logging.get_logger(__name__)


def recursive_converter(converter, value):
    if isinstance(value, list):
        new_value = []
        for v in value:
            new_value += [recursive_converter(converter, v)]
        return new_value
    else:
        return converter(value)


def divide_to_patches(
    image: np.ndarray,
    grid_pinpoint: Tuple[int, int],
    input_data_format: Union[ChannelDimension, str],
) -> List[List[np.ndarray]]:
    """"""
    patches = []
    height, width = get_image_size(image, channel_dim=input_data_format)
    grid_height, grid_width = int(height / grid_pinpoint[0]), int(width / grid_pinpoint[1])
    for i in range(0, height, grid_height):
        for j in range(0, width, grid_width):
            if input_data_format == ChannelDimension.LAST:
                patch = image[i : i + grid_height, j : j + grid_width]
            else:
                patch = image[:, i : i + grid_height, j : j + grid_width]

            patches.append(patch)

    return patches


def ensure_divide(length, patch_size):
    return max(round(length / patch_size) * patch_size, patch_size)


def _get_patch_output_size(
    image_size: Tuple[int, int],
    grid_pinpoint: Tuple[int, int],
    size: Tuple[int, int],
    patch_size: int = 14,
    allow_upscale: bool = True,
) -> Tuple[int, int]:
    original_height, original_width = image_size
    target_grid_height, target_grid_width = grid_pinpoint

    refine_height = ensure_divide(original_height, target_grid_height)
    refine_width = ensure_divide(original_width, target_grid_width)

    grid_height = refine_height / target_grid_height
    grid_width = refine_width / target_grid_width

    best_height, best_width = find_best_resize(
        (grid_height, grid_width),
        size,
        patch_size,
        allow_upscale=allow_upscale,
    )

    target_resolution = (best_height * target_grid_height, best_width * target_grid_width)
    return target_resolution


def find_best_resize(
    image_size: Tuple[int, int],
    size: Tuple[int, int],
    patch_size: int,
    allow_upscale: Optional[bool] = None,
) -> Tuple[int, int]:
    height, width = image_size

    if (height * width > size[0] * size[1]) or allow_upscale:
        r = width / height
        # NOTE: height와 width를 동시에 해야하는거 아닌가? 근데 왜 이걸 번갈아 가면서 하지?
        height = int(size[0] / math.sqrt(r))
        width = int(height * r)

    best_height = ensure_divide(height, patch_size)
    best_width = ensure_divide(width, patch_size)

    return (best_height, best_width)


class MiniCPMOImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_center_crop: bool = True,
        do_rescale: bool = True,
        patch_size: int = 14,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        size = size if size is not None else {"shortest_edge": 448}
        size = get_size_dict(size, default_to_square=False)

        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.patch_size = patch_size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else [0.5, 0.5, 0.5]
        self.image_std = image_std if image_std is not None else [0.5, 0.5, 0.5]
        self.do_convert_rgb = do_convert_rgb

    # Copied from transformers.models.clip.image_processing_clip.CLIPImageProcessor.resize with CLIP->LLaVa
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image. The shortest edge of the image is resized to size["shortest_edge"], with the longest edge
        resized to keep the input aspect ratio.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        default_to_square = True
        if "shortest_edge" in size:
            size = size["shortest_edge"]
            default_to_square = False
        elif "height" in size and "width" in size:
            size = (size["height"], size["width"])
        else:
            raise ValueError("Size must contain either 'shortest_edge' or 'height' and 'width'.")

        output_size = get_resize_output_image_size(
            image,
            size=size,
            default_to_square=default_to_square,
            input_data_format=input_data_format,
        )

        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def _resize_for_patching(
        self,
        image: np.ndarray,
        image_size: Tuple[int, int],
        grid_pinpoint: Tuple[int, int],
        size: Tuple[int, int],
        resample: PILImageResampling,
        input_data_format: Union[ChannelDimension, str],
        patch_size: int = 14,
        allow_upscale: bool = True,
    ) -> np.ndarray:
        """"""
        new_height, new_width = _get_patch_output_size(image_size, grid_pinpoint, size, patch_size, allow_upscale)

        # Resize the image
        resized_image = resize(image, (new_height, new_width), resample=resample, input_data_format=input_data_format)

        return resized_image

    def find_best_grid_pinpoint(
        self,
        image_size: Tuple[int, int],
        size: Tuple[int, int],
        max_slice_num: int = 9,
        never_split: Optional[bool] = None,
    ) -> Optional[Tuple[int, int]]:
        original_height, original_width = image_size

        log_ratio = math.log(original_width / original_height)
        multiple = min(math.ceil(original_width * original_height / (size[0] * size[1])), max_slice_num)

        if multiple <= 1 or never_split:
            return [1, 1]

        candidate_split_grids_nums = []
        for i in [multiple - 1, multiple, multiple + 1]:
            if i == 1 or i > max_slice_num:
                continue
            candidate_split_grids_nums.append(i)

        candidate_grids = []
        for split_grids_nums in candidate_split_grids_nums:
            m = 1
            while m <= split_grids_nums:
                if split_grids_nums % m == 0:
                    candidate_grids.append([split_grids_nums // m, m])
                m += 1

        selected_grid = [1, 1]
        min_error = float("inf")
        for grid in candidate_grids:
            error = abs(log_ratio - math.log(grid[1] / grid[0]))
            if error < min_error:
                selected_grid = grid
                min_error = error

        return selected_grid

    def _get_image_patches(
        self,
        image: np.ndarray,
        size: tuple,
        patch_size: int,
        resample: PILImageResampling,
        data_format: ChannelDimension,
        input_data_format: ChannelDimension,
        max_slice_num: int = 9,
    ) -> List[np.ndarray]:
        """
        Process an image with variable resolutions by dividing it into patches.

        Args:
            image (np.array):
                The input image to be processed.
            grid_pinpoints (List):
                A string representation of a list of possible resolutions.
            size (`tuple`):
                Size to resize the original image to.
            patch_size (`int`):
                Size of the patches to divide the image into.
            resample (`PILImageResampling`):
                Resampling filter to use if resizing the image.
            data_format (`ChannelDimension` or `str`):
                The channel dimension format for the output image.
            input_data_format (`ChannelDimension` or `str`):
                The channel dimension format of the input image.

        Returns:
            List[np.array]: A list of NumPy arrays containing the processed image patches.
        """

        image_size = get_image_size(image, input_data_format)
        best_grid_pinpoint = self.find_best_grid_pinpoint(image_size, size, max_slice_num, never_split=False)

        resized_image = self._resize_for_patching(
            image, image_size, best_grid_pinpoint, size, resample, input_data_format, patch_size
        )
        if best_grid_pinpoint == [1, 1]:
            resized_image = to_channel_dimension_format(
                resized_image, channel_dim=data_format, input_channel_dim=input_data_format
            )
            return [resized_image]

        patches = divide_to_patches(resized_image, best_grid_pinpoint, input_data_format)

        target_size = find_best_resize(image_size, size, patch_size, allow_upscale=True)
        resized_original_image = resize(
            image,
            size=target_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
        )

        image_patches = [resized_original_image] + patches
        image_patches = [
            to_channel_dimension_format(patch, channel_dim=data_format, input_channel_dim=input_data_format)
            for patch in image_patches
        ]

        return image_patches

    def _preprocess(
        self,
        images: ImageInput,
        patch_size: int = 14,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Preprocess an image or batch of images. Copy of the `preprocess` method from `CLIPImageProcessor`.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing. Shortest edge of the image is resized to size["shortest_edge"], with
                the longest edge resized to keep the input aspect ratio.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
                has an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
                `True`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        images = make_list_of_images(images)

        all_images, target_sizes = [], []
        for image in images:
            if do_rescale:
                image = self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)

            if do_normalize:
                image = self.normalize(
                    image=image, mean=image_mean, std=image_std, input_data_format=input_data_format
                )

            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)

            height, width = get_image_size(image, data_format)

            patches_height = height // patch_size
            patches_width = width // patch_size

            target_sizes.append((patches_height, patches_width))

            # original is MiniCPMVImageProcessor.reshape_by_patch

            reshaped = image.reshape(3, patches_height, patch_size, patches_width, patch_size)
            transposed = np.transpose(reshaped, (0, 2, 1, 3, 4))
            patches = transposed.reshape(3, patch_size, -1)

            all_images.append(patches)

        return all_images, target_sizes

    def _pad_for_batching(
        self,
        pixel_values: List[np.ndarray],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Pads images on the `num_of_patches` dimension with zeros to form a batch of same number of patches.

        Args:
            pixel_values (`List[np.ndarray]`):
                An array of pixel values of each images of shape (`batch_size`, `num_patches`, `image_in_3D`)
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. Can be one of:
                    - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                    - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                If unset, will use same as the input image.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the input image. Can be one of:
                    - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                    - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                If unset, will use the inferred format of the input image.

        Returns:
            List[`np.ndarray`]: The padded images.
        """

        if data_format == ChannelDimension.FIRST:
            max_elem = max(pixel_value.shape[-1] for pixel_value in pixel_values)
        else:
            max_elem = max(pixel_value.shape[1] for pixel_value in pixel_values)

        paded_pixel_values, masks = [], []
        for pixel_value in pixel_values:
            if data_format == ChannelDimension.FIRST:
                pad_pos = ((0, 0), (0, max_elem - pixel_value.shape[-1]))
            else:
                pad_pos = ((0, max_elem - pixel_value.shape[1]), (0, 0))

            paded_pixel_value = pad(
                pixel_value,
                padding=pad_pos,
            )
            mask = np.ones_like(paded_pixel_value)

            if data_format == ChannelDimension.FIRST:
                mask[:, :, pixel_value.shape[-1] :] = 0
            else:
                mask[:, pixel_value.shape[1] :, :] = 0

            paded_pixel_values.append(paded_pixel_value)
            masks.append(mask)

        paded_pixel_values = np.stack(paded_pixel_values)
        masks = np.stack(masks)

        return paded_pixel_values, masks

    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Dict[str, int] = None,
        resample: Optional[PILImageResampling] = None,
        patch_size: Optional[Dict[str, int]] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        max_slice_num: Optional[int] = 9,
    ) -> BatchFeature:
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        size = get_size_dict(size, param_name="size", default_to_square=False)
        resample = resample if resample is not None else self.resample
        patch_size = patch_size if patch_size is not None else self.patch_size
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        images = make_flat_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        images = [to_numpy_array(image) / 255 for image in images]

        if do_rescale and is_scaled_image(images[0]):
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        pixel_values, target_sizes = [], []
        for image in images:
            # convert image into a list of patches
            # we intentially use the same data format as the input data format
            image_patches = self._get_image_patches(
                image,
                size=(size["shortest_edge"], size["shortest_edge"])
                if "shortest_edge" in size
                else (min(size["height"], size["width"]), min(size["height"], size["width"])),
                patch_size=patch_size,
                resample=resample,
                data_format=input_data_format,
                input_data_format=input_data_format,
                max_slice_num=max_slice_num,
            )

            image_patches

            image_patches, patches_sizes = self._preprocess(
                image_patches,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                data_format=data_format,
                input_data_format=input_data_format,
            )

            pixel_values.extend(image_patches)
            target_sizes.extend(patches_sizes)

        pixel_values, pixel_attention_mask = self._pad_for_batching(
            pixel_values,
            data_format=data_format,
            input_data_format=input_data_format,
        )
        target_sizes = np.array(target_sizes)

        batch_feature = BatchFeature(
            data={
                "pixel_values": pixel_values,
                "target_sizes": target_sizes,
                "pixel_attention_mask": pixel_attention_mask,
            },
            tensor_type=return_tensors,
        )
        return batch_feature
