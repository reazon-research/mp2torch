import math
from dataclasses import dataclass, field
from enum import Enum

import einops
import torch
import torchvision

from mp2torch.types.tensor import SegmentedTensor

if torchvision.__version__ >= "0.16.0":
    import torchvision.transforms.v2.functional as F
else:
    import torchvision.transforms.functional as F


class BorderMode(Enum):
    """BorderMode
    Pixel extrapolation methods. See @border_mode."""

    BORDER_UNSPECIFIED = 0
    BORDER_ZERO = 1
    BORDER_REPLICATE = 2


@dataclass
class ImageToTensorCalculatorOptions:
    # The width and height of output tensor. The output tensor would have the
    # input image width/height if not set.
    output_tensor_width: int
    output_tensor_height: int

    output_tensor_float_range: dict[str, float] = field(
        default_factory={"min": 0.0, "max": 1.0}
    )

    # If true, image region will be extracted and copied into tensor keeping region aspect ratio,
    # which usually results in letterbox padding. Otherwise,
    # if false, image region is stretched to fill output tensor fully.
    keep_aspect_ratio: bool = True

    # BORDER_REPLICATE is used by default.
    border_mode: int = 2


class ImageToTensorCalculator(object):
    def __init__(self, options: ImageToTensorCalculatorOptions) -> None:
        self.__options = options

    def get_roi(
        self,
        input_width: int,
        input_height: int,
        batch_size: int | None = None,
        roi: torch.Tensor | SegmentedTensor | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor | SegmentedTensor:
        """mediapipe/mediapipe/calculators/tensor/image_to_tensor_utils.cc::GetRoi
        Returns
        -------
        torch.Tensor
            [[x_center, y_center, width, height, rotation], ...]
        """
        if roi is not None:
            roi = roi.clone()
            roi[..., 0] = roi[..., 0] * input_width  # x_center
            roi[..., 1] = roi[..., 1] * input_height  # y_center
            roi[..., 2] = roi[..., 2] * input_width  # width
            roi[..., 3] = roi[..., 3] * input_height  # height
            return roi
        else:
            return einops.repeat(
                torch.tensor(
                    [
                        0.5 * input_width,
                        0.5 * input_height,
                        input_width,
                        input_height,
                        0.0,
                    ],
                    device=device,
                ),
                "f -> b f",
                b=batch_size,
            ).clone()

    def get_rotated_subrect_to_rect_transform_matrix(
        self,
        roi: torch.Tensor,
        width: int,
        height: int,
        flip_horizontally: bool = False,
    ) -> torch.Tensor:
        """mediapipe/mediapipe/calculators/tensor/image_to_tensor_utils.cc::GetRotatedSubRectToRectTransformMatrix
        roi: [x_center, y_center, width, height]"""
        matrix = torch.zeros((roi.size(0), 16), device=roi.device)
        # The resulting matrix is multiplication of below commented out matrices:
        #   post_scale_matrix
        #     * translate_matrix
        #     * rotate_matrix
        #     * flip_matrix
        #     * scale_matrix
        #     * initial_translate_matrix

        # Matrix to convert X,Y to [-0.5, 0.5] range "initial_translate_matrix"
        # { 1.0f,  0.0f, 0.0f, -0.5f}
        # { 0.0f,  1.0f, 0.0f, -0.5f}
        # { 0.0f,  0.0f, 1.0f,  0.0f}
        # { 0.0f,  0.0f, 0.0f,  1.0f}

        a = roi[..., 2]
        b = roi[..., 3]
        # Matrix to scale X,Y,Z to sub rect "scale_matrix"
        # Z has the same scale as X.
        # {   a, 0.0f, 0.0f, 0.0f}
        # {0.0f,    b, 0.0f, 0.0f}
        # {0.0f, 0.0f,    a, 0.0f}
        # {0.0f, 0.0f, 0.0f, 1.0f}

        flip = -1 if flip_horizontally else 1
        # Matrix for optional horizontal flip around middle of output image.
        # { fl  , 0.0f, 0.0f, 0.0f}
        # { 0.0f, 1.0f, 0.0f, 0.0f}
        # { 0.0f, 0.0f, 1.0f, 0.0f}
        # { 0.0f, 0.0f, 0.0f, 1.0f}

        c = torch.cos(roi[..., 4])
        d = torch.sin(roi[..., 4])
        # Matrix to do rotation around Z axis "rotate_matrix"
        # {    c,   -d, 0.0f, 0.0f}
        # {    d,    c, 0.0f, 0.0f}
        # { 0.0f, 0.0f, 1.0f, 0.0f}
        # { 0.0f, 0.0f, 0.0f, 1.0f}

        e = roi[..., 0]
        f = roi[..., 1]
        # Matrix to do X,Y translation of sub rect within parent rect
        # "translate_matrix"
        # {1.0f, 0.0f, 0.0f, e   }
        # {0.0f, 1.0f, 0.0f, f   }
        # {0.0f, 0.0f, 1.0f, 0.0f}
        # {0.0f, 0.0f, 0.0f, 1.0f}

        g = 1.0 / width
        h = 1.0 / height
        # Matrix to scale X,Y,Z to [0.0, 1.0] range "post_scale_matrix"
        # {g,    0.0f, 0.0f, 0.0f}
        # {0.0f, h,    0.0f, 0.0f}
        # {0.0f, 0.0f,    g, 0.0f}
        # {0.0f, 0.0f, 0.0f, 1.0f}

        # row 1
        matrix[..., 0] = a * c * flip * g
        matrix[..., 1] = -b * d * g
        matrix[..., 2] = 0.0
        matrix[..., 3] = (-0.5 * a * c * flip + 0.5 * b * d + e) * g

        # row 2
        matrix[..., 4] = a * d * flip * h
        matrix[..., 5] = b * c * h
        matrix[..., 6] = 0.0
        matrix[..., 7] = (-0.5 * b * c - 0.5 * a * d * flip + f) * h

        # row 3
        matrix[..., 8] = 0.0
        matrix[..., 9] = 0.0
        matrix[..., 10] = a * g
        matrix[..., 11] = 0.0

        # row 4
        matrix[..., 12] = 0.0
        matrix[..., 13] = 0.0
        matrix[..., 14] = 0.0
        matrix[..., 15] = 1.0
        return matrix

    def pad_roi(
        self,
        input_tensor_width: int,
        input_tensor_height: int,
        keep_aspect_ratio: bool,
        roi: torch.Tensor,
    ) -> tuple[int, int]:
        """mediapipe/mediapipe/calculators/tensor/image_to_tensor_utils.cc::PadRoi"""
        if not keep_aspect_ratio:
            return (0, 0)
        width = roi[0][2]
        height = roi[0][3]
        tensor_aspect_ratio = input_tensor_height / input_tensor_width
        roi_aspect_ratio = height / width
        vertical_padding = 0
        horizontal_padding = 0
        if tensor_aspect_ratio > roi_aspect_ratio:
            vertical_padding = (1.0 - roi_aspect_ratio / tensor_aspect_ratio) / 2
            vertical_padding = int(vertical_padding * width * tensor_aspect_ratio)
        else:
            horizontal_padding = (1.0 - tensor_aspect_ratio / roi_aspect_ratio) / 2
            horizontal_padding = int(horizontal_padding * height / tensor_aspect_ratio)
        return horizontal_padding, vertical_padding  # left/right, top/bottom

    def convert(
        self,
        images: torch.Tensor,
        roi: torch.Tensor,
        range_min: float,
        range_max: float,
    ) -> torch.Tensor:
        """mediapipe/mediapipe/calculators/tensor/image_to_tensor_converter_frame_buffer.cc"""
        raise NotImplementedError

    def process(
        self,
        images: torch.Tensor | SegmentedTensor,  # IMAGE
        roi: torch.Tensor | SegmentedTensor | None = None,  # NORM_RECT
    ) -> tuple[torch.Tensor | SegmentedTensor, torch.Tensor, SegmentedTensor]:
        """
        roi: [x_center, y_center, width, height, rotation(radian)]
        """
        input_roi = roi is not None
        if images.ndim == 3:
            images = images.unsqueeze(dim=0)
        if images.size(-1) == 3:
            images = einops.rearrange(images, "b h w c -> b c h w")

        diff = (
            self.__options.output_tensor_float_range["max"]
            - self.__options.output_tensor_float_range["min"]
        )
        if images.dtype != torch.float:
            images = images.to(torch.float32)
        else:
            if (images >= self.__options.output_tensor_float_range["min"]).all() and (
                images <= self.__options.output_tensor_float_range["max"]
            ).all():
                return einops.rearrange(images, "b c h w -> b h w c")
        B, _, H, W = images.size()
        # get roi of real scale
        roi = self.get_roi(
            input_width=W, input_height=H, batch_size=B, roi=roi, device=images.device
        )
        padding = self.pad_roi(
            input_tensor_width=self.__options.output_tensor_width,
            input_tensor_height=self.__options.output_tensor_height,
            keep_aspect_ratio=self.__options.keep_aspect_ratio,
            roi=roi,
        )  # calculate padding
        images = F.pad(
            images, padding=padding, fill=0, padding_mode="constant"
        )  # padding to keep aspect ratio

        if input_roi:
            images = [
                torch.stack(
                    [
                        F.resize(
                            F.crop(
                                F.rotate(
                                    image,
                                    angle=rotation * 180 / math.pi,  # rad -> deg
                                    center=(int(x_center), int(y_center)),
                                    interpolation=F.InterpolationMode.BILINEAR,
                                ),
                                top=int(y_center - height / 2),
                                left=int(x_center - width / 2),
                                width=int(width),
                                height=int(height),
                            ),
                            size=(
                                self.__options.output_tensor_height,
                                self.__options.output_tensor_width,
                            ),
                            interpolation=F.InterpolationMode.BILINEAR,
                        )
                        for (
                            x_center,
                            y_center,
                            width,
                            height,
                            rotation,
                        ) in regions.tolist()
                    ]
                )
                for image, regions in zip(images, roi.get_all_segments())
                if regions.nelement() > 0
            ]
            images = SegmentedTensor(tensors=images, segments=roi.segments)
            images = (
                images / 255.0 * diff + self.__options.output_tensor_float_range["min"]
            )
            matrix = self.get_rotated_subrect_to_rect_transform_matrix(
                roi, width=W, height=H, flip_horizontally=False
            )
            return einops.rearrange(images, "b c h w -> b h w c"), matrix

        roi[..., 2] = images.size(3)  # width after padding
        roi[..., 3] = images.size(2)  # height after padding
        matrix = self.get_rotated_subrect_to_rect_transform_matrix(
            roi, width=W, height=H, flip_horizontally=False
        )
        images = F.resize(
            images,
            (
                self.__options.output_tensor_height,
                self.__options.output_tensor_width,
            ),
            interpolation=F.InterpolationMode.NEAREST,
        )  # resize
        images = images / 255.0 * diff + self.__options.output_tensor_float_range["min"]
        return einops.rearrange(images, "b c h w -> b h w c"), matrix

    def __call__(
        self, images: torch.Tensor, roi: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.process(images=images, roi=roi)
