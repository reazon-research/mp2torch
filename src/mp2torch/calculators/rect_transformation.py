"""mediapipe/mediapipe/calculators/util/rect_transformation_calculator.proto
mediapipe/mediapipe/calculators/util/rect_transformation_calculator.cc"""
from dataclasses import dataclass

import torch


@dataclass
class RectTransformationCalculatorOptions:
    """RectTransformationCalculatorOptions
    mediapipe/mediapipe/calculators/util/rect_transformation_calculator.proto"""

    scale_x: float = 1.0
    scale_y: float = 1.0
    rotation: float | None = None
    rotation_degrees: int | None = None
    shift_x: float | None = None
    shift_y: float | None = None
    square_long: bool | None = None
    square_short: bool | None = None


class RectTransformationcalculator:
    """RectTransformationcalculator
    mediapipe/mediapipe/calculators/util/rect_transformation_calculator.cc"""

    def __init__(self, options: RectTransformationCalculatorOptions) -> None:
        self.__options = options

    def transform_rect(
        self, rects: torch.Tensor, image_width: int, image_height: int
    ) -> torch.Tensor:
        """rects: [x_center, y_center, width, height, rotation] before rotation transformation
        where values are normalized scale"""
        rotation = rects[..., 4]  # degrees
        if (
            self.__options.rotation is not None
            or self.__options.rotation_degrees is not None
        ):
            rotation = self.compute_new_rotation(rotation=rotation)  # -PI to PI

        # rects[..., 2] = torch.abs(rects[..., 2] / torch.cos(rotation / 180.0))  # width
        # rects[..., 3] = torch.abs(rects[..., 3] / torch.cos(rotation / 180.0))  # height
        width = rects[..., 2] * image_width  # rescale
        height = rects[..., 3] * image_height  # rescale

        if self.__options.square_long:
            long_side = torch.max(width, height)
            width = long_side
            height = long_side
        elif self.__options.square_short:
            short_side = torch.min(width, height)
            width = short_side
            height = short_side
        rects[..., 2] = width * self.__options.scale_x / image_width  # normalize
        rects[..., 3] = height * self.__options.scale_y / image_height  # normalize
        return rects

    def compute_new_rotation(self, rotation: torch.Tensor) -> torch.Tensor:
        if self.__options.rotation is not None:
            rotation += self.__options.rotation
        elif self.__options.rotation_degrees is not None:
            rotation += torch.pi * self.__options.rotation_degrees / 180.0
        return self.normalize_radians(rotation)

    def normalize_radians(self, angle: torch.Tensor) -> torch.Tensor:
        """-PI to PI"""
        return angle - 2 * torch.pi * torch.floor(
            (angle - (-torch.pi)) / (2 * torch.pi)
        )

    def process(
        self, rects: torch.Tensor, image_size: torch.Tensor | tuple[int, int]
    ) -> torch.Tensor:
        return self.transform_rect(
            rects=rects, image_height=image_size[0], image_width=image_size[1]
        )

    def __call__(self, rects: torch.Tensor, image_size: torch.Tensor) -> torch.tensor:
        return self.process(rects=rects, image_size=image_size)
