"""mediapipe/mediapipe/calculators/util/detections_to_rects_calculator.cc"""
import math
from dataclasses import dataclass
from enum import Enum

import torch

from mp2torch.types.tensor import SegmentedTensor


class ConversionMode(Enum):
    DEFAULT = 0
    USE_BOUNDING_BOX = 1
    USE_KEYPOINTS = 2


@dataclass
class Category:
    """Category
    mediapipe/mediapipe/tasks/cc/components/containers/category.h"""

    index: int
    score: float
    category_name: str | None = None
    display_name: str | None = None


@dataclass
class Rect:
    """Rect
    mediapipe/mediapipe/framework/formats/rect.proto"""

    x_center: int
    y_center: int
    height: int
    width: int
    rotation: float = 0.0
    rect_id: int | None = None


@dataclass
class NormalizedRect:
    """NormalizedRect
    mediapipe/mediapipe/framework/formats/rect.protoh"""

    x_center: float
    y_center: float
    height: float
    width: float
    rotation: float = 0.0  # angle is clockwise in radians
    rect_id: int | None = None


@dataclass
class NormalizedKeypoint:
    """NormalizedKeypoint
    mediapipe/mediapipe/tasks/cc/components/containers/keypoint.h"""

    x: float
    y: float
    label: str | None = None
    score: float | None = None


@dataclass
class Detection:
    """Detection
    mediapipe/mediapipe/tasks/cc/components/containers/detection_result.h"""

    categories: list[Category]
    bbox: Rect
    keypoints: list[NormalizedKeypoint] | None = None


@dataclass
class DetectionsToRectsCalculatorOptions:
    start_keypoint_index: int
    end_keypoint_index: int
    target_angle: float = 0.0
    rotate: bool = False
    output_zero_rect_for_empty_detections: bool = True
    conversion_mode: int = ConversionMode.DEFAULT


class DetectionsToRectsCalculator(object):
    def __init__(self, options: DetectionsToRectsCalculatorOptions) -> None:
        self.__options = options

    def get_image_size(self, images: torch.Tensor):
        if images.size(-1) == 3:  # [..., H, W, C]
            return images.size()[-3:-1]
        return images.size()[-2:]  # [..., C, H, W]

    def compute_rotation(
        self,
        detections: SegmentedTensor,  # detections from BlazeFace
        detection_spec: torch.Tensor | tuple,  # [height, width]
    ) -> SegmentedTensor:  # radian
        x0 = (
            detections[..., [4 + self.__options.start_keypoint_index * 2]]
            * detection_spec[1]
        )  # left eye
        y0 = (
            detections[..., [4 + self.__options.start_keypoint_index * 2 + 1]]
            * detection_spec[0]
        )  # left eye
        x1 = (
            detections[..., [4 + self.__options.end_keypoint_index * 2]]
            * detection_spec[1]
        )  # right eye
        y1 = (
            detections[..., [4 + self.__options.end_keypoint_index * 2 + 1]]
            * detection_spec[0]
        )  # right eye
        return self.normalize_radians(
            self.__options.target_angle - torch.atan2(-(y1 - y0), x1 - x0)
        )  # rad

    def rect_from_box(
        self, box: SegmentedTensor
    ) -> SegmentedTensor:  # [[x_center, y_center, width, height], ...]
        # rects = torch.zeros_like(box, device=box.device)
        rects = box.clone()
        rects[..., 0] = box[..., 1] + (box[..., 3] - box[..., 1]) / 2  # x_center
        rects[..., 1] = box[..., 0] + (box[..., 2] - box[..., 0]) / 2  # y_center
        rects[..., 2] = box[..., 3] - box[..., 1]  # width
        rects[..., 3] = box[..., 2] - box[..., 0]  # height
        return rects

    def detection_to_rect(
        self, detections: torch.Tensor, detection_spec: torch.Tensor
    ) -> torch.Tensor:
        if self.__options.conversion_mode in (
            ConversionMode.DEFAULT,
            ConversionMode.USE_BOUNDING_BOX,
        ):
            return self.rect_from_box(box=detections[..., :4])
        else:
            raise NotImplementedError

    def normalize_radians(
        self, angle: SegmentedTensor | torch.Tensor
    ) -> SegmentedTensor | torch.Tensor:
        return angle - 2 * torch.pi * torch.floor((angle - (-torch.pi)) / (2 * math.pi))

    def process(
        self,
        detections: SegmentedTensor,
        detection_spec: torch.Tensor,
    ) -> SegmentedTensor:
        if len(detections) == 0:
            if self.__options.output_zero_rect_for_empty_detections:
                return []
            raise NotImplementedError

        if self.__options.rotate:
            rotations = self.compute_rotation(
                detections=detections, detection_spec=detection_spec
            )
            rects = self.rect_from_box(box=detections[..., :4])
            # set rotation
            return torch.cat(
                [rects, rotations], dim=-1
            )  # [[x_center, y_center, width, height, rotation(radian)], ...]
        else:
            return torch.cat(
                [
                    self.rect_from_box(detections[..., :4]),
                    torch.zeros((detections.size(0), 1)),
                ],
                dim=-1,
            )

    def __call__(
        self,
        detections: SegmentedTensor,
        detection_spec: torch.Tensor,
    ) -> SegmentedTensor:
        return self.process(detections, detection_spec)
