"""Inspired by mediapipe/mediapipe/calculators/tensor/tensors_to_detections_calculator.cc"""
from dataclasses import dataclass
from enum import Enum

import torch

from mp2torch.utils.scores import overlap_similarity


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
    mediapipe/mediapipe/tasks/cc/components/containers/rect.h"""

    left: int
    top: int
    right: int
    bottom: int


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
class TensorMapping:
    detections_tensor_index: int
    classes_tensor_index: int
    scores_tensor_index: int
    num_detections_tensor_index: int
    anchors_tensor_index: int


@dataclass
class BoxBoundariesIndices:
    ymin: int
    xmin: int
    ymax: int
    xmax: int


class BoxFormat(Enum):
    # if UNSPECIFIED, the calculator assumes YXHW
    UNSPECIFIED = 0
    # bbox [y_center, x_center, height, width], keypoint [y, x]
    YXHW = 1
    # bbox [x_center, y_center, width, height], keypoint [x, y]
    XYWH = 2
    # bbox [xmin, ymin, xmax, ymax], keypoint [x, y]
    XYXY = 3


@dataclass
class TensorsToDetectionsCalculatorOptions:
    num_classes: int = 1
    num_anchors: int = 896
    num_coords: int = 16
    score_clipping_thresh: float = 100.0
    x_scale: float = 128.0
    y_scale: float = 128.0
    h_scale: float = 128.0
    w_scale: float = 128.0
    min_score_thresh: float = 0.75
    min_suppression_threshold: float = 0.3


class TensorsToDetectionsCalculator(object):
    """TensorsToDetectionsCalculator
    Inspired by https://github.com/hollance/BlazeFace-PyTorch/blob/master/blazeface.py
    """

    def __init__(self, options: TensorsToDetectionsCalculatorOptions) -> None:
        self.__options = options

    def _decode_boxes(self, raw_boxes: torch.Tensor, anchors: torch.Tensor):
        anchors = anchors.to(raw_boxes.device)

        # bounding box
        x_center = (
            raw_boxes[..., 0] / self.__options.x_scale * anchors[:, 2] + anchors[:, 0]
        )
        y_center = (
            raw_boxes[..., 1] / self.__options.y_scale * anchors[:, 3] + anchors[:, 1]
        )

        w = raw_boxes[..., 2] / self.__options.w_scale * anchors[:, 2]
        h = raw_boxes[..., 3] / self.__options.h_scale * anchors[:, 3]

        raw_boxes[..., 0] = y_center - h / 2.0  # ymin
        raw_boxes[..., 1] = x_center - w / 2.0  # xmin
        raw_boxes[..., 2] = y_center + h / 2.0  # ymax
        raw_boxes[..., 3] = x_center + w / 2.0  # xmax

        # keypoints
        raw_boxes[..., 4:16:2] = (
            raw_boxes[..., 4:16:2] / self.__options.x_scale * anchors[:, [2]]
            + anchors[:, [0]]
        )  # x of keypoints
        raw_boxes[..., 5:16:2] = (
            raw_boxes[..., 5:16:2] / self.__options.y_scale * anchors[:, [3]]
            + anchors[:, [1]]
        )  # y of keypoints

        return raw_boxes

    def tensors_to_detections(
        self,
        raw_box_tensor: torch.Tensor,
        raw_score_tensor: torch.Tensor,
        anchors: torch.Tensor,
    ):
        """This is based on the source code from:
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.proto"""
        detection_boxes = self._decode_boxes(raw_box_tensor, anchors)

        raw_score_tensor = raw_score_tensor.clamp(
            -self.__options.score_clipping_thresh, self.__options.score_clipping_thresh
        )
        detection_scores = raw_score_tensor.sigmoid().squeeze(dim=-1)

        # Note: we stripped off the last dimension from the scores tensor
        # because there is only has one class. Now we can simply use a mask
        # to filter out the boxes with too low confidence.
        mask = detection_scores >= self.__options.min_score_thresh

        # Because each image from the batch can have a different number of
        # detections, process them one at a time using a loop.
        output_detections = []
        for i in range(raw_box_tensor.shape[0]):
            boxes = detection_boxes[i, mask[i]]
            scores = detection_scores[i, mask[i]].unsqueeze(dim=-1)
            output_detections.append(torch.cat((boxes, scores), dim=-1))

        return output_detections

    def weighted_non_max_suppression(self, detections: torch.Tensor):
        """This is based on the source code from:
        mediapipe/calculators/util/non_max_suppression_calculator.cc
        mediapipe/calculators/util/non_max_suppression_calculator.proto"""
        if len(detections) == 0:
            return []

        output_detections = []

        # Sort the detections from highest to lowest score.
        remaining = torch.argsort(detections[:, 16], descending=True)

        while len(remaining) > 0:
            detection = detections[remaining[0]]

            # Compute the overlap between the first box and the other
            # remaining boxes. (Note that the other_boxes also include
            # the first_box.)
            first_box = detection[:4]
            other_boxes = detections[remaining, :4]
            ious = overlap_similarity(first_box, other_boxes)

            # If two detections don't overlap enough, they are considered
            # to be from different faces.
            mask = ious > self.__options.min_suppression_threshold
            overlapping = remaining[mask]
            remaining = remaining[~mask]

            # Take an average of the coordinates from the overlapping
            # detections, weighted by their confidence scores.
            weighted_detection = detection.clone()
            if len(overlapping) > 1:
                coordinates = detections[overlapping, :16]
                scores = detections[overlapping, 16:17]
                total_score = scores.sum()
                weighted = (coordinates * scores).sum(dim=0) / total_score
                weighted_detection[:16] = weighted
                # weighted_detection[16] = total_score / len(overlapping)
                weighted_detection[16] = detection[-1]  # score keeps original score

            output_detections.append(weighted_detection)

        return output_detections
