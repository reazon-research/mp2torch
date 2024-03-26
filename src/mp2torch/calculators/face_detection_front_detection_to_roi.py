"""mediapipe/mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt"""
import torch

from mp2torch.calculators.detections_to_rects import (
    DetectionsToRectsCalculator,
    DetectionsToRectsCalculatorOptions,
)
from mp2torch.calculators.rect_transformation import (
    RectTransformationcalculator,
    RectTransformationCalculatorOptions,
)
from mp2torch.types.tensor import SegmentedTensor


class FaceDetectionFrontDetectionToROI:
    """FaceDetectionFrontDetectionTpROI
    mediapipe/mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt
    """

    def __init__(
        self, scale_x: float = 1.5, scale_y: float = 1.5, rotate: bool = False
    ) -> None:
        self.detections_to_rects = DetectionsToRectsCalculator(
            options=DetectionsToRectsCalculatorOptions(
                start_keypoint_index=0,
                end_keypoint_index=1,
                target_angle=0.0,
                rotate=rotate,
            )
        )  # mediapipe/mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt:L16-L31
        self.rect_transformation = RectTransformationcalculator(
            options=RectTransformationCalculatorOptions(
                scale_x=scale_x,
                scale_y=scale_y,
                square_long=True,
            )
        )  # mediapipe/mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt:L33-L47

    def process(
        self,
        detections: SegmentedTensor,
        image_size: torch.Tensor | tuple[int, int],
    ) -> SegmentedTensor:
        roi = self.detections_to_rects(detections, image_size)  # Normalized
        roi = self.rect_transformation(roi, image_size=image_size)  # Normalized
        return roi

    def __call__(
        self, detection: SegmentedTensor, image_size: torch.Tensor
    ) -> SegmentedTensor:
        return self.process(detection, image_size=image_size)
