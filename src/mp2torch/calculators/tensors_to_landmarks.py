"""mediapipe/mediapipe/calculators/tensor/tensors_to_landmarks_calculator.cc
mediapipe/mediapipe/calculators/tensor/tensors_to_landmarks_calculator.cc"""

from dataclasses import dataclass

import torch


@dataclass
class TensorsToLandmarksOptions:
    """TensorsToLandmarksOptions
    mediapipe/mediapipe/calculators/tensor/tensors_to_landmarks_calculator.proto"""

    num_landmarks: int
    input_image_width: int
    input_image_height: int
    flip_vertically: bool = False
    normalize_z: float = 1.0
    visibility_activation: str | None = None
    presence_activation: str | None = None


@dataclass
class Landmark:
    x: float
    y: float
    z: float


class TensorsToLandmarks:
    """TensorsToLandmarks
    mediapipe/mediapipe/calculators/tensor/tensors_to_landmarks_calculator.cc"""

    def __init__(self, options: TensorsToLandmarksOptions) -> None:
        self.__options = options

    def process(self, tensors: torch.Tensor):
        pass
