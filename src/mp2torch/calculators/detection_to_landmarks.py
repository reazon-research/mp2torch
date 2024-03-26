"""mediapipe/mediapipe/calculators/util/detection_to_landmarks_calculator.cc"""
import torch


class DetectionToLandmarkscalculator:
    def __init__(self) -> None:
        pass

    def convert_detection_to_landmarks(self, detections: torch.Tensor):
        """
        Parameters
        ----------
        detection: torch.Tensor
            The outputs from BlazeFace

        Returns
        -------
        torch.Tensor
        """
        return detections[..., 4:-1]

    def __call__(self, detections: torch.Tensor) -> torch.Tensor:
        return self.convert_detection_to_landmarks(detections)
