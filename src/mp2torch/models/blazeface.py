from dataclasses import dataclass

import einops
import numpy as np
import torch
import torch.nn as nn

import onnx2torch
from mp2torch.calculators.image_to_tensor import (
    ImageToTensorCalculator,
    ImageToTensorCalculatorOptions,
)
from mp2torch.calculators.ssd_anchors_calculator import SsdAnchorsCalculator
from mp2torch.calculators.tensors_to_detections import (
    TensorsToDetectionsCalculator,
    TensorsToDetectionsCalculatorOptions,
)
from mp2torch.models.wrapper import BatchedReshape
from mp2torch.types.tensor import SegmentedTensor


@dataclass
class OutputBlazeFace:
    filtered_detections: list[torch.Tensor] | list[np.ndarray] | list[
        list[int, float]
    ] | SegmentedTensor
    input_images: np.ndarray | None = None  # [B, H, W, (BGR)]


class FaceDetectionShortRange(nn.Module):
    def __init__(self, onnx_path: str) -> None:
        """FaceDetectionShortRange
        FaceDetectionShortRange model without any preprocessing or postprocessing

        Parameters
        ----------
        onnx_path: str
            Path of short range face detection model converted from .tflite to onnx
        """
        super().__init__()
        self.backbone = onnx2torch.convert(onnx_path)
        reshapes = [attr for attr in dir(self.backbone) if "reshape" in attr]
        for attr in reshapes:
            setattr(
                self.backbone,
                attr,
                BatchedReshape(attr, getattr(self.backbone, attr)),
            )
    @property
    def device(self):
        return self.backbone.classificator_8.weight.device

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        inputs: torch.Tensor
            Batched input images in the form of [B, C, H, W]

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Bounding boxes and scores
        """
        return self.backbone(inputs)


class BlazeFace(nn.Module):
    def __init__(
        self,
        onnx_path: str,
        min_score_threshold: float = 0.5,
        width: int = 128,
        height: int = 128,
        rescale: bool = True,
        mode: str = "short",
    ) -> None:
        """BlazeFace
        Parameters
        ----------
        onnx_path: str
            Path of face detection model converted from .tflite to onnx
        min_score_threshold: float, default=0.5
            Threshold of detection confidence by the face detection model
        width: int, default=128
            Image width to be resized before feeding into the face detection model
        height: int, default=128
            Image width to be resized before feeding into the face detection model
        rescale: bool, default=True
            Whether to scale the output range [0.0, 1.0] to [0, (W | H)], where W and H are the input image width and height, respectively.
        mode: str, default='short'
            short means short range detection and long means long range detection.
            You will not use long mode since the long model conversion from .tflite to onnx probably fails.
        """
        super().__init__()
        self.image_to_tensor = ImageToTensorCalculator(
            options=ImageToTensorCalculatorOptions(
                output_tensor_width=width,
                output_tensor_height=height,
                output_tensor_float_range={"min": -1.0, "max": 1.0},
                keep_aspect_ratio=True,
            )
        )  # mediapipe/mediapipe/modules/face_detection/face_detection.pbtxt:L41-L60
        self.anchors_calculator = SsdAnchorsCalculator()
        self.blazeface = onnx2torch.convert(onnx_path)
        self.tensors_to_detections = TensorsToDetectionsCalculator(
            options=TensorsToDetectionsCalculatorOptions(
                min_score_thresh=min_score_threshold,
                x_scale=float(width),
                y_scale=float(height),
                h_scale=float(height),
                w_scale=float(width),
            )
        )
        self.rescale = rescale

        self._get_anchors(mode=mode)
        reshapes = [attr for attr in dir(self.blazeface) if "reshape" in attr]
        for attr in reshapes:
            setattr(
                self.blazeface,
                attr,
                BatchedReshape(attr, getattr(self.blazeface, attr)),
            )

    def _get_anchors(self, mode: str = "short") -> None:
        self.anchors = self.anchors_calculator.generate_anchors(
            options=self.anchors_calculator.options[mode]
        )

    @property
    def device(self):
        return self.blazeface.classificator_8.weight.device

    def forward(
        self,
        images: torch.Tensor,
        rescale: bool | None = None,
        return_tensors: str | None = None,
        return_device: str | None = None,
        output_input_images: bool | None = None,
    ) -> OutputBlazeFace:
        """
        Parameters
        ----------
        images: torch.Tensor
            Batched images
        rescale: bool, optional
            Whether to scale the output range [0.0, 1.0] to [0, (W | H)], where W and H are the input image width and height, respectively.
            If rescale is not given, self.rescale will be used.
        return_tensors: str, optional
            What type to output tensors.
            You can use np|numpy, pt|torch, and list, and tensors of mp2torch.types.SegmentedTesnor will returns if `return_tensors` is not given.
        return_device: str, optional
            What device to output tensors.
            When `cpu` is given, output tensors are forced to transer to cpu and then return.
        output_input_images: bool, optional
            Wheater to output the input images or not.

        Returns
        -------
        OutputBlazeFace
        """
        if images.size(-1) == 3:  # channel-last
            _, H, W, _ = images.size()
        else:
            _, _, H, W = images.size()
        inputs, transform_matrix = self.image_to_tensor(images=images)
        with torch.no_grad():
            bboxes, scores = self.blazeface(inputs)

        detections = self.tensors_to_detections.tensors_to_detections(
            bboxes, scores, self.anchors
        )

        filtered_detections = []
        for i in range(len(detections)):
            faces = self.tensors_to_detections.weighted_non_max_suppression(
                detections[i]
            )
            faces = (
                torch.stack(faces)
                if len(faces) > 0
                else torch.empty((0, 17), device=detections[i].device)
            )
            topk = faces[..., -1].topk(k=faces.size(0))  # indices for sorting
            filtered_detections.append(faces[topk.indices])  # sorting by confidence

        filtered_detections = SegmentedTensor(filtered_detections)
        transform_matrix = self._ajust_transform_matrix(
            detections=filtered_detections, matrix=transform_matrix
        )
        filtered_detections = self._project_detections(
            detections=filtered_detections, matrix=transform_matrix
        )

        if rescale or self.rescale:
            filtered_detections = self._rescale_bbox(filtered_detections, h=H, w=W)

        if return_device == "cpu":
            filtered_detections = filtered_detections.cpu()

        if return_tensors == "np" or return_tensors == "numpy":
            filtered_detections = [
                detections.cpu().numpy()
                for detections in filtered_detections.get_all_segments()
            ]
        elif return_tensors == "list":
            filtered_detections = [
                detections.tolist()
                for detections in filtered_detections.get_all_segments()
            ]
        elif return_tensors == "pt" or return_tensors == "torch":
            filtered_detections = [detections for detections in filtered_detections]

        input_images = None
        if output_input_images:
            if images.size(-1) != 3:
                images = einops.rearrange(images, "b c h w -> b h w c")
            input_images = images[..., [2, 1, 0]].detach().cpu().numpy()
        return OutputBlazeFace(
            filtered_detections=filtered_detections,
            input_images=input_images,
        )

    def _ajust_transform_matrix(
        self, detections: SegmentedTensor, matrix: torch.Tensor
    ) -> torch.Tensor:
        new_matrix = []
        for i, segment in enumerate(detections.get_all_segments()):
            size = segment.size(0)
            new_matrix.append(einops.repeat(matrix[i], "f -> b f", b=size))
        return SegmentedTensor(tensors=new_matrix)

    def _project_detections(
        self, detections: torch.Tensor, matrix: torch.Tensor
    ) -> torch.Tensor:
        """mediapipe/mediapipe/calculators/util/detection_projection_calculator.cc"""
        projections = detections.clone()
        projections[..., [1, 3]] = (
            detections[..., [1, 3]] * matrix[..., [0]]
            + detections[..., [0, 2]] * matrix[..., [1]]
            + matrix[..., [3]]
        )  # xmin, xmax
        projections[..., [0, 2]] = (
            detections[..., [1, 3]] * matrix[..., [4]]
            + detections[..., [0, 2]] * matrix[..., [5]]
            + matrix[..., [7]]
        )  # ymin, ymax
        projections[..., 4:16:2] = (
            detections[..., 4:16:2] * matrix[..., [0]]
            + detections[..., 5:16:2] * matrix[..., [1]]
            + matrix[..., [3]]
        )  # x of keypoints
        projections[..., 5:16:2] = (
            detections[..., 4:16:2] * matrix[..., [4]]
            + detections[..., 5:16:2] * matrix[..., [5]]
            + matrix[..., [7]]
        )  # y of keypoints
        return projections

    def _rescale_bbox(self, detections: torch.Tensor, h: int, w: int) -> torch.Tensor:
        assert detections.ndim == 2
        detections[..., [0, 2]] = detections[..., [0, 2]] * h
        detections[..., [1, 3]] = detections[..., [1, 3]] * w
        detections[..., 4:16:2] = detections[..., 4:16:2] * w
        detections[..., 5:16:2] = detections[..., 5:16:2] * h
        return detections

    def _convert_coordinates(
        self, detections: torch.Tensor, images: torch.Tensor
    ) -> torch.Tensor:  # [ymin, xmin, ymax, xmax, x, y, ..., score]
        H, W = images.size()[-2:]
        max_size = max(H, W)
        height_norm_half = (detections[..., 2] - detections[..., 0]) / 2  # [0, 1]
        width_norm_half = (detections[..., 3] - detections[..., 1]) / 2  # [0, 1]
        y_center = detections[..., 2] - height_norm_half
        x_center = detections[..., 3] - width_norm_half
        detections[..., 0] = y_center - height_norm_half * max_size / H  # ymin
        detections[..., 2] = y_center + height_norm_half * max_size / H  # ymax
        detections[..., 1] = x_center - width_norm_half * max_size / W  # xmin
        detections[..., 3] = x_center + width_norm_half * max_size / W  # xmax
        return detections
