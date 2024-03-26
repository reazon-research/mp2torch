import warnings
from dataclasses import dataclass

import torch
import torch.nn as nn

import onnx2torch
from mp2torch.calculators.clip_vector_size import (
    ClipVectorSizeCalculator,
    ClipVectorSizeCalculatorOptions,
)
from mp2torch.calculators.face_detection_front_detection_to_roi import (
    FaceDetectionFrontDetectionToROI,
)
from mp2torch.calculators.image_to_tensor import (
    ImageToTensorCalculator,
    ImageToTensorCalculatorOptions,
)
from mp2torch.calculators.tensors_to_floats import (
    TensorsToFloatsCalculator,
    TensorsToFloatsCalculatorOptions,
)
from mp2torch.calculators.thresholding import (
    Thresholdingcalculator,
    ThresholdingcalculatorOptions,
)
from mp2torch.models.blazeface import BlazeFace, OutputBlazeFace
from mp2torch.models.wrapper import BatchedReshape
from mp2torch.types.tensor import SegmentedTensor
from mp2torch.utils.scores import overlap_similarity


@dataclass
class OutputFaceLandmarker:
    face_rect_from_detections: SegmentedTensor
    face_detection_scores: SegmentedTensor
    landmarks: SegmentedTensor
    face_ids: SegmentedTensor | None = None
    face_landmarks: SegmentedTensor | None = None


class FaceMesh(nn.Module):
    def __init__(self, onnx_path: str) -> None:
        """FaceMesh
        FaceMesh model without any preprocessing or postprocessing

        Parameters
        ----------
        onnx_path: str
            Path of facemesh model converted from .tflite to onnx
        """
        super().__init__()
        self.backbone = onnx2torch.convert(onnx_path)
        self.backbone.conv2d_31__203 = BatchedReshape(
            name="conv2d_31__203", module=self.backbone.conv2d_31__203
        )
        self.backbone.conv2d_21__241 = BatchedReshape(
            name="conv2d_21__241", module=self.backbone.conv2d_21__241
        )

    @property
    def device(self):
        return self.backbone.conv2d_21.weight.device

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        inputs: torch.Tensor
            Input mouth roi images whose size is [B, C, 192, 192]

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Face landmarks and face flag tensors whose range is [0.0, 1.0]
        """
        return self.backbone(inputs)


class FaceLandmarker(nn.Module):
    def __init__(
        self,
        onnx_path: str,
        onnx_path_face_detection_short_range: str | None = None,
        static_image_mode: bool = True,
        max_num_faces: int = 1,  # -1 means all faces detection
        refine_landmarks: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        scale_x: float = 1.5,
        scale_y: float = 1.5,
        rotate: bool = True,
    ) -> None:
        """FaceLandmarker
        Parameters
        ----------
        onnx_path: str
            Path of fashmesh model converted from .tflite to onnx
        onnx_path_face_detection_short_range: str, optional
            Path of face detection short range model converted from .tflite to onnx
        static_image_mode: bool, default=True
            Behavior for `static_image_mode=False` is unstable, so `static_image_mode=False` is deprecated.
        max_num_faces: int, default=1
            The maximum number of detected faces and -1 means that all faces are detected.
        refine_landmarks: bool, default=False
            Wheather to use the model refining landmarks or not.
        min_detection_confidence: float, default=0.5
            Threshold of detection confidence by the face detection short range model
        min_tracking_confidence: float, default=0.5
            Threshold of fashmesh confidence by the fashmesh model
        scale_x: float, default=1.5
        scale_y: float, default=1.5
        rotate: bool, default=True
        """
        super().__init__()
        self.image_to_tensor = ImageToTensorCalculator(
            ImageToTensorCalculatorOptions(
                output_tensor_width=192,
                output_tensor_height=192,
                output_tensor_float_range={"min": 0.0, "max": 1.0},
                keep_aspect_ratio=False,
            )
        )  # mediapipe/mediapipe/modules/face_landmark/face_landmark_cpu.pbtxt:L46-L61
        self.tensors_to_floats = TensorsToFloatsCalculator(
            options=TensorsToFloatsCalculatorOptions(activation="sigmoid")
        )  # mediapipe/mediapipe/modules/face_landmark/face_landmark_cpu.pbtxt:L126-L135
        self.thresholding = Thresholdingcalculator(
            options=ThresholdingcalculatorOptions(threshold=min_tracking_confidence)
        )  # mediapipe/mediapipe/modules/face_landmark/face_landmark_cpu.pbtxt:L139-L148
        self.face_detection_model = BlazeFace(
            onnx_path=onnx_path_face_detection_short_range,
            min_score_threshold=min_detection_confidence,
            width=128,
            height=128,
            rescale=False,
            mode="short",
        )  # mediapipe/mediapipe/modules/face_landmark/face_landmark_front_cpu.pbtxt:L101-L106
        self.clip_detection_vector_size = ClipVectorSizeCalculator(
            options=ClipVectorSizeCalculatorOptions(max_vec_size=max_num_faces)
        )  # mediapipe/mediapipe/modules/face_landmark/face_landmark_front_cpu.pbtxt:L108-L114
        self.face_detection_front_detection_to_roi = FaceDetectionFrontDetectionToROI(
            scale_x=scale_x, scale_y=scale_y, rotate=rotate
        )  # mediapipe/mediapipe/modules/face_landmark/face_landmark_front_cpu.pbtxt:L137-L144

        self.facemesh = onnx2torch.convert(onnx_path)
        self.facemesh.conv2d_31__203 = BatchedReshape(
            name="conv2d_31__203", module=self.facemesh.conv2d_31__203
        )
        self.facemesh.conv2d_21__241 = BatchedReshape(
            name="conv2d_21__241", module=self.facemesh.conv2d_21__241
        )
        self.max_num_faces = max_num_faces
        self.use_prev_landmarks = not static_image_mode
        self.with_attention = refine_landmarks
        self.min_tracking_confidence = min_tracking_confidence

    @property
    def device(self):
        return self.facemesh.conv2d_21.weight.device

    def associate_face_rects(
        self,
        face_rect_from_detection: SegmentedTensor,
        prev_face_rect_from_detection: SegmentedTensor,
        prev_face_rect_ids: SegmentedTensor,
    ) -> tuple[SegmentedTensor, SegmentedTensor]:
        if len(face_rect_from_detection) != len(prev_face_rect_from_detection):
            raise ValueError

        face_rect_ids = []
        for face_rect, prev_face_rect, prev_face_rect_id in zip(
            face_rect_from_detection.get_all_segments(),
            prev_face_rect_from_detection.get_all_segments(),
            prev_face_rect_ids.get_all_segments(),
        ):
            if prev_face_rect.size(0) > 0:
                iou = overlap_similarity(
                    face_rect, prev_face_rect
                )  # [face_rect.size(0), prev_face_rect.size(0)]
                associated = iou.max(dim=1)
                associated_indices = associated.indices
                associated_scores = associated.values
                associated_ids = prev_face_rect_id[associated_indices]
                associated_confidence = (
                    associated_scores >= self.min_tracking_confidence
                )
                new_ids = torch.arange(
                    start=prev_face_rect_id.max() + 1,
                    end=prev_face_rect_id.max()
                    + 1
                    + associated_confidence[~associated_confidence].size(0),
                )  # new face ids where extracted face does not exist in the givin previous faces
                associated_ids[~associated_confidence] = new_ids
            else:
                associated_ids = torch.arange(
                    0, face_rect.size(0), dtype=torch.int64, device=face_rect.device
                )
            face_rect_ids.append(associated_ids)
        return SegmentedTensor(tensors=face_rect_ids)

    def get_face_rect_ids(
        self, face_rect_from_detection: SegmentedTensor
    ) -> SegmentedTensor:
        return SegmentedTensor(
            tensors=[
                torch.arange(
                    0,
                    segment.size(0),
                    dtype=torch.int64,
                    device=face_rect_from_detection.device,
                )
                for segment in face_rect_from_detection.get_all_segments()
            ]
        )

    def _project_landmark(
        self,
        landmarks: torch.Tensor,
        roi: torch.Tensor,
    ) -> torch.Tensor:
        """mediapipe/mediapipe/calculators/util/landmark_projection_calculator.cc
        landmarks: normalized
        roi: [x_center, y_center, width, height, rotation(rad)]
        """
        x = landmarks[..., ::3] - 0.5  # x
        y = landmarks[..., 1::3] - 0.5  # y
        new_x = torch.cos(roi[..., [-1]]) * x - torch.sin(roi[..., [-1]]) * y
        new_y = torch.sin(roi[..., [-1]]) * x + torch.cos(roi[..., [-1]]) * y
        new_x = new_x * roi[..., [2]] + roi[..., [0]]
        new_y = new_y * roi[..., [3]] + roi[..., [1]]
        new_z = landmarks[..., 2::3] * roi[..., [2]]  # Scale Z coordinate as X.
        landmarks[..., ::3] = new_x
        landmarks[..., 1::3] = new_y
        landmarks[..., 2::3] = new_z
        return landmarks

    def convert_roi(self, roi: SegmentedTensor) -> SegmentedTensor:
        """convert_roi
        convert norm bbox [x_center, y_center, width, height] into [ymin, xmin, ymax, xmax]
        """
        new_roi = roi.clone()
        new_roi[..., 0] = roi[..., 1] - roi[..., 3] / 2  # ymin
        new_roi[..., 1] = roi[..., 0] - roi[..., 2] / 2  # xmin
        new_roi[..., 2] = roi[..., 1] + roi[..., 3] / 2  # ymax
        new_roi[..., 3] = roi[..., 0] + roi[..., 2] / 2  # xmax
        return new_roi

    def sort_detections_by_face_association(
        self,
        face_rect_from_detection: SegmentedTensor,
        prev_face_rect_from_detection: SegmentedTensor,
    ) -> SegmentedTensor:
        sorted_rects = []
        segments = {}
        start = 0
        for i, (face_rect, prev_face_rect) in enumerate(
            zip(
                face_rect_from_detection.get_all_segments(),
                prev_face_rect_from_detection.get_all_segments(),
            )
        ):
            if prev_face_rect.size(0) > 0:
                iou = overlap_similarity(face_rect, prev_face_rect)
                associated = iou.max(dim=1)
                ids = torch.arange(face_rect.size(0))
                rest_ids = ids[~torch.isin(ids, associated.indices)]
                face_rect = torch.cat(
                    [face_rect[associated.indices], face_rect[rest_ids]]
                )
                sorted_rects.append(face_rect)
                segments[i] = (start, start + face_rect.size(0))
                start += face_rect.size(0)
            else:
                segments[i] = (start, start)
        return SegmentedTensor(tensors=torch.cat(sorted_rects), segments=segments)

    def forward(
        self,
        images: torch.Tensor,
        prev_face_rect_from_detection: SegmentedTensor | None = None,
        rescale: bool = False,
        output_face_landmarks: bool | None = None,
    ) -> OutputFaceLandmarker:
        blaze_out: OutputBlazeFace = self.face_detection_model(
            images=images, rescale=False
        )  # all the normalized face detections: [ymin, xmin, ymax, xmax]
        all_face_detections = blaze_out.filtered_detections
        if self.use_prev_landmarks and prev_face_rect_from_detection is not None:
            all_face_detections = self.sort_detections_by_face_association(
                face_rect_from_detection=all_face_detections,
                prev_face_rect_from_detection=prev_face_rect_from_detection,
            )  # sorted blazeface results by prev_face_rect_from_detection
        face_detections = self.clip_detection_vector_size(
            all_face_detections
        )  # up to `max_num_faces` detections for each image
        detection_scores = face_detections[..., -1]

        # ImagePropertiesCalculator (get image size)
        if images.size(-1) == 3:
            image_size = images.size()[-3:-1]
        else:
            image_size = images.size()[-2:]

        face_rect_from_detection = self.face_detection_front_detection_to_roi(
            face_detections, image_size
        )  # convert detections to roi properties: [x_center, y_center, width, height, rotation]
        # NORM_RECT: roi -> LandmarkProjection::intput_stream

        if face_rect_from_detection.nelement() == 0:  # No face detected
            warnings.warn("No face detected", UserWarning)
            return OutputFaceLandmarker(
                face_rect_from_detections=SegmentedTensor(
                    [], segments={k: (0, 0) for k in range(images.size(0))}
                ),
                face_detection_scores=SegmentedTensor(
                    [], segments={k: (0, 0) for k in range(images.size(0))}
                ),
                landmarks=SegmentedTensor(
                    [], segments={k: (0, 0) for k in range(images.size(0))}
                ),
                face_landmarks=SegmentedTensor(
                    [], segments={k: (0, 0) for k in range(images.size(0))}
                ),
            )
        inputs, _ = self.image_to_tensor(
            images=images, roi=face_rect_from_detection
        )  # roi image size (192x192), transform matrix (unused)
        with torch.no_grad():
            landmark_tensors, face_flag_tensor = self.facemesh(
                inputs
            )  # landmarks, face presence probs
        face_presence_score = self.tensors_to_floats(face_flag_tensor)  # sigmoid
        face_presence = self.thresholding(face_presence_score)  # float score to bool
        ensured_landmark_tensors = landmark_tensors[face_presence.squeeze(-1)]

        roi_mask = torch.squeeze(face_presence)
        if roi_mask.ndim == 0:
            roi_mask = torch.unsqueeze(roi_mask, dim=0)
        ensured_roi = face_rect_from_detection.clone()[roi_mask]
        face_rect_from_detection = self.convert_roi(
            face_rect_from_detection[..., :4]
        )  # [x_center, y_center, width, height, rotation] -> [top, left, right, bottom]

        ensured_face_rect_from_detection = face_rect_from_detection[roi_mask]
        ensured_face_detection_scores = detection_scores[roi_mask]

        ensured_landmark_tensors[..., ::3] = (
            ensured_landmark_tensors[..., ::3] / 192
        )  # normalize width
        ensured_landmark_tensors[..., 1::3] = (
            ensured_landmark_tensors[..., 1::3] / 192
        )  # normalize height
        ensured_landmark_tensors[..., 2::3] = (
            ensured_landmark_tensors[..., 2::3] / 192 / 1.0
        )  # normalize z by width

        # landmark projection
        landmarks = self._project_landmark(
            landmarks=ensured_landmark_tensors, roi=ensured_roi
        )  # dev
        face_landmarks = None
        if output_face_landmarks:
            face_landmarks = face_detections[..., 4:16][roi_mask]
            face_landmarks_ = face_landmarks.clone()
            face_landmarks_[..., ::3] = face_landmarks[..., ::2]
            face_landmarks_[..., 1::3] = face_landmarks[..., 1::2]
            face_landmarks_ = torch.concat(
                face_landmarks_[..., ::3], face_landmarks_[..., 1::3]
            )
        if rescale:
            if face_landmarks is not None:
                face_landmarks[..., ::3] *= image_size[1]  # x
                face_landmarks[..., 1::3] *= image_size[0]  # y
            ensured_face_rect_from_detection[..., [0, 2]] *= image_size[0]  # ymin, ymax
            ensured_face_rect_from_detection[..., [1, 3]] *= image_size[1]  # xmin, xmax
            landmarks[..., ::3] *= image_size[1]  # x
            landmarks[..., 1::3] *= image_size[0]  # y
            landmarks[..., 2::3] *= image_size[1]  # z
        return OutputFaceLandmarker(
            face_rect_from_detections=ensured_face_rect_from_detection,
            face_detection_scores=ensured_face_detection_scores,
            landmarks=landmarks,
            face_landmarks=face_landmarks,
        )
