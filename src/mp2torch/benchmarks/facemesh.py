import timeit
from functools import partial

import cv2
import ffmpegcv
import numpy as np
from tqdm import tqdm, trange

import mediapipe as mp
from mp2torch.models.facemesh import FaceLandmarker, OutputFaceLandmarker
from mp2torch.utils.data import VideoFramesBatchLoader


class FaceLandmarkerBenchmark:
    def __init__(
        self,
        onnx_path: str = "models/onnx/face_landmark.onnx",
        onnx_path_face_detection_short_range="models/onnx/face_detection_short_range.onnx",
        static_image_mode: bool = True,
        max_num_faces: int = -1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        trial: int = 10,
        devices: list[str] = ["cpu", "cuda:0"],
        batch_sizes: list[int] = [1, 8, 16, 32],
    ) -> None:
        self.facemesh = FaceLandmarker(
            onnx_path=onnx_path,
            onnx_path_face_detection_short_range=onnx_path_face_detection_short_range,
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            rotate=True,
        )
        self.devices = devices
        self.trial = trial
        self.batch_sizes = batch_sizes
        self.min_tracking_confidence = min_tracking_confidence
        self.min_detection_confidence = min_detection_confidence

    def benchmark(self, video_path: str, metrics: list[str] = ["accuracy", "speed"]):
        if "accuracy" in metrics:
            print("=== Accuracy Benchmark ===")
            res_mp = self._process_mp_facemesh(video_path)
            res_mp2torch = self._process_facemesh(video_path, 64)
            self._check_detection_error(res_mp2torch, res_mp)
            self._calculate_mae(res_mp2torch, res_mp)
        if "speed" in metrics:
            print("=== Speed Benchmark ===")
            time_mp = timeit.repeat(
                partial(self._process_mp_facemesh, video_path),
                repeat=self.trial,
                number=1,
            )
            print(f"DEVICE: CPU | mediapipe | {min(time_mp):.3f} s/file")
            for device in self.devices:
                self.facemesh = self.facemesh.to(device)
                for batch_size in self.batch_sizes:
                    time_mp2torch = timeit.repeat(
                        partial(self._process_facemesh, video_path, batch_size),
                        repeat=self.trial,
                        number=1,
                    )
                    print(
                        f"DEVICE: {device.upper()} | mp2torch | #BATCH: {batch_size} | {min(time_mp2torch):.3f} s/file"
                    )

    def _check_detection_error(self, res_mp2torch, res_mp):
        print("*" * 30)
        print("DETECTION ERROR FRAMES")
        for i, (mp2torch_one, mp_one) in enumerate(zip(res_mp2torch, res_mp)):
            if mp2torch_one.shape != mp_one.shape:
                print(
                    f"FRAME {i} | mp vs mp2torch: {mp_one.shape[0]} vs {mp2torch_one.shape[0]}"
                )
        print("*" * 30)

    def _calculate_mae(self, res_mp2torch, res_mp):
        print("*" * 30)
        print("Absolute Error")
        means, mins, maxs = [], [], []
        for i, (mp2torch_one, mp_one) in enumerate(zip(res_mp2torch, res_mp)):
            if mp2torch_one.shape[0] == 0 or mp_one.shape[0] == 0 or mp2torch_one.shape != mp_one.shape:
                continue
            mp2torch_out = np.stack(mp2torch_one)
            mp_out = np.stack(mp_one)
            diff = np.abs(mp2torch_out - mp_out)
            means.append(diff.mean())
            mins.append(diff.min())
            maxs.append(diff.max())
        print(f"MEAN: {sum(means) / len(means)} | MIN: {min(mins)} | MAX: {max(maxs)}")
        print("*" * 30)

    def _process_facemesh(self, video_path: str, batch_size: int):
        loader = VideoFramesBatchLoader(batch_size)
        res = []
        for batch, has_frame in tqdm(loader.make_loader(video_path), leave=False):
            outs: OutputFaceLandmarker = self.facemesh(batch.to(self.facemesh.device))
            if outs.landmarks.numel() < 1404:
                for _ in range(len(batch)):
                    res.append(np.zeros((0, 1404)))
                continue
            for out in outs.landmarks.get_all_segments():
                res.append(out.cpu().numpy())
        return res

    def _process_mp_facemesh(self, video_path: str):
        res = []
        cap = ffmpegcv.VideoCapture(video_path)
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=20,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as mp_facemesh:
            for _ in trange(len(cap), leave=False):
                _, frame = cap.read()
                ret = mp_facemesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if ret.multi_face_landmarks:
                    res.append(self._convert_landmarks(ret_mp=ret, frame=frame))
                else:
                    res.append(np.zeros((0, 468 * 3)))
        return res

    def _convert_landmarks(self, ret_mp, frame) -> np.ndarray:
        converted = np.array(
            [
                [
                    coord
                    for i in range(468)
                    for coord in (
                        landmarks.landmark[i].x,
                        landmarks.landmark[i].y,
                        landmarks.landmark[i].z,
                    )
                ]
                for landmarks in ret_mp.multi_face_landmarks
            ]
        )
        return converted
