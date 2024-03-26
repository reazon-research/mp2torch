import timeit
from functools import partial

import cv2
import ffmpegcv
import numpy as np
from tqdm import tqdm

import mediapipe as mp
from mp2torch.models.blazeface import BlazeFace
from mp2torch.utils.data import VideoFramesBatchLoader


class BlazeFaceBenchmark:
    def __init__(
        self,
        onnx_path: str = "models/onnx/face_detection_short_range.onnx",
        width: int = 128,
        height: int = 128,
        min_score_threshold: float = 0.5,
        trial: int = 10,
        devices: list[str] = ["cpu", "cuda:0"],
        batch_sizes: list[int] = [1, 8, 16, 32],
        backend: str = "ffmpegcv",  # or cv2
    ) -> None:
        self.blazeface = BlazeFace(
            onnx_path=onnx_path,
            min_score_threshold=min_score_threshold,
            width=width,
            height=height,
            rescale=False,
        )
        self.devices = devices
        self.min_score_threshold = min_score_threshold
        self.trial = trial
        self.batch_sizes = batch_sizes
        self.backend = backend

    def benchmark(self, video_path: str, metrics: list[str] = ["accuracy", "speed"]):
        if "accuracy" in metrics:
            print("=== Accuracy Benchmark ===")
            res_blz = self._process_blazeface(video_path, 32)
            res_mp = self._process_mp_blazeface(video_path)
            self._check_detection_error(res_blz, res_mp)
            self._calculate_mae(res_blz, res_mp)
        if "speed" in metrics:
            print("=== Speed Benchmark ===")
            time_mp = timeit.repeat(
                partial(self._process_mp_blazeface, video_path),
                repeat=self.trial,
                number=1,
            )
            print(f"DEVICE: CPU | mediapipe | {min(time_mp):.3f} s/file")
            for device in self.devices:
                self.blazeface = self.blazeface.to(device)
                for batch_size in self.batch_sizes:
                    time_blz = timeit.repeat(
                        partial(self._process_blazeface, video_path, batch_size),
                        repeat=self.trial,
                        number=1,
                    )
                    print(
                        f"DEVICE: {device.upper()} | mp2torch | #BATCH: {batch_size} | {min(time_blz):.3f} s/file"
                    )

    def _check_detection_error(self, res_blz: list, res_mp: list):
        print("*" * 30)
        print("DETECTION ERROR FRAMES")
        for i, (blz_one, mp_one) in enumerate(zip(res_blz, res_mp)):
            if blz_one.shape != mp_one.shape:
                print(
                    f"FRAME: {i} | mp vs blz: {mp_one.shape[0]} vs {blz_one.shape[0]}"
                )
        print()
        print("*" * 30)

    def _calculate_mae(self, res_blz, res_mp):
        print("*" * 30)
        print("Absolute Error")
        means, mins, maxs = [], [], []
        for i, (blz_one, mp_one) in enumerate(zip(res_blz, res_mp)):
            if blz_one.shape[0] == 0 or mp_one.shape[0] == 0 or blz_one.shape != mp_one.shape:
                continue
            blz_out = np.stack(blz_one)
            mp_out = np.stack(mp_one)
            diff = np.abs(blz_out - mp_out)
            means.append(diff.mean())
            mins.append(diff.min())
            maxs.append(diff.max())
        print(f"MEAN: {sum(means) / len(means)} | MIN: {min(mins)} | MAX: {max(maxs)}")
        print("*" * 30)

    def _process_blazeface(
        self, video_path: str, batch_size: int, resize: tuple[int, int] | None = None
    ):
        loader = VideoFramesBatchLoader(batch_size=batch_size)
        res = []
        for batch, has_frames in tqdm(loader.make_loader(video_path), leave=False):
            outs = self.blazeface(
                batch.to(self.blazeface.device),
                return_tensors="np",
            )
            for out in outs.filtered_detections:
                res.append(out)
        return res

    def _process_mp_blazeface(self, video_path: str):
        res = []
        if self.backend == "cv2":
            cap = cv2.VideoCapture(video_path)
            nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            cap = ffmpegcv.noblock(
                ffmpegcv.VideoCapture,
                video_path,
            )
            nframes = len(cap)
        with mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=self.min_score_threshold,
            model_selection=0,
        ) as mp_blazeface:
            for _ in range(nframes):
                ret, frame = cap.read()
                if not ret:
                    break
                ret_blz = mp_blazeface.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if ret_blz.detections:
                    res.append(self._norm_detections(ret_blz.detections))
                else:
                    res.append(np.zeros((0, 17)))
        cap.release()
        return res

    def _norm_detections(self, detections) -> np.ndarray:
        normalized_detections = []
        for detection in detections:
            bbox = detection.location_data.relative_bounding_box
            keypoints = detection.location_data.relative_keypoints
            normalized_detections.append(
                [
                    bbox.ymin,
                    bbox.xmin,
                    bbox.ymin + bbox.height,  # ymax
                    bbox.xmin + bbox.width,  # xmax
                ]
                + [
                    coord
                    for keypoint in keypoints
                    for coord in [keypoint.x, keypoint.y]
                ]
                + [detection.score.pop(0)]
            )
        return np.array(normalized_detections)
