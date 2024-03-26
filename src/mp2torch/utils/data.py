import collections
import math
import multiprocessing as mp
from multiprocessing.dummy import Pool  # multithreading-based
from pathlib import Path
from typing import Iterator

import cv2
import einops
import ffmpegcv
import numpy as np
import torch


class VideoBatchLoader(object):
    def __init__(self, preload_size: int = 1, backend: str = "cv2") -> None:
        """VideoBatchLoader
        Dataloader which load video frames with batched form

        Parameters
        ----------
        preload_size: int, default=1
            The number of frames preloaded when firstly opning video.
            `Too many open files` error will occur if batch size is too large.
            If the error occurs, `preload_size` is set to large and then the error can be prevented
            while the RAM usage grows.
        """
        self.loaders: list[cv2.VideoCapture] | None = None
        self.preload_size = preload_size
        self.backend = backend

    def _release_loaders(self) -> None:
        for loader in self.loaders:
            loader.release()
        self.loaders = None

    def _batched_load_inner(
        self, loader: ffmpegcv.VideoCapture
    ) -> tuple[np.ndarray, int]:
        if loader.isOpened():
            ret, frame = loader.read()
            if ret:
                return frame, 1
        return (
            np.zeros(
                (
                    int(loader.height),
                    int(loader.width),
                    3,
                ),
                dtype=np.uint8,
            ),
            0,
        )

    def _batched_load(self) -> torch.Tensor:
        n_cpus = mp.cpu_count()
        with Pool(processes=min(n_cpus, len(self.loaders))) as pool:
            try:
                ret = pool.map_async(self._batched_load_inner, self.loaders).get()
            except Exception as e:
                pool.close()
                self._release_loaders()
                raise e
        batch, has_frame = list(zip(*ret))
        batch = einops.rearrange(
            torch.from_numpy(np.stack(batch, axis=0)[..., [2, 1, 0]]),
            "b h w c -> b c h w",
        )  # bgr -> rgb
        has_frame = torch.tensor(has_frame)
        return batch, has_frame

    def make_loader(
        self,
        video_paths: list[str | Path],
        codec: str = "h264",
        crop_xywh: tuple[int, int, int, int] | None = None,
        resize: tuple[int, int] | None = None,
        resize_keepratio: bool = True,
        resize_keepratioalign: str = "center",
        preload_size: int | None = None,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        video_paths: list[str | Path]
            The video files for batched loading
        codec: str, default='h264'
            Codec to encode video using `ffmpegcv`. When you want encoding on gpu, you can set `codec` to like 'h264_cuvid'
        crop_xywh: tuple[int, int, int, int], optional
            `crop_xywh` must be set in the form of (left, top, width, height) when you want crop a frame by `ffmpeg`.
        resize: tuple[int, int], optional
            `resize` must be set in the form of (width, height) when you want resize a frame by `ffmpeg`.
            Resizing is performed after cropping
        resize_keepratio: bool, defalut=True
            Wheather keep a ratio of an input frame or not when resizing the frame
        resize_keepratioalign: str, default='center'
            Where to align a frame when resizing the frame
        preload_size: int, optional
            The number of frames preloaded when firstly opning video.
            If this parameter is givin, overwrite `self.preload_size`, so you use this with caution.

        Returns
        -------
        tuple[torch.Tesnor, torch.Tensor]
            batched video frames and boolean tensors which represent whether frame exists or not
        """
        if self.loaders is not None:
            self._release_loaders()
        if preload_size is not None:  # override
            self.preload_size = preload_size
        if self.backend == "cv2":
            self.loaders = [
                cv2.VideoCapture(
                    video_path if isinstance(video_path, str) else video_path.as_posix()
                )
                for video_path in video_paths
            ]
        else:
            self.loaders = [
                ffmpegcv.VideoCapture(
                    video_path if isinstance(video_path, str) else video_path.as_posix(),
                    codec=codec,
                    crop_xywh=crop_xywh,
                    resize=resize,
                    resize_keepratio=resize_keepratio,
                    resize_keepratioalign=resize_keepratioalign,
                )
                for video_path in video_paths
            ]
        return self

    def __iter__(self):
        pool = collections.deque([], maxlen=self.preload_size)
        for _ in range(len(self)):
            if len(pool) == self.preload_size:
                yield pool.popleft()
            pool.append(self._batched_load())
        while pool:
            yield pool.popleft()
        self._release_loaders()

    def __len__(self) -> int:
        if self.loaders is None:
            return 0
        def get_nframes(loader):
            nframes = (
                int(self.loader.get(cv2.CAP_PROP_FRAME_COUNT))
                if self.backend == "cv2"
                else len(self.laoder)
            )
        return int(max([get_nframes(loader) for loader in self.loaders]))


class VideoFramesBatchLoader(object):
    def __init__(self, batch_size: int = 1, backend: str = "cv2"):
        self.batch_size = batch_size
        self.loader = None
        self.backend = backend

    def make_loader(
        self,
        video_path: str | Path,
        batch_size: int | None = None,
        codec: str = "h264",
        crop_xywh: tuple[int, int, int, int] | None = None,
        resize: tuple[int, int] | None = None,
        resize_keepratio: bool = True,
        resize_keepratioalign: str = "center",
    ):
        """
        Parameters
        ----------
        video_paths: list[str | Path]
            The video files for batched loading
        batch_size: int, optional
            Batch size
        codec: str, default='h264'
            Codec to encode video using `ffmpegcv`. When you want encoding on gpu, you can set `codec` to like 'h264_cuvid'
        crop_xywh: tuple[int, int, int, int], optional
            `crop_xywh` must be set in the form of (left, top, width, height) when you want crop a frame by `ffmpeg`.
        resize: tuple[int, int], optional
            `resize` must be set in the form of (width, height) when you want resize a frame by `ffmpeg`.
            Resizing is performed after cropping
        resize_keepratio: bool, defalut=True
            Wheather keep a ratio of an input frame or not when resizing the frame
        resize_keepratioalign: str, default='center'
            Where to align a frame when resizing the frame

        Returns
        -------
        tuple[torch.Tesnor, torch.Tensor]
            batched video frames and boolean tensors which represent whether frame exists or not
        """
        if self.loader is not None:
            self.loader.release()
        if batch_size is not None:
            self.batch_size = batch_size
        if self.backend == "cv2":
            self.loader = cv2.VideoCapture(
                video_path if isinstance(video_path, str) else video_path.as_posix()
            )
        else:
            self.loader = ffmpegcv.VideoCapture(
                video_path if isinstance(video_path, str) else video_path.as_posix(),
                codec=codec,
                crop_xywh=crop_xywh,
                resize=resize,
                resize_keepratio=resize_keepratio,
                resize_keepratioalign=resize_keepratioalign,
            )
        return self

    def _collate(self, batch: list[tuple]) -> tuple[torch.Tensor, torch.Tensor]:
        has_frame, batch = list(zip(*batch))
        batch = einops.rearrange(
            torch.from_numpy(np.stack(batch, axis=0)[..., [2, 1, 0]]),
            "b h w c -> b c h w",
        )  # bgr -> rgb
        has_frame = torch.tensor(has_frame)
        return batch, has_frame

    def __iter__(self):
        batch = []
        nframes = (
            int(self.loader.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.backend == "cv2"
            else len(self.loader)
        )
        for _ in range(nframes):
            batch.append(self.loader.read())
            if len(batch) == self.batch_size:
                yield self._collate(batch)
                batch = []
        if batch:
            yield self._collate(batch)
        self.loader.release()

    def __len__(self) -> int:
        if self.loader is None:
            return 0
        nframes = (
            int(self.loader.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.backend == "cv2"
            else len(self.laoder)
        )
        return math.ceil(nframes / self.batch_size)


class ImageBatchLoader(object):
    def __init__(self, batch_size: int = 1, resize: tuple[int, int] | None = None):
        self.batch_size = batch_size
        self.resize = resize
        self.loaders = None

    def make_loader(
        self,
        image_paths: list[str | Path],
        batch_size: int | None = None,
        resize: tuple[int, int] | None = None,
    ):
        if batch_size is not None:
            self.batch_size = batch_size
        if resize is not None:
            self.resize = resize
        self.loaders = image_paths
        return self

    def _load_image(self, filename: str | Path):
        if isinstance(filename, Path):
            filename = filename.as_posix()
        image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        if self.resize is not None:
            width, height, _ = image.shape
            aspect = height / width
            resize_aspect = self.resize[1] / self.resize[0]
            vpad, hpad = 0, 0
            if aspect > resize_aspect:
                vpad = (1.0 - resize_aspect / aspect) / 2
                vpad = int(vpad * width * aspect)
            else:
                hpad = (1.0 - aspect / resize_aspect) / 2
                hpad = int(hpad * height / aspect)
            image = cv2.copyMakeBorder(
                image, vpad, vpad, hpad, hpad, cv2.BORDER_CONSTANT, (0, 0, 0)
            )
            image = cv2.resize(image, self.resize, interpolation=cv2.INTER_NEAREST)
        return image

    def __iter__(self):
        n_cpus = mp.cpu_count()
        batch_start = 0
        batch_end = 0
        for batch_end in range(self.batch_size, len(self.loaders), self.batch_size):
            with Pool(processes=min(n_cpus, self.batch_size)) as pool:
                ret = pool.map(self._load_image, self.loaders[batch_start:batch_end])
            batch = einops.rearrange(
                torch.from_numpy(np.stack(ret, axis=0)),
                "b h w c -> b c h w",
            )
            yield batch
            batch_start = batch_end

        if batch_end < len(self.loaders):
            with Pool(processes=min(n_cpus, self.batch_size)) as pool:
                ret = pool.map(self._load_image, self.loaders[batch_end:])
            batch = einops.rearrange(
                torch.from_numpy(np.stack(ret, axis=0)),
                "b h w c -> b c h w",
            )
            yield batch
        self.loaders = None

    def __len__(self) -> int:
        if self.loaders is None:
            return 0
        return len(self.loaders)
