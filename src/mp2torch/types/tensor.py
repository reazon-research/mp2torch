import functools
import warnings
from typing import Iterable, TypeVar

import torch

warnings.simplefilter("ignore", UserWarning)

Self = TypeVar("Self", bound="SegmentedTensor")


class SegmentedTensor(torch.Tensor):
    def __init__(
        self,
        tensors: list[torch.Tensor] | torch.Tensor,
        segments: dict[int, tuple[int, int]] | None = None,
    ):
        if segments is None:
            self._register_segment(tensors)
        else:
            self.segments = segments.copy()
            self.n_segments = len(segments)

    def __new__(
        cls,
        tensors: list[torch.Tensor] | torch.Tensor,
        segments: dict[int, tuple[int, int]] | None = None,
    ) -> Self:
        if isinstance(tensors, list):
            if len(tensors) > 0:
                return super().__new__(cls, torch.cat(tensors))
            return super().__new__(cls, torch.empty([]))
        return super().__new__(cls, tensors)

    @classmethod
    @functools.lru_cache
    def _torch_return_types(cls) -> tuple[type, ...]:
        return (torch.return_types.max, torch.return_types.min, torch.return_types.topk)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        ret = super().__torch_function__(func, types, args, kwargs)
        if isinstance(ret, cls._torch_return_types()):  # e.g., max, min, ...
            if args[0].size(0) == ret.values.size(0):  # override segments
                ret.values.segments = args[0].segments
                ret.indices.segments = args[0].segments
        if not isinstance(ret, SegmentedTensor):  # e.g., __set__, print
            return ret
        segments = None
        if kwargs is None:
            kwargs = {}
        if len(args) > 0:
            for arg in args:
                if isinstance(arg, SegmentedTensor):
                    segments = arg.segments
                    break
                elif isinstance(arg, list):
                    for inner in arg:
                        if isinstance(inner, SegmentedTensor):
                            segments = inner.segments
                            break
        if segments is None:
            for _, v in kwargs.items():
                if isinstance(v, SegmentedTensor):
                    segments = v.segments
                    break
        if isinstance(ret, SegmentedTensor):
            ret.segments = segments
            ret.n_segments = len(segments)
        return ret

    def _register_segment(self, tensors: list[torch.Tensor]) -> None:
        self.n_segments = len(tensors)
        self.segments = {}
        start = 0
        for i in range(len(tensors)):
            self.segments[i] = (start, start + tensors[i].shape[0])
            start += tensors[i].size(0)

    def clone(self) -> Self:
        new = super().clone()
        new.segments = self.segments.copy()
        new.n_segments = self.n_segments
        return new

    def __len__(self) -> int:
        return self.n_segments

    def __iter__(self):
        iterator = super().__iter__()
        segments = self.segments.copy()
        for i, ret in enumerate(iterator):
            start = 0
            new_segments = {}
            for k, v in enumerate(segments.values()):
                if v[0] <= i < v[1]:
                    new_segments[k] = (start, start + 1)
                    start += 1
                else:
                    new_segments[k] = (start, start)
            ret.segments = new_segments
            ret.n_segments = len(new_segments)
            yield ret

    def __getitem__(self, indices) -> Self:
        ret = super().__getitem__(indices)
        if isinstance(indices, tuple) and indices[0] == Ellipsis:
            ret.segments = self.segments.copy()
            ret.n_segments = self.n_segments
        else:
            if isinstance(indices, int):
                indices = [indices]
            elif isinstance(indices, torch.Tensor):
                if indices.size(0) != self.data.size(0):
                    raise NotImplementedError
                if indices.dtype == torch.bool:
                    for _ in range(1, indices.ndim):
                        indices = indices.any(dim=1)
                    indices = [i for i, is_true in enumerate(indices) if is_true]
                else:
                    raise NotImplementedError
            elif slice.start is None:
                indices = [slice.stop]
            else:
                indices = list(
                    range(
                        indices.start,
                        indices.stop,
                        indices.step if indices.step is not None else 1,
                    )
                )
            segments = {}
            start = 0
            for k, v in self.segments.items():
                contains = len([idx for idx in indices if v[0] <= idx and idx < v[1]])
                segments[k] = (start, start + contains)
                start += contains
            ret.segments = segments
            ret.n_segments = len(segments)
        return ret

    def to(self, *args, **kwargs) -> Self:
        ret = super().to(*args, **kwargs)
        ret.segments = self.segments.copy()
        ret.n_segments = self.n_segments
        return ret

    def to_tensor(self) -> torch.Tensor:
        return torch.empty(
            self.size(), dtype=self.dtype, device=self.device
        ).new_tensor(self.data)

    def get_all_segments(self) -> Iterable[torch.Tensor]:
        for idx in range(self.n_segments):
            yield self.get_segment(idx=idx)

    def get_segment(self, idx: int) -> torch.Tensor:
        segment = self[self.segments[idx][0] : self.segments[idx][1]]
        return torch.zeros(
            segment.size(), dtype=segment.dtype, device=segment.device
        ).new_tensor(segment.data)

    def update_segment(self, sizes: list[int]) -> None:
        start = 0
        for i, size in enumerate(sizes):
            self.segments[i] = (start, start + size)
            start += size

    def is_empty(self) -> bool:
        for start, end in self.segments.values():
            if not (start == end == 0):
                return False
        return True
