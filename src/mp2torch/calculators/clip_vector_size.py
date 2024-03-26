"""mediapipe/mediapipe/calculators/core/clip_vector_size_calculator.proto
mediapipe/mediapipe/calculators/core/clip_vector_size_calculator.h
mediapipe/mediapipe/calculators/core/clip_vector_size_calculator.cc"""
from dataclasses import dataclass

import torch

from mp2torch.types.tensor import SegmentedTensor


@dataclass
class ClipVectorSizeCalculatorOptions:
    max_vec_size: int = 1


class ClipVectorSizeCalculator:
    """ClipVectorSizecalculator
    mediapipe/mediapipe/calculators/core/clip_vector_size_calculator.h"""

    def __init__(self, options: ClipVectorSizeCalculatorOptions) -> None:
        self.__options = options
        if self.__options.max_vec_size == 0:
            raise ValueError(
                "max_vec_size should be greater than or equal to 1, or less than 0. "
                "negative max_vec_size means passing all vectors."
            )

    def process(
        self, inputs: list[torch.Tensor] | torch.Tensor | SegmentedTensor
    ) -> torch.Tensor | list[torch.Tensor]:
        if self.__options.max_vec_size < 0:
            return inputs.clone()
        if isinstance(inputs, SegmentedTensor):
            inputs = SegmentedTensor(
                [
                    segment[: self.__options.max_vec_size]
                    for segment in inputs.get_all_segments()
                ]
            )
            return inputs
        return [inp[: self.__options.max_vec_size].clone() for inp in inputs]

    def __call__(self, inputs: list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
        return self.process(inputs)
