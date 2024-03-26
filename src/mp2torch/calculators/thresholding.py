from dataclasses import dataclass

import torch


@dataclass
class ThresholdingcalculatorOptions:
    threshold: float


class Thresholdingcalculator:
    def __init__(self, options: ThresholdingcalculatorOptions) -> None:
        self.__options = options

    def process(self, inputs: torch.Tensor) -> torch.BoolTensor:
        return inputs >= self.__options.threshold

    def __call__(self, inputs: torch.Tensor) -> torch.BoolTensor:
        return self.process(inputs)
