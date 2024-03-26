from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class TensorsToFloatsCalculatorOptions:
    activation: str | None = None


class TensorsToFloatsCalculator:
    def __init__(self, options: TensorsToFloatsCalculatorOptions) -> None:
        self.__options = options

    def process(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.__options.activation.lower() == "sigmoid":
            return F.sigmoid(inputs)
        elif self.__options.activation is None:
            return inputs
        raise NotImplementedError

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.process(inputs)
