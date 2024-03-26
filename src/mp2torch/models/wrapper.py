import torch
import torch.nn as nn


class BatchedReshape(nn.Module):
    def __init__(self, name: str, module: nn.Module) -> None:
        """BatchedReshape"""
        super().__init__()
        self.name = name
        self.module = module

    def forward(self, inputs: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
        B = inputs.size(0)
        shape[0] = B
        inputs = self.module(inputs, shape)
        return inputs
