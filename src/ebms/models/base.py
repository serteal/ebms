from abc import ABC, abstractmethod

import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class BaseEnergyModel(ABC, nn.Module):
    @abstractmethod
    def forward(self, x: Float[Tensor, "batch_size d_model"]) -> Float[Tensor, " batch_size"]:
        """Compute energy for input x"""
        pass
