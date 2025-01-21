from abc import ABC, abstractmethod

import torch
from jaxtyping import Float
from torch import Tensor

from ebms.models import BaseEnergyModel


class BaseLoss(ABC):
    @abstractmethod
    def __call__(
        self,
        model: BaseEnergyModel,
        pos_x: Float[Tensor, "batch_size d_model"],
        **kwargs,
    ) -> torch.Tensor:
        pass
