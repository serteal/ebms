from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch
from jaxtyping import Float
from torch import Tensor

from ebms.models import BaseEnergyModel


class BaseSampler(ABC):
    @abstractmethod
    def sample(
        self,
        x: Float[Tensor, "batch_size d_model"],
        model: BaseEnergyModel,
        n_steps: int,
        **kwargs,
    ) -> Float[Tensor, "batch_size d_model"]:
        """Base sampling method"""
        pass
