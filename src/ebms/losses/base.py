from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from ebms.models import BaseEnergyModel
from ebms.samplers import BaseSampler


class BaseLoss(ABC):
    @abstractmethod
    def __call__(
        self,
        model: BaseEnergyModel,
        pos_x: Float[Tensor, "batch_size d_model"],
        **kwargs,
    ) -> torch.Tensor:
        pass
