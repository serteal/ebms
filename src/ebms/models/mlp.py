import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from .base import BaseEnergyModel


class MLP(BaseEnergyModel):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Float[Tensor, "batch_size d_model"]) -> Float[Tensor, "batch_size"]:
        return self.net(x)
