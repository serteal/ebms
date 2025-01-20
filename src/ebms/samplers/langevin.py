from typing import Optional

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from ebms.models import BaseEnergyModel

from .base import BaseSampler


class LangevinMC(BaseSampler):
    def __init__(self, stepsize: float, noise_scale: Optional[float] = None):
        self.stepsize = stepsize
        self.noise_scale = noise_scale or np.sqrt(stepsize * 2)

    def sample(
        self,
        x: Float[Tensor, "batch_size d_model"],
        model: BaseEnergyModel,
        n_steps: int,
        **kwargs,
    ) -> Float[Tensor, "batch_size d_model"]:
        x = x.detach().requires_grad_(True)

        for _ in range(n_steps):
            energy = model(x)
            grad = torch.autograd.grad(energy.sum(), x)[0]

            # Normalize gradients to prevent extreme steps
            grad_norm = torch.norm(grad, dim=-1, keepdim=True)
            grad = grad / (grad_norm + 1e-6)

            noise = torch.randn_like(x) * self.noise_scale
            x = x - self.stepsize * grad + noise
            x = x.detach().requires_grad_(True)

        return x.detach()
