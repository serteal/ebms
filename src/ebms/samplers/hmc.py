import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from ebms.models import BaseEnergyModel

from .base import BaseSampler


class HamiltonianMC(BaseSampler):
    def __init__(self, step_size: float = 0.01, leapfrog_steps: int = 10, mass: float = 1.0):
        self.step_size = step_size
        self.leapfrog_steps = leapfrog_steps
        self.mass = mass

    def sample(
        self,
        x: Float[Tensor, "batch_size d_model"],
        model: BaseEnergyModel,
        n_steps: int,
        **kwargs,
    ) -> Float[Tensor, "batch_size d_model"]:
        x = x.detach().requires_grad_(True)

        for _ in range(n_steps):
            momentum = torch.randn_like(x) * np.sqrt(self.mass)
            init_energy = model(x).squeeze()
            init_kinetic = 0.5 * (momentum**2 / self.mass).sum(1)

            current_x = x
            current_momentum = momentum

            # Leapfrog integration
            for _ in range(self.leapfrog_steps):
                energy = model(current_x).squeeze()
                grad = torch.autograd.grad(energy.sum(), current_x)[0]
                current_momentum = current_momentum - 0.5 * self.step_size * grad

                current_x = current_x + self.step_size * (current_momentum / self.mass)
                current_x = current_x.detach().requires_grad_(True)

                energy = model(current_x).squeeze()
                grad = torch.autograd.grad(energy.sum(), current_x)[0]
                current_momentum = current_momentum - 0.5 * self.step_size * grad

            final_energy = model(current_x).squeeze()
            final_kinetic = 0.5 * (current_momentum**2 / self.mass).sum(1)

            # Metropolis-Hastings correction
            energy_diff = (final_energy + final_kinetic) - (init_energy + init_kinetic)
            acceptance = torch.rand_like(energy_diff) < torch.exp(-energy_diff)
            x = torch.where(acceptance.unsqueeze(1), current_x, x)
            x = x.detach().requires_grad_(True)

        return x.detach()
