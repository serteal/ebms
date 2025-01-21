import torch
from jaxtyping import Float
from torch import Tensor

from ebms.models import BaseEnergyModel
from ebms.samplers import BaseSampler

from .base import BaseLoss


class MaximumLikelihoodMCMCLoss(BaseLoss):
    def __init__(
        self,
        sampler: BaseSampler,
        alpha: float = 0.4,
        buffer_size: int = 10000,
        n_sampler_steps: int = 50,
        device: str = "cpu",
    ):
        self.alpha = alpha
        self.replay_buffer = torch.randn(buffer_size, 2).to(device)
        self.buffer_size = buffer_size
        self.sampler = sampler
        self.n_sampler_steps = n_sampler_steps
        self.device = device

    def __call__(
        self,
        model: BaseEnergyModel,
        pos_x: Float[Tensor, "batch_size d_model"],
        **kwargs,
    ) -> torch.Tensor:
        pos_x = pos_x.to(self.device)
        buffer_idx = torch.randint(0, self.buffer_size, (pos_x.shape[0],))
        neg_x = self.replay_buffer[buffer_idx]
        neg_x = self.sampler.sample(neg_x, model, n_steps=self.n_sampler_steps)
        self.replay_buffer[buffer_idx] = neg_x.detach()

        pos_out = model(pos_x)
        neg_out = model(neg_x)

        energy_term = pos_out.mean() - neg_out.mean()
        regularization_term = (pos_out**2 + neg_out**2).mean()
        return energy_term + self.alpha * regularization_term
