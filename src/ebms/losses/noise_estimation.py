import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from ebms.models import BaseEnergyModel

from .base import BaseLoss


class NoiseContrastiveEstimationLoss(BaseLoss):
    def __init__(
        self,
        noise_distribution: str = "gaussian",
        noise_std: float = 1.0,
        num_noise_samples: int = 1,
    ):
        """
        Args:
            noise_distribution: Type of noise distribution ("gaussian" or "uniform")
            noise_std: Standard deviation for Gaussian noise or range for uniform noise
            num_noise_samples: Number of noise samples per real sample (k in NCE)
        """
        self.noise_distribution = noise_distribution
        self.noise_std = noise_std
        self.num_noise_samples = num_noise_samples

    def _sample_noise(
        self, pos_x: Float[Tensor, "batch_size d_model"]
    ) -> Float[Tensor, "batch_size * num_noise_samples d_model"]:
        batch_size = pos_x.shape[0]
        shape = (batch_size * self.num_noise_samples,) + pos_x.shape[1:]

        if self.noise_distribution == "gaussian":
            return torch.randn(shape, device=pos_x.device) * self.noise_std
        elif self.noise_distribution == "uniform":
            return torch.rand(shape, device=pos_x.device) * 2 * self.noise_std - self.noise_std
        else:
            raise ValueError(f"Unknown noise distribution: {self.noise_distribution}")

    def _noise_log_prob(
        self, x: Float[Tensor, "batch_size * num_noise_samples d_model"]
    ) -> torch.Tensor:
        if self.noise_distribution == "gaussian":
            return -0.5 * (x**2).sum(1) / (self.noise_std**2) - x.shape[1] * np.log(
                self.noise_std * np.sqrt(2 * np.pi)
            )
        elif self.noise_distribution == "uniform":
            # Uniform distribution has constant log probability within its support
            return -torch.sum(torch.log(2 * self.noise_std * torch.ones_like(x)), dim=1)
        else:
            raise ValueError(f"Unknown noise distribution: {self.noise_distribution}")

    def __call__(
        self,
        model: BaseEnergyModel,
        pos_x: Float[Tensor, "batch_size d_model"],
        **kwargs,
    ) -> torch.Tensor:
        batch_size = pos_x.shape[0]

        # Generate noise samples
        noise_samples = self._sample_noise(pos_x)
        all_samples = torch.cat([pos_x, noise_samples], dim=0)

        # Get energy scores (negative to match low energy = high probability)
        energies = -model(all_samples)

        # Split energies for real and noise samples
        pos_energies = energies[:batch_size]
        neg_energies = energies[batch_size:].view(batch_size, self.num_noise_samples)

        # Compute log probabilities
        pos_log_prob = pos_energies - self._noise_log_prob(pos_x)
        neg_log_prob = neg_energies - self._noise_log_prob(
            noise_samples.view(batch_size, self.num_noise_samples, -1)
        )

        # NCE loss (negative log-likelihood of the classifier)
        pos_term = -F.logsigmoid(pos_log_prob)
        neg_term = -F.logsigmoid(-neg_log_prob).sum(1)
        loss = (pos_term + neg_term).mean()
        return loss
