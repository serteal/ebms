import torch
from jaxtyping import Float
from torch import Tensor

from ebms.models import BaseEnergyModel

from .base import BaseLoss


class ScoreMatchingLoss(BaseLoss):
    def __call__(
        self,
        model: BaseEnergyModel,
        pos_x: Float[Tensor, "batch_size d_model"],
        **kwargs,
    ) -> torch.Tensor:
        pos_x.requires_grad_(True)

        # Compute energy (negative to match low energy = high probability)
        energy = -model(pos_x)

        # Compute first-order gradients
        grad_energy = torch.autograd.grad(energy.sum(), pos_x, create_graph=True)[0]

        # Compute trace of Hessian using Hutchinson's trace estimator
        epsilon = torch.randn_like(pos_x)
        grad_eps = torch.sum(grad_energy * epsilon, dim=1)
        hut_trace = torch.autograd.grad(grad_eps.sum(), pos_x, create_graph=True)[0]
        hutchinson_trace = torch.sum(hut_trace * epsilon, dim=1)

        loss = 0.5 * (grad_energy**2).sum(1).mean() + hutchinson_trace.mean()
        return loss


class DenoisingScoreMatchingLoss(BaseLoss):
    def __init__(self, sigma: float = 0.1):
        self.sigma = sigma

    def __call__(
        self,
        model: BaseEnergyModel,
        pos_x: Float[Tensor, "batch_size d_model"],
        **kwargs,
    ) -> torch.Tensor:
        # Add noise to the data
        noise = torch.randn_like(pos_x) * self.sigma
        perturbed_x = pos_x + noise

        # Compute score (negative gradient of energy)
        perturbed_x.requires_grad_(True)
        energy = -model(perturbed_x)
        score = torch.autograd.grad(energy.sum(), perturbed_x, create_graph=True)[0]

        # The score should match the noise direction
        target_score = -noise / (self.sigma**2)

        loss = 0.5 * ((score - target_score) ** 2).sum(1).mean()
        return loss


class SlicedScoreMatchingLoss(BaseLoss):
    def __call__(
        self,
        model: BaseEnergyModel,
        pos_x: Float[Tensor, "batch_size d_model"],
        **kwargs,
    ) -> torch.Tensor:
        pos_x.requires_grad_(True)

        # Random projection vectors
        v = torch.randn_like(pos_x)
        v = v / torch.norm(v, dim=1, keepdim=True)

        # Compute energy and its gradient
        energy = -model(pos_x)
        grad_energy = torch.autograd.grad(energy.sum(), pos_x, create_graph=True)[0]

        # Project gradient onto random vectors
        grad_v = (grad_energy * v).sum(1)

        # Compute second directional derivative
        grad2_v = torch.autograd.grad(grad_v.sum(), pos_x, create_graph=True)[0]
        grad2_v = (grad2_v * v).sum(1)

        loss = 0.5 * (grad_energy**2).sum(1).mean() + grad2_v.mean()
        return loss
