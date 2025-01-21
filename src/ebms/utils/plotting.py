from typing import Tuple

import matplotlib.pyplot as plt
import torch
from jaxtyping import Float
from torch import Tensor

from ebms.models import BaseEnergyModel
from ebms.samplers import BaseSampler


def generate_viz_samples(
    model: BaseEnergyModel,
    sampler: BaseSampler,
    device: torch.device,
    sample_shape: Tuple[int, ...],
    num_samples: int = 1000,
    num_sampling_steps: int = 100,
    **kwargs,
) -> Float[Tensor, "batch_size d_model"]:
    init_samples = torch.randn(num_samples, *sample_shape).to(device)
    return sampler.sample(init_samples, model, num_sampling_steps, **kwargs)


def visualize_samples(
    model: BaseEnergyModel,
    sampler: BaseSampler,
    train_data: Float[Tensor, "batch_size d_model"],
    num_generated_samples: int = 1000,
    num_sampling_steps: int = 100,
    title: str = "Real vs Generated Samples",
    **kwargs,
) -> plt.Figure:
    """Visualize real and generated samples with energy contours"""

    device = next(model.parameters()).device
    real_samples = train_data.cpu()
    generated_samples = generate_viz_samples(
        model,
        sampler,
        device=device,
        num_samples=num_generated_samples,
        sample_shape=train_data.shape[1:],
        num_sampling_steps=num_sampling_steps,
        **kwargs,
    ).cpu()

    # Create figure with custom gridspec
    fig = plt.figure(figsize=(20, 5))
    gs = fig.add_gridspec(1, 3)

    # First subplot (samples)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(
        real_samples[:, 0].numpy(),
        real_samples[:, 1].numpy(),
        alpha=0.2,
        color="red",
        label="Real",
        s=1,
    )
    ax1.scatter(
        generated_samples[:, 0].numpy(),
        generated_samples[:, 1].numpy(),
        alpha=0.7,
        label="Generated",
        s=5,
    )
    ax1.set_title("Real vs Generated Samples")
    ax1.grid(True)
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    ax1.legend()

    # Create grid for energy landscape
    x = torch.linspace(-4, 4, 100)
    y = torch.linspace(-4, 4, 100)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1).to(device)

    with torch.no_grad():
        energies = model(grid_points).reshape(100, 100).cpu()

    # Second subplot (contour)
    ax2 = fig.add_subplot(gs[0, 1])
    contour = ax2.contour(X.numpy(), Y.numpy(), energies.numpy(), levels=20, cmap="rainbow")
    plt.colorbar(contour, ax=ax2, label="Energy")
    ax2.set_title("Energy Landscape")
    ax2.grid(True)
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)

    ax2.scatter(
        real_samples[:, 0].numpy(),
        real_samples[:, 1].numpy(),
        alpha=0.1,
        color="red",
        s=1,
    )

    # Third subplot (3D)
    ax3 = fig.add_subplot(gs[0, 2], projection="3d")
    surf = ax3.plot_surface(X.numpy(), Y.numpy(), energies.numpy(), cmap="rainbow", alpha=1.0)
    plt.colorbar(surf, ax=ax3, label="Energy")
    ax3.set_xlim(-4, 4)
    ax3.set_ylim(-4, 4)
    ax3.view_init(elev=45, azim=15)
    ax3.set_title("3D Energy Landscape")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

    return fig
