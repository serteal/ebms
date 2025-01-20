from typing import Tuple

import numpy as np
import torch


class ToyDataGenerator:
    @staticmethod
    def generate_8gaussians(num_samples: int) -> torch.Tensor:
        z = torch.randn(num_samples, 2)
        scale = 4
        sq2 = 1 / np.sqrt(2)
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (sq2, sq2),
            (-sq2, sq2),
            (sq2, -sq2),
            (-sq2, -sq2),
        ]
        centers = torch.tensor([(scale * x, scale * y) for x, y in centers])
        return (
            sq2 * (0.1 * z + centers[torch.randint(len(centers), size=(num_samples,))])
        ).float()

    @staticmethod
    def generate_2spirals(num_samples: int) -> torch.Tensor:
        z = torch.randn(num_samples, 2)
        n = torch.sqrt(torch.rand(num_samples // 2)) * 540 * (2 * np.pi) / 360
        d1x = -torch.cos(n) * n + torch.rand(num_samples // 2) * 0.5
        d1y = torch.sin(n) * n + torch.rand(num_samples // 2) * 0.5
        x = (
            torch.cat([torch.stack([d1x, d1y], dim=1), torch.stack([-d1x, -d1y], dim=1)], dim=0)
            / 3
        )
        return (x + 0.1 * z).float()

    @staticmethod
    def generate_4squares(num_samples: int) -> torch.Tensor:
        points_per_square = num_samples // 4
        centers = torch.tensor(
            [
                [-1.5, 1.5],
                [1.5, 1.5],
                [-1.5, -1.5],
                [1.5, -1.5],
            ]
        )

        square_size = 1.0
        noise = 0.1
        points = []

        for center in centers:
            x = torch.empty(points_per_square).uniform_(
                center[0] - square_size / 2, center[0] + square_size / 2
            )
            y = torch.empty(points_per_square).uniform_(
                center[1] - square_size / 2, center[1] + square_size / 2
            )

            x += torch.randn(points_per_square) * noise
            y += torch.randn(points_per_square) * noise

            points.append(torch.stack([x, y], dim=1))

        return torch.cat(points, dim=0)

    @staticmethod
    def generate(dataset_name: str, num_samples: int) -> torch.Tensor:
        if dataset_name == "8gaussians":
            return ToyDataGenerator.generate_8gaussians(num_samples)
        elif dataset_name == "2spirals":
            return ToyDataGenerator.generate_2spirals(num_samples)
        elif dataset_name == "4squares":
            return ToyDataGenerator.generate_4squares(num_samples)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
