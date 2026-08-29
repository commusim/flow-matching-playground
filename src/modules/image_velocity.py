import math
import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension=32):
        super().__init__()
        half = dimension // 2
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(half).float() / max(half - 1, 1)
        )
        self.register_buffer("frequencies", frequencies)

    def forward(self, time):
        angles = time * self.frequencies[None, :]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class MNISTVelocityCNN(nn.Module):
    """Small convolutional velocity predictor for 28x28 pixel-space Flow Matching."""

    def __init__(self, hidden=64, time_dim=32):
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(time_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.input = nn.Conv2d(1, hidden, 3, padding=1)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, hidden),
                    nn.SiLU(),
                    nn.Conv2d(hidden, hidden, 3, padding=1),
                ),
                nn.Sequential(
                    nn.GroupNorm(8, hidden),
                    nn.SiLU(),
                    nn.Conv2d(hidden, hidden, 3, padding=1),
                ),
                nn.Sequential(
                    nn.GroupNorm(8, hidden),
                    nn.SiLU(),
                    nn.Conv2d(hidden, hidden, 3, padding=1),
                ),
            ]
        )
        self.output = nn.Sequential(
            nn.GroupNorm(8, hidden), nn.SiLU(), nn.Conv2d(hidden, 1, 3, padding=1)
        )

    def forward(self, image, time):
        features = self.input(image)
        time_features = self.time_projection(self.time_embedding(time)).view(
            len(image), -1, 1, 1
        )
        features = features + time_features
        for block in self.blocks:
            features = features + block(features)
        return self.output(features)


class ConditionalResidualBlock(nn.Module):
    """Residual block modulated by time and label after GroupNorm."""

    def __init__(self, hidden, condition_dim):
        super().__init__()
        self.norm = nn.GroupNorm(8, hidden)
        self.modulation = nn.Linear(condition_dim, 2 * hidden)
        self.conv = nn.Conv2d(hidden, hidden, 3, padding=1)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, features, condition):
        scale, shift = self.modulation(condition).chunk(2, dim=1)
        scale = scale.view(len(features), -1, 1, 1)
        shift = shift.view(len(features), -1, 1, 1)
        residual = self.norm(features)
        residual = residual * (1 + scale) + shift
        residual = self.conv(torch.nn.functional.silu(residual))
        return features + residual


class ConditionalMNISTVelocityCNN(nn.Module):
    """MNIST velocity model with AdaGN label/time conditioning in every block."""

    def __init__(self, hidden=64, time_dim=32, label_dim=32, classes=10):
        super().__init__()
        if hidden % 8 != 0:
            raise ValueError("hidden must be divisible by 8 for GroupNorm")
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.label_embedding = nn.Embedding(classes, label_dim)
        self.condition_projection = nn.Sequential(
            nn.Linear(time_dim + label_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.input = nn.Conv2d(1, hidden, 3, padding=1)
        self.blocks = nn.ModuleList(
            [ConditionalResidualBlock(hidden, hidden) for _ in range(4)]
        )
        self.output_norm = nn.GroupNorm(8, hidden)
        self.output_modulation = nn.Linear(hidden, 2 * hidden)
        self.output_conv = nn.Conv2d(hidden, 1, 3, padding=1)
        nn.init.zeros_(self.output_modulation.weight)
        nn.init.zeros_(self.output_modulation.bias)

    def _condition(self, time, label):
        time_features = self.time_embedding(time)
        label_features = self.label_embedding(label)
        return self.condition_projection(
            torch.cat([time_features, label_features], dim=1)
        )

    def forward(self, image, time, label, return_features=False):
        condition = self._condition(time, label)
        features = self.input(image)
        intermediate = {"input": features}
        for index, block in enumerate(self.blocks, start=1):
            features = block(features, condition)
            intermediate[f"block_{index}"] = features
        scale, shift = self.output_modulation(condition).chunk(2, dim=1)
        output_features = self.output_norm(features)
        output_features = output_features * (1 + scale[:, :, None, None])
        output_features = output_features + shift[:, :, None, None]
        velocity = self.output_conv(torch.nn.functional.silu(output_features))
        if return_features:
            return velocity, intermediate
        return velocity
