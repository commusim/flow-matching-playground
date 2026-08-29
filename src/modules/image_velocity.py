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
