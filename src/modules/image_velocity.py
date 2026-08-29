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


class ConditionalMNISTVelocityCNN(nn.Module):
    """Label-conditioned pixel-space velocity model for MNIST."""

    def __init__(self, hidden=64, time_dim=32, label_dim=32, classes=10):
        super().__init__()
        if hidden % 8 != 0:
            raise ValueError("hidden must be divisible by 8 for GroupNorm")
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(time_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.label_embedding = nn.Embedding(classes, label_dim)
        self.label_projection = nn.Sequential(
            nn.Linear(label_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.input = nn.Conv2d(1, hidden, 3, padding=1)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, hidden),
                    nn.SiLU(),
                    nn.Conv2d(hidden, hidden, 3, padding=1),
                )
                for _ in range(3)
            ]
        )
        self.output = nn.Sequential(
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 3, padding=1),
        )

    def forward(self, image, time, label, return_features=False):
        features = self.input(image)
        time_features = self.time_projection(self.time_embedding(time))
        label_features = self.label_projection(self.label_embedding(label))
        features = features + (time_features + label_features).view(
            len(image), -1, 1, 1
        )
        intermediate = {"input": features}
        for index, block in enumerate(self.blocks, start=1):
            features = features + block(features)
            intermediate[f"block_{index}"] = features
        velocity = self.output(features)
        if return_features:
            return velocity, intermediate
        return velocity
