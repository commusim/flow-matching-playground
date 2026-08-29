import math
import torch
from torch import nn


class TimeEmbedding(nn.Module):
    def __init__(self, frequencies=8):
        super().__init__()
        self.register_buffer(
            "frequencies", 2.0 ** torch.arange(frequencies).float() * math.pi
        )

    @property
    def output_dim(self):
        return 2 * len(self.frequencies) + 1

    def forward(self, time):
        angles = time * self.frequencies[None, :]
        return torch.cat([time, torch.sin(angles), torch.cos(angles)], dim=1)


class VelocityMLP(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.time_embedding = TimeEmbedding()
        self.network = nn.Sequential(
            nn.Linear(2 + self.time_embedding.output_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, points, time):
        return self.network(torch.cat([points, self.time_embedding(time)], dim=1))


class ConditionalVelocityMLP(nn.Module):
    def __init__(self, hidden=128, condition_dim=16, conditions=2):
        super().__init__()
        self.time_embedding = TimeEmbedding()
        self.condition_embedding = nn.Embedding(conditions, condition_dim)
        input_dim = 2 + self.time_embedding.output_dim + condition_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, points, time, condition):
        features = torch.cat(
            [points, self.time_embedding(time), self.condition_embedding(condition)],
            dim=1,
        )
        return self.network(features)
