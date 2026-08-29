import torch
from torch import nn


class MNISTVAE(nn.Module):
    def __init__(self, latent_channels=8):
        super().__init__()
        self.latent_channels = latent_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.SiLU(),
        )
        self.to_mean = nn.Conv2d(64, latent_channels, 1)
        self.to_logvar = nn.Conv2d(64, latent_channels, 1)
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 64, 3, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, images, sample=True):
        features = self.encoder(images)
        mean = self.to_mean(features)
        logvar = self.to_logvar(features).clamp(-12, 8)
        if not sample:
            return mean
        return mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)

    def decode(self, latent):
        return self.decoder(latent)

    def forward(self, images):
        features = self.encoder(images)
        mean = self.to_mean(features)
        logvar = self.to_logvar(features).clamp(-12, 8)
        latent = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        return self.decode(latent), mean, logvar


class LatentConditionalVelocityCNN(nn.Module):
    def __init__(self, latent_channels=8, hidden=64, condition_dim=64, classes=10):
        super().__init__()
        from src.modules.image_velocity import (
            ConditionalResidualBlock,
            SinusoidalTimeEmbedding,
        )

        self.time_embedding = SinusoidalTimeEmbedding(32)
        self.label_embedding = nn.Embedding(classes, 32)
        self.condition = nn.Sequential(
            nn.Linear(64, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
            nn.SiLU(),
        )
        self.input = nn.Conv2d(latent_channels, hidden, 3, padding=1)
        self.blocks = nn.ModuleList(
            [ConditionalResidualBlock(hidden, condition_dim) for _ in range(4)]
        )
        self.output = nn.Conv2d(hidden, latent_channels, 3, padding=1)

    def forward(self, latent, time, labels):
        condition = self.condition(
            torch.cat([self.time_embedding(time), self.label_embedding(labels)], dim=1)
        )
        features = self.input(latent)
        for block in self.blocks:
            features = block(features, condition)
        return self.output(torch.nn.functional.silu(features))
