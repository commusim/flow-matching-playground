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


class UnconditionalMNISTVelocityCNN(nn.Module):
    """Original unconditional MNIST velocity model."""

    def __init__(self, hidden=64, time_dim=32, input_channels=1):
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(time_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.input = nn.Conv2d(input_channels, hidden, 3, padding=1)
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
            nn.Conv2d(hidden, input_channels, 3, padding=1),
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


class AdditiveConditionalMNISTVelocityCNN(nn.Module):
    """Original label-conditioned model with one additive condition injection."""

    def __init__(
        self, hidden=64, time_dim=32, label_dim=32, classes=10, input_channels=1
    ):
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
        self.input = nn.Conv2d(input_channels, hidden, 3, padding=1)
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
            nn.Conv2d(hidden, input_channels, 3, padding=1),
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


class AdaGNConditionalMNISTVelocityCNN(nn.Module):
    """Label-conditioned model with AdaGN modulation in every residual block."""

    def __init__(
        self, hidden=64, time_dim=32, label_dim=32, classes=10, input_channels=1
    ):
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
        self.input = nn.Conv2d(input_channels, hidden, 3, padding=1)
        self.blocks = nn.ModuleList(
            [ConditionalResidualBlock(hidden, hidden) for _ in range(4)]
        )
        self.output_norm = nn.GroupNorm(8, hidden)
        self.output_modulation = nn.Linear(hidden, 2 * hidden)
        self.output_conv = nn.Conv2d(hidden, input_channels, 3, padding=1)
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


MNISTVelocityCNN = UnconditionalMNISTVelocityCNN
ConditionalMNISTVelocityCNN = AdaGNConditionalMNISTVelocityCNN


class AdaGNResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, condition_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.mod1 = nn.Linear(condition_dim, 2 * in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.mod2 = nn.Linear(condition_dim, 2 * out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    @staticmethod
    def _modulate(features, condition, norm, modulation):
        scale, shift = modulation(condition).chunk(2, dim=1)
        return norm(features) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

    def forward(self, features, condition):
        residual = self.conv1(
            torch.nn.functional.silu(
                self._modulate(features, condition, self.norm1, self.mod1)
            )
        )
        residual = self.conv2(
            torch.nn.functional.silu(
                self._modulate(residual, condition, self.norm2, self.mod2)
            )
        )
        return self.skip(features) + residual


class ConditionalImageUNet(nn.Module):
    """Configurable multi-scale U-Net velocity model for image-space Flow Matching."""

    def __init__(self, input_channels=1, base=32, classes=11, condition_dim=128):
        super().__init__()
        if base % 8 != 0:
            raise ValueError("base must be divisible by 8 for GroupNorm")
        self.null_label = classes - 1
        self.time_embedding = SinusoidalTimeEmbedding(32)
        self.label_embedding = nn.Embedding(classes, 32)
        self.condition = nn.Sequential(
            nn.Linear(64, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
            nn.SiLU(),
        )
        self.input = nn.Conv2d(input_channels, base, 3, padding=1)
        self.enc1 = AdaGNResBlock(base, base, condition_dim)
        self.down1 = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)
        self.enc2 = AdaGNResBlock(base * 2, base * 2, condition_dim)
        self.down2 = nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1)
        self.middle = AdaGNResBlock(base * 4, base * 4, condition_dim)
        self.dec2 = AdaGNResBlock(base * 6, base * 2, condition_dim)
        self.dec1 = AdaGNResBlock(base * 3, base, condition_dim)
        self.output = nn.Conv2d(base, input_channels, 3, padding=1)

    def _condition(self, time, labels):
        return self.condition(
            torch.cat([self.time_embedding(time), self.label_embedding(labels)], dim=1)
        )

    def forward(self, image, time, labels, return_features=False):
        condition = self._condition(time, labels)
        skip1 = self.enc1(self.input(image), condition)
        skip2 = self.enc2(self.down1(skip1), condition)
        middle = self.middle(self.down2(skip2), condition)
        up2 = torch.nn.functional.interpolate(
            middle, size=skip2.shape[-2:], mode="nearest"
        )
        up2 = self.dec2(torch.cat([up2, skip2], dim=1), condition)
        up1 = torch.nn.functional.interpolate(
            up2, size=skip1.shape[-2:], mode="nearest"
        )
        up1 = self.dec1(torch.cat([up1, skip1], dim=1), condition)
        velocity = self.output(torch.nn.functional.silu(up1))
        if return_features:
            return velocity, {
                "encoder_full": skip1,
                "encoder_half": skip2,
                "bottleneck": middle,
                "decoder_half": up2,
                "decoder_full": up1,
            }
        return velocity


ConditionalMNISTUNet = ConditionalImageUNet
