import torch
from torch import nn
from src.modules.image_velocity import SinusoidalTimeEmbedding


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

    def _modulate(self, x, condition, norm, mod):
        scale, shift = mod(condition).chunk(2, 1)
        return norm(x) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

    def forward(self, x, condition):
        residual = self.conv1(
            torch.nn.functional.silu(
                self._modulate(x, condition, self.norm1, self.mod1)
            )
        )
        residual = self.conv2(
            torch.nn.functional.silu(
                self._modulate(residual, condition, self.norm2, self.mod2)
            )
        )
        return self.skip(x) + residual


class ConditionalMNISTUNet(nn.Module):
    """Small multi-scale U-Net velocity model with per-block AdaGN conditioning."""

    def __init__(self, base=32, classes=11, condition_dim=128):
        super().__init__()
        self.null_label = classes - 1
        self.time_embedding = SinusoidalTimeEmbedding(32)
        self.label_embedding = nn.Embedding(classes, 32)
        self.condition = nn.Sequential(
            nn.Linear(64, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
            nn.SiLU(),
        )
        self.input = nn.Conv2d(1, base, 3, padding=1)
        self.enc1 = AdaGNResBlock(base, base, condition_dim)
        self.down1 = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)
        self.enc2 = AdaGNResBlock(base * 2, base * 2, condition_dim)
        self.down2 = nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1)
        self.middle = AdaGNResBlock(base * 4, base * 4, condition_dim)
        self.dec2 = AdaGNResBlock(base * 4 + base * 2, base * 2, condition_dim)
        self.dec1 = AdaGNResBlock(base * 2 + base, base, condition_dim)
        self.output = nn.Conv2d(base, 1, 3, padding=1)

    def _condition(self, time, labels):
        return self.condition(
            torch.cat([self.time_embedding(time), self.label_embedding(labels)], 1)
        )

    def forward(self, image, time, labels, return_features=False):
        condition = self._condition(time, labels)
        skip1 = self.enc1(self.input(image), condition)
        skip2 = self.enc2(self.down1(skip1), condition)
        middle = self.middle(self.down2(skip2), condition)
        up2 = torch.nn.functional.interpolate(middle, scale_factor=2, mode="nearest")
        up2 = self.dec2(torch.cat([up2, skip2], 1), condition)
        up1 = torch.nn.functional.interpolate(up2, scale_factor=2, mode="nearest")
        up1 = self.dec1(torch.cat([up1, skip1], 1), condition)
        velocity = self.output(torch.nn.functional.silu(up1))
        if return_features:
            return velocity, {
                "encoder_28": skip1,
                "encoder_14": skip2,
                "bottleneck_7": middle,
                "decoder_14": up2,
                "decoder_28": up1,
            }
        return velocity
