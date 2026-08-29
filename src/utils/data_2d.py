import math
import torch


def sample_noise(n, device, scale=1.15):
    return torch.randn(n, 2, device=device) * scale


def sample_moons(n, device, noise_std=0.07):
    n1 = n // 2
    a1 = torch.rand(n1, device=device) * math.pi
    a2 = torch.rand(n - n1, device=device) * math.pi
    moon1 = torch.stack([torch.cos(a1), torch.sin(a1)], dim=1)
    moon2 = torch.stack([1 - torch.cos(a2), 0.5 - torch.sin(a2)], dim=1)
    points = torch.cat([moon1, moon2], dim=0)
    points += noise_std * torch.randn_like(points)
    points[:, 0] -= 0.5
    points[:, 1] -= 0.25
    return points * 1.65


def sample_ring(n, device, noise_std=0.08):
    angle = torch.rand(n, device=device) * 2 * math.pi
    radius = 1.35 + noise_std * torch.randn(n, device=device)
    return torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], dim=1)


def sample_by_condition(labels, device):
    result = torch.empty(len(labels), 2, device=device)
    for condition, sampler in ((0, sample_moons), (1, sample_ring)):
        mask = labels == condition
        count = int(mask.sum())
        if count:
            result[mask] = sampler(count, device)
    return result
