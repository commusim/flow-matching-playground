import math
import torch


def linear_flow_batch(data, noise, time):
    state = (1 - time) * noise + time * data
    target_velocity = data - noise
    return state, target_velocity


def flow_matching_loss(predicted, target):
    return torch.mean((predicted - target) ** 2)


def sinkhorn_coupling(source, target, regularization=0.12, iterations=60):
    cost = torch.cdist(source, target).pow(2)
    log_kernel = -cost / regularization
    n, m = cost.shape
    log_source = torch.full((n,), -math.log(n), device=source.device)
    log_target = torch.full((m,), -math.log(m), device=source.device)
    source_scale = torch.zeros_like(log_source)
    target_scale = torch.zeros_like(log_target)
    for _ in range(iterations):
        source_scale = log_source - torch.logsumexp(
            log_kernel + target_scale[None, :], dim=1
        )
        target_scale = log_target - torch.logsumexp(
            log_kernel + source_scale[:, None], dim=0
        )
    plan = torch.exp(log_kernel + source_scale[:, None] + target_scale[None, :])
    probabilities = plan / (plan.sum(dim=1, keepdim=True) + 1e-12)
    indices = torch.multinomial(probabilities, 1).squeeze(1)
    return target[indices], cost.detach(), plan.detach()
