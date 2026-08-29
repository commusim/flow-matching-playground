import torch


@torch.no_grad()
def euler_integrate(model, initial, steps, condition=None):
    model.eval()
    x = initial.clone()
    frames = [x.detach().cpu()]
    dt = 1.0 / steps
    for index in range(steps):
        time = torch.full((len(x), 1), index / steps, device=x.device)
        velocity = model(x, time) if condition is None else model(x, time, condition)
        x = x + dt * velocity
        frames.append(x.detach().cpu())
    return frames
