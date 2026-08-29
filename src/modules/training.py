import numpy as np
import torch
from src.modules.flow import flow_matching_loss, linear_flow_batch


def train_unconditional(model, data_sampler, noise_sampler, device, config):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=1e-5
    )
    losses = []
    for step in range(1, config["steps"] + 1):
        data = data_sampler(config["batch_size"], device)
        noise = noise_sampler(config["batch_size"], device)
        time = torch.rand(config["batch_size"], 1, device=device)
        state, target = linear_flow_batch(data, noise, time)
        loss = flow_matching_loss(model(state, time), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step == 1 or step % max(1, config["steps"] // 10) == 0:
            print(f"step {step}/{config['steps']} loss {np.mean(losses[-100:]):.5f}")
    return losses


def train_conditional(model, batch_builder, device, config):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=1e-5
    )
    losses = []
    transport = []
    for step in range(1, config["steps"] + 1):
        data, noise, condition, extra = batch_builder(config["batch_size"], device)
        time = torch.rand(config["batch_size"], 1, device=device)
        state, target = linear_flow_batch(data, noise, time)
        loss = flow_matching_loss(model(state, time, condition), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if extra is not None:
            transport.append(extra)
        if step == 1 or step % max(1, config["steps"] // 10) == 0:
            print(f"step {step}/{config['steps']} loss {np.mean(losses[-100:]):.5f}")
    return losses, transport
