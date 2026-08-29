"""Conditional 2D Flow Matching: a visual learning demo.

One shared network learns two condition-dependent transports:
  condition 0 -> two moons
  condition 1 -> ring
The same initial noise is used for both conditions in comparisons.
"""

import argparse
import math
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch
from torch import nn

RED, BLUE, GRAY = "#EF4444", "#2563EB", "#94A3B8"


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_targets(n, labels, device):
    """Sample target points according to each condition label."""
    out = torch.empty(n, 2, device=device)
    mask0 = labels == 0
    n0 = int(mask0.sum())
    if n0:
        half = n0 // 2
        a1 = torch.rand(half, device=device) * math.pi
        a2 = torch.rand(n0 - half, device=device) * math.pi
        m1 = torch.stack([torch.cos(a1), torch.sin(a1)], 1)
        m2 = torch.stack([1 - torch.cos(a2), 0.5 - torch.sin(a2)], 1)
        moons = torch.cat([m1, m2], 0)
        moons[:, 0] -= 0.5
        moons[:, 1] -= 0.25
        moons = 1.65 * (moons + 0.07 * torch.randn_like(moons))
        out[mask0] = moons[torch.randperm(n0, device=device)]
    mask1 = ~mask0
    n1 = int(mask1.sum())
    if n1:
        angle = torch.rand(n1, device=device) * 2 * math.pi
        radius = 1.35 + 0.08 * torch.randn(n1, device=device)
        out[mask1] = torch.stack(
            [radius * torch.cos(angle), radius * torch.sin(angle)], 1
        )
    return out


class ConditionalVelocityMLP(nn.Module):
    def __init__(self, hidden=128, condition_dim=16):
        super().__init__()
        self.condition_embedding = nn.Embedding(2, condition_dim)
        self.net = nn.Sequential(
            nn.Linear(2 + 1 + condition_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, t, condition):
        c = self.condition_embedding(condition)
        return self.net(torch.cat([x, t, c], 1))


def train(model, device, steps, batch_size=768):
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    losses = []
    model.train()
    for step in range(1, steps + 1):
        condition = torch.randint(0, 2, (batch_size,), device=device)
        data = sample_targets(batch_size, condition, device)
        noise = torch.randn_like(data) * 1.15
        t = torch.rand(batch_size, 1, device=device)
        xt = (1 - t) * noise + t * data
        target_velocity = data - noise
        loss = ((model(xt, t, condition) - target_velocity) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
        if step == 1 or step % max(1, steps // 10) == 0:
            print("step %d/%d  loss %.5f" % (step, steps, np.mean(losses[-100:])))
    return losses


@torch.no_grad()
def integrate(model, initial, condition_id, ode_steps, device):
    x = initial.clone().to(device)
    c = torch.full((len(x),), condition_id, dtype=torch.long, device=device)
    frames = [x.cpu()]
    dt = 1.0 / ode_steps
    for i in range(ode_steps):
        t = torch.full((len(x), 1), i / ode_steps, device=device)
        x = x + dt * model(x, t, c)
        frames.append(x.cpu())
    return frames


def decorate(ax, title):
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.grid(alpha=0.18)
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])


def scatter(ax, points, color, alpha=0.55, size=7, label=None):
    p = points.detach().cpu().numpy() if torch.is_tensor(points) else points
    ax.scatter(
        p[:, 0], p[:, 1], s=size, c=color, alpha=alpha, edgecolors="none", label=label
    )


@torch.no_grad()
def velocity_grid(model, device, time, condition):
    gx, gy = np.meshgrid(np.linspace(-2.7, 2.7, 19), np.linspace(-2.5, 2.5, 17))
    grid_np = np.c_[gx.ravel(), gy.ravel()]
    grid = torch.tensor(grid_np, dtype=torch.float32, device=device)
    tt = torch.full((len(grid), 1), float(time), device=device)
    cc = torch.full((len(grid),), condition, dtype=torch.long, device=device)
    v = model(grid, tt, cc).cpu().numpy()
    return gx, gy, grid_np, v


def make_overview(model, device, initial, trajectories, targets, path):
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5), constrained_layout=True)
    for row, (name, color) in enumerate([("Two moons", RED), ("Ring", BLUE)]):
        scatter(axes[row, 0], initial, GRAY)
        decorate(axes[row, 0], name + ": same initial noise")
        scatter(axes[row, 1], targets[row], color)
        decorate(axes[row, 1], name + ": target")
        scatter(axes[row, 2], targets[row], color, alpha=0.15)
        scatter(axes[row, 2], trajectories[row][-1], color)
        decorate(axes[row, 2], name + ": generated")
        gx, gy, _, v = velocity_grid(model, device, 0.5, row)
        speed = np.linalg.norm(v, axis=1)
        axes[row, 3].quiver(
            gx,
            gy,
            v[:, 0],
            v[:, 1],
            speed,
            cmap="viridis",
            scale=7,
            angles="xy",
            scale_units="xy",
        )
        decorate(axes[row, 3], name + ": velocity field at t=0.5")
    fig.suptitle(
        "Conditional Flow Matching: one model, condition selects a velocity field",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_trajectory_grid(trajectories, targets, path):
    indices = np.linspace(0, len(trajectories[0]) - 1, 8).astype(int)
    fig, axes = plt.subplots(2, 8, figsize=(22, 6), constrained_layout=True)
    for row, (name, color) in enumerate([("moons", RED), ("ring", BLUE)]):
        for col, idx in enumerate(indices):
            scatter(axes[row, col], targets[row], color, alpha=0.12, size=5)
            scatter(axes[row, col], trajectories[row][idx], color, alpha=0.58, size=6)
            decorate(
                axes[row, col],
                "%s  t=%.2f" % (name, idx / (len(trajectories[row]) - 1)),
            )
    fig.suptitle(
        "The same noise follows two different condition-dependent ODE trajectories",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_velocity_time_grid(model, device, trajectories, path):
    times = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    fig, axes = plt.subplots(2, 6, figsize=(19, 6.8), constrained_layout=True)
    last_q = None
    for row, (name, color) in enumerate([("moons", RED), ("ring", BLUE)]):
        for col, time in enumerate(times):
            gx, gy, _, v = velocity_grid(model, device, time, row)
            speed = np.linalg.norm(v, axis=1)
            idx = int(round(time * (len(trajectories[row]) - 1)))
            scatter(axes[row, col], trajectories[row][idx], color, alpha=0.13, size=4)
            last_q = axes[row, col].quiver(
                gx,
                gy,
                v[:, 0],
                v[:, 1],
                speed,
                cmap="viridis",
                clim=(0, 3),
                scale=7,
                angles="xy",
                scale_units="xy",
            )
            decorate(axes[row, col], "%s  t=%.1f" % (name, time))
    fig.colorbar(last_q, ax=axes.ravel().tolist(), shrink=0.72, label="speed")
    fig.suptitle(
        "Each condition defines an entire time-varying family of velocity fields",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_difference_fields(model, device, path):
    times = [0, 0.25, 0.5, 0.75, 1.0]
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.8), constrained_layout=True)
    for ax, time in zip(axes, times):
        gx, gy, _, v0 = velocity_grid(model, device, time, 0)
        _, _, _, v1 = velocity_grid(model, device, time, 1)
        delta = v0 - v1
        mag = np.linalg.norm(delta, axis=1)
        ax.quiver(
            gx,
            gy,
            delta[:, 0],
            delta[:, 1],
            mag,
            cmap="magma",
            scale=7,
            angles="xy",
            scale_units="xy",
        )
        decorate(ax, "v(moons)-v(ring)\nt=%.2f" % time)
    fig.suptitle(
        "Condition effect: difference between the two predicted velocity fields",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_paired_paths(initial, trajectories, path):
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ids = np.linspace(0, len(initial) - 1, 28).astype(int)
    scatter(ax, initial[ids], GRAY, alpha=0.8, size=22, label="same starting noise")
    for idx in ids:
        p0 = np.array([frame[idx].numpy() for frame in trajectories[0]])
        p1 = np.array([frame[idx].numpy() for frame in trajectories[1]])
        ax.plot(p0[:, 0], p0[:, 1], color=RED, alpha=0.38, linewidth=1)
        ax.plot(p1[:, 0], p1[:, 1], color=BLUE, alpha=0.38, linewidth=1)
    scatter(
        ax, trajectories[0][-1][ids], RED, alpha=0.8, size=18, label="moons endpoints"
    )
    scatter(
        ax, trajectories[1][-1][ids], BLUE, alpha=0.8, size=18, label="ring endpoints"
    )
    decorate(ax, "Same particles split into different paths when condition changes")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_animation(trajectories, targets, path):
    keep = np.linspace(
        0, len(trajectories[0]) - 1, min(45, len(trajectories[0]))
    ).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    scatters = []
    for row, (name, color) in enumerate(
        [("condition 0: moons", RED), ("condition 1: ring", BLUE)]
    ):
        scatter(axes[row], targets[row], color, alpha=0.10, size=6)
        sc = axes[row].scatter([], [], s=8, c=color, alpha=0.7, edgecolors="none")
        scatters.append(sc)
        decorate(axes[row], name)
    title = fig.suptitle("")

    def update(frame):
        idx = keep[frame]
        for row in range(2):
            scatters[row].set_offsets(trajectories[row][idx].numpy())
        title.set_text(
            "Same noise, different condition   t=%.2f"
            % (idx / (len(trajectories[0]) - 1))
        )
        return scatters + [title]

    FuncAnimation(fig, update, frames=len(keep), interval=90).save(
        path, writer=PillowWriter(fps=12)
    )
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--particles", type=int, default=1000)
    p.add_argument("--ode-steps", type=int, default=80)
    p.add_argument("--output-dir", default="outputs_conditional")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-animation", action="store_true")
    args = p.parse_args()
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    model = ConditionalVelocityMLP().to(device)
    losses = train(model, device, args.steps)
    torch.save(
        model.state_dict(), os.path.join(args.output_dir, "conditional_checkpoint.pt")
    )
    initial = torch.randn(args.particles, 2) * 1.15
    trajectories = [
        integrate(model, initial, 0, args.ode_steps, device),
        integrate(model, initial, 1, args.ode_steps, device),
    ]
    targets = [
        sample_targets(
            args.particles,
            torch.zeros(args.particles, dtype=torch.long, device=device),
            device,
        ),
        sample_targets(
            args.particles,
            torch.ones(args.particles, dtype=torch.long, device=device),
            device,
        ),
    ]
    make_overview(
        model,
        device,
        initial,
        trajectories,
        targets,
        os.path.join(args.output_dir, "01_conditional_overview.png"),
    )
    make_trajectory_grid(
        trajectories,
        targets,
        os.path.join(args.output_dir, "02_trajectories_over_time.png"),
    )
    make_velocity_time_grid(
        model,
        device,
        trajectories,
        os.path.join(args.output_dir, "03_velocity_fields_over_time.png"),
    )
    make_difference_fields(
        model,
        device,
        os.path.join(args.output_dir, "04_condition_difference_fields.png"),
    )
    make_paired_paths(
        initial,
        trajectories,
        os.path.join(args.output_dir, "05_same_noise_paired_paths.png"),
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, alpha=0.2)
    w = min(100, max(1, len(losses) // 10))
    ax.plot(
        np.arange(w - 1, len(losses)),
        np.convolve(losses, np.ones(w) / w, "valid"),
        color=BLUE,
    )
    ax.set(title="Conditional Flow Matching loss", xlabel="step", ylabel="MSE")
    ax.grid(alpha=0.2)
    fig.savefig(os.path.join(args.output_dir, "06_loss.png"), dpi=180)
    plt.close(fig)
    if not args.no_animation:
        make_animation(
            trajectories,
            targets,
            os.path.join(args.output_dir, "07_same_noise_two_conditions.gif"),
        )
    print("device:", device)
    print("saved visualizations to", args.output_dir)


if __name__ == "__main__":
    main()
