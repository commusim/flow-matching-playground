"""A compact, visual 2D Flow Matching experiment.

The convention used here is noise at t=0 and data at t=1:
    x_t = (1-t) * x_noise + t * x_data
    dx_t/dt = x_data - x_noise
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


COLORS = {
    "noise": "#94A3B8",
    "data": "#EF4444",
    "generated": "#2563EB",
    "path": "#7C3AED",
    "grid": "#CBD5E1",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_moons(n, device, noise_std=0.07):
    """Generate a two-moons dataset without requiring scikit-learn."""
    n1 = n // 2
    n2 = n - n1
    a1 = torch.rand(n1, device=device) * math.pi
    a2 = torch.rand(n2, device=device) * math.pi
    moon1 = torch.stack([torch.cos(a1), torch.sin(a1)], dim=1)
    moon2 = torch.stack([1.0 - torch.cos(a2), 0.5 - torch.sin(a2)], dim=1)
    x = torch.cat([moon1, moon2], dim=0)
    x = x + noise_std * torch.randn_like(x)
    x[:, 0] -= 0.5
    x[:, 1] -= 0.25
    return x * 1.65


def sample_noise(n, device):
    return torch.randn(n, 2, device=device) * 1.15


class TimeEmbedding(nn.Module):
    def __init__(self, frequencies=8):
        super().__init__()
        freq = 2.0 ** torch.arange(frequencies).float() * math.pi
        self.register_buffer("freq", freq)

    @property
    def output_dim(self):
        return 2 * len(self.freq) + 1

    def forward(self, t):
        angles = t * self.freq[None, :]
        return torch.cat([t, torch.sin(angles), torch.cos(angles)], dim=1)


class VelocityMLP(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.time_embedding = TimeEmbedding(8)
        dim = 2 + self.time_embedding.output_dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, self.time_embedding(t)], dim=1))


def train(model, device, steps, batch_size, lr, log_every):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    losses = []
    model.train()
    for step in range(1, steps + 1):
        x_data = sample_moons(batch_size, device)
        x_noise = sample_noise(batch_size, device)
        t = torch.rand(batch_size, 1, device=device)

        # Conditional straight path and its exact conditional velocity.
        x_t = (1.0 - t) * x_noise + t * x_data
        target_velocity = x_data - x_noise
        predicted_velocity = model(x_t, t)
        loss = torch.mean((predicted_velocity - target_velocity) ** 2)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

        if step == 1 or step % log_every == 0:
            recent = np.mean(losses[-min(log_every, len(losses)) :])
            print("step {:5d}/{:5d} | loss {:.6f}".format(step, steps, recent))
    return losses


@torch.no_grad()
def integrate_ode(model, initial, ode_steps):
    """Euler integration of dx/dt = v_theta(x,t), retaining all frames."""
    model.eval()
    x = initial.clone()
    trajectory = [x.detach().cpu()]
    dt = 1.0 / ode_steps
    for i in range(ode_steps):
        t = torch.full((len(x), 1), i / ode_steps, device=x.device)
        velocity = model(x, t)
        x = x + dt * velocity
        trajectory.append(x.detach().cpu())
    return trajectory


def style_axis(ax, title):
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-2.8, 2.8)
    ax.set_aspect("equal")
    ax.grid(True, color=COLORS["grid"], alpha=0.25, linewidth=0.7)
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])


def scatter(ax, points, color, label=None, alpha=0.65, size=7):
    p = points.detach().cpu().numpy() if torch.is_tensor(points) else points
    ax.scatter(
        p[:, 0],
        p[:, 1],
        s=size,
        c=color,
        alpha=alpha,
        edgecolors="none",
        label=label,
        rasterized=True,
    )


@torch.no_grad()
def plot_overview(model, device, output_path, particles):
    n = min(particles, 1600)
    x_data = sample_moons(n, device)
    x_noise = sample_noise(n, device)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)

    scatter(axes[0, 0], x_data, COLORS["data"])
    style_axis(axes[0, 0], "1. Target distribution  x_data")
    scatter(axes[0, 1], x_noise, COLORS["noise"])
    style_axis(axes[0, 1], "2. Gaussian source  x_noise")

    for t_value, color in zip(
        [0.0, 0.25, 0.5, 0.75, 1.0],
        ["#94A3B8", "#60A5FA", "#A78BFA", "#FB7185", "#EF4444"],
    ):
        xt = (1.0 - t_value) * x_noise + t_value * x_data
        scatter(
            axes[0, 2],
            xt[:350],
            color,
            label="t={:.2f}".format(t_value),
            alpha=0.42,
            size=6,
        )
    style_axis(axes[0, 2], "3. Conditional straight paths  x_t")
    axes[0, 2].legend(frameon=False, fontsize=8, markerscale=1.7)

    # Show paired conditional target velocities. These are noisy because each
    # point is paired with a random endpoint; the network learns their mean.
    mid_t = 0.5
    xt = (1.0 - mid_t) * x_noise + mid_t * x_data
    true_v = x_data - x_noise
    ids = torch.linspace(0, n - 1, 120).long().to(device)
    qx = xt[ids].cpu().numpy()
    qv = true_v[ids].cpu().numpy()
    scatter(axes[1, 0], xt, "#DDD6FE", alpha=0.25, size=5)
    axes[1, 0].quiver(
        qx[:, 0],
        qx[:, 1],
        qv[:, 0],
        qv[:, 1],
        color="#7C3AED",
        alpha=0.70,
        angles="xy",
        scale_units="xy",
        scale=8,
    )
    style_axis(
        axes[1, 0], "4. Training targets at t=0.5\n(pairwise conditional velocities)"
    )

    gx, gy = np.meshgrid(np.linspace(-2.7, 2.7, 18), np.linspace(-2.3, 2.3, 16))
    grid = torch.tensor(
        np.stack([gx.ravel(), gy.ravel()], axis=1), dtype=torch.float32, device=device
    )
    tgrid = torch.full((len(grid), 1), 0.5, device=device)
    learned = model(grid, tgrid).cpu().numpy()
    speed = np.linalg.norm(learned, axis=1)
    axes[1, 1].quiver(
        grid[:, 0].cpu(),
        grid[:, 1].cpu(),
        learned[:, 0],
        learned[:, 1],
        speed,
        cmap="viridis",
        alpha=0.9,
        angles="xy",
        scale_units="xy",
        scale=7,
    )
    style_axis(axes[1, 1], "5. Learned marginal velocity field at t=0.5")

    trajectory = integrate_ode(model, sample_noise(n, device), 80)
    scatter(axes[1, 2], x_data, COLORS["data"], label="target", alpha=0.20, size=7)
    scatter(
        axes[1, 2],
        trajectory[-1],
        COLORS["generated"],
        label="generated",
        alpha=0.62,
        size=7,
    )
    style_axis(axes[1, 2], "6. ODE result at t=1")
    axes[1, 2].legend(frameon=False, fontsize=9, markerscale=2)

    fig.suptitle(
        "2D Flow Matching: learn a time-dependent velocity field",
        fontsize=17,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


@torch.no_grad()
def plot_trajectory(trajectory, target, output_path):
    indices = np.linspace(0, len(trajectory) - 1, 8).astype(int)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    for ax, idx in zip(axes.ravel(), indices):
        scatter(ax, target, COLORS["data"], alpha=0.12, size=6)
        scatter(ax, trajectory[idx], COLORS["generated"], alpha=0.62, size=7)
        t = idx / float(len(trajectory) - 1)
        style_axis(ax, "t = {:.2f}".format(t))
    fig.suptitle(
        "ODE sampling: particles follow the learned vector field",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.005,
        "Red shadow = target distribution    Blue = transported particles",
        ha="center",
        fontsize=10,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


@torch.no_grad()
def plot_velocity_fields_over_time(model, device, trajectory, output_path):
    """Visualize the learned time-dependent vector field at nine times."""
    times = np.linspace(0.0, 1.0, 9)
    gx, gy = np.meshgrid(np.linspace(-2.8, 2.8, 19), np.linspace(-2.4, 2.4, 17))
    grid_np = np.stack([gx.ravel(), gy.ravel()], axis=1)
    grid = torch.tensor(grid_np, dtype=torch.float32, device=device)

    fig, axes = plt.subplots(3, 3, figsize=(13, 12), constrained_layout=True)
    quivers = []
    for ax, t_value in zip(axes.ravel(), times):
        tgrid = torch.full((len(grid), 1), float(t_value), device=device)
        velocity = model(grid, tgrid).cpu().numpy()
        speed = np.linalg.norm(velocity, axis=1)

        frame_idx = int(round(t_value * (len(trajectory) - 1)))
        particles = trajectory[frame_idx].numpy()
        ax.scatter(
            particles[:, 0],
            particles[:, 1],
            s=5,
            c="#CBD5E1",
            alpha=0.28,
            edgecolors="none",
            rasterized=True,
        )
        q = ax.quiver(
            grid_np[:, 0],
            grid_np[:, 1],
            velocity[:, 0],
            velocity[:, 1],
            speed,
            cmap="viridis",
            clim=(0.0, 3.0),
            angles="xy",
            scale_units="xy",
            scale=8.0,
            alpha=0.92,
            width=0.004,
        )
        quivers.append(q)
        style_axis(ax, "t = {:.3f}".format(t_value))

    cbar = fig.colorbar(quivers[-1], ax=axes.ravel().tolist(), shrink=0.78, pad=0.02)
    cbar.set_label("Predicted speed  ||v_theta(x,t)||")
    fig.suptitle(
        "The learned velocity field changes with time", fontsize=17, fontweight="bold"
    )
    fig.text(
        0.5,
        0.008,
        "Gray particles show the generated distribution at each time; arrows show local velocity.",
        ha="center",
        fontsize=10,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_loss(losses, output_path):
    values = np.asarray(losses)
    window = max(1, min(100, len(values) // 20))
    smooth = np.convolve(values, np.ones(window) / window, mode="valid")
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    ax.plot(values, color="#93C5FD", alpha=0.35, linewidth=0.8, label="raw")
    ax.plot(
        np.arange(window - 1, len(values)),
        smooth,
        color="#1D4ED8",
        linewidth=2,
        label="moving average",
    )
    ax.set_title("Flow Matching velocity regression loss", fontweight="bold")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean squared error")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_animation(trajectory, target, output_path):
    # Downsample frames so GIF creation remains fast and readable.
    keep = np.linspace(0, len(trajectory) - 1, min(50, len(trajectory))).astype(int)
    target_np = target.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    ax.scatter(
        target_np[:, 0],
        target_np[:, 1],
        s=8,
        c=COLORS["data"],
        alpha=0.12,
        edgecolors="none",
    )
    current = ax.scatter(
        [], [], s=10, c=COLORS["generated"], alpha=0.72, edgecolors="none"
    )
    title = ax.set_title("")
    style_axis(ax, "")
    ax.text(
        0.02,
        0.02,
        "red shadow: target",
        transform=ax.transAxes,
        fontsize=9,
        color="#B91C1C",
    )

    def update(frame_number):
        idx = keep[frame_number]
        pts = trajectory[idx].numpy()
        current.set_offsets(pts)
        title.set_text(
            "Flow Matching ODE sampling   t={:.2f}".format(
                idx / float(len(trajectory) - 1)
            )
        )
        return current, title

    animation = FuncAnimation(fig, update, frames=len(keep), interval=90, blit=False)
    animation.save(output_path, writer=PillowWriter(fps=12))
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Visual 2D Flow Matching demo")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--particles", type=int, default=1400)
    parser.add_argument("--ode-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--no-animation", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print("device:", device)
    print("Convention: t=0 noise -> t=1 data")

    model = VelocityMLP(args.hidden).to(device)
    losses = train(
        model,
        device,
        args.steps,
        args.batch_size,
        args.lr,
        log_every=max(1, args.steps // 20),
    )
    torch.save(
        {"model": model.state_dict(), "args": vars(args)},
        os.path.join(args.output_dir, "checkpoint.pt"),
    )

    initial = sample_noise(args.particles, device)
    target = sample_moons(args.particles, device)
    trajectory = integrate_ode(model, initial, args.ode_steps)

    plot_loss(losses, os.path.join(args.output_dir, "loss_curve.png"))
    plot_overview(
        model,
        device,
        os.path.join(args.output_dir, "flow_matching_overview.png"),
        args.particles,
    )
    plot_trajectory(
        trajectory, target, os.path.join(args.output_dir, "sampling_trajectory.png")
    )
    plot_velocity_fields_over_time(
        model,
        device,
        trajectory,
        os.path.join(args.output_dir, "velocity_fields_over_time.png"),
    )
    if not args.no_animation:
        make_animation(
            trajectory, target, os.path.join(args.output_dir, "flow_animation.gif")
        )

    print(
        "\nDone. Open the files in '{}' to inspect the learned flow.".format(
            args.output_dir
        )
    )


if __name__ == "__main__":
    main()
