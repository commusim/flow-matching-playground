import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

COLORS = {
    "noise": "#94A3B8",
    "data": "#EF4444",
    "generated": "#2563EB",
    "grid": "#CBD5E1",
}


def style_axis(ax, title):
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-2.8, 2.8)
    ax.set_aspect("equal")
    ax.grid(True, color=COLORS["grid"], alpha=0.25)


def scatter(ax, points, color, alpha=0.6, size=7, label=None):
    values = points.detach().cpu().numpy() if torch.is_tensor(points) else points
    ax.scatter(
        values[:, 0],
        values[:, 1],
        s=size,
        c=color,
        alpha=alpha,
        edgecolors="none",
        label=label,
    )


def plot_loss(losses, path):
    values = np.asarray(losses)
    window = max(1, min(100, len(values) // 10))
    smooth = np.convolve(values, np.ones(window) / window, mode="valid")
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(values, alpha=0.25, label="raw")
    ax.plot(
        np.arange(window - 1, len(values)), smooth, linewidth=2, label="moving average"
    )
    ax.set(title="Flow Matching loss", xlabel="step", ylabel="MSE")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_trajectory(frames, target, path, color="#2563EB", title="ODE trajectory"):
    indices = np.linspace(0, len(frames) - 1, 8).astype(int)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    for ax, index in zip(axes.ravel(), indices):
        scatter(ax, target, COLORS["data"], alpha=0.12, size=5)
        scatter(ax, frames[index], color, alpha=0.62, size=6)
        style_axis(ax, f"t={index / (len(frames) - 1):.2f}")
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def velocity_grid(model, device, time, condition=None):
    gx, gy = np.meshgrid(np.linspace(-2.7, 2.7, 18), np.linspace(-2.3, 2.3, 16))
    points = torch.tensor(
        np.c_[gx.ravel(), gy.ravel()], dtype=torch.float32, device=device
    )
    times = torch.full((len(points), 1), float(time), device=device)
    if condition is None:
        velocity = model(points, times)
    else:
        labels = torch.full((len(points),), condition, dtype=torch.long, device=device)
        velocity = model(points, times, labels)
    return gx, gy, velocity.detach().cpu().numpy()


def plot_velocity_times(model, device, path, conditions=None):
    times = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    conditions = [None] if conditions is None else list(conditions)
    fig, axes = plt.subplots(
        len(conditions),
        len(times),
        figsize=(19, 3.6 * len(conditions)),
        squeeze=False,
        constrained_layout=True,
    )
    for row, condition in enumerate(conditions):
        for column, time in enumerate(times):
            gx, gy, velocity = velocity_grid(model, device, time, condition)
            speed = np.linalg.norm(velocity, axis=1)
            axes[row, column].quiver(
                gx, gy, velocity[:, 0], velocity[:, 1], speed, cmap="viridis", scale=7
            )
            label = "unconditional" if condition is None else f"condition={condition}"
            style_axis(axes[row, column], f"{label}, t={time:.1f}")
    fig.suptitle("Time-dependent velocity fields", fontsize=15, fontweight="bold")
    fig.savefig(path, dpi=180)
    plt.close(fig)
