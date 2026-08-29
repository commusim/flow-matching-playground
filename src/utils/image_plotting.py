import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from torchvision.utils import make_grid


def _to_grid(images, columns=8):
    images = images.detach().cpu().clamp(-1, 1)
    return make_grid((images + 1) / 2, nrow=columns, padding=2).permute(1, 2, 0).numpy()


def plot_image_overview(real, noise, generated, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    for ax, images, title in zip(
        axes,
        (real, noise, generated),
        ("Real MNIST", "Gaussian noise", "Generated samples"),
    ):
        ax.imshow(_to_grid(images), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontweight="bold")
        ax.axis("off")
    fig.suptitle("Pixel-space Flow Matching", fontsize=16, fontweight="bold")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_image_trajectory(frames, path, samples=8):
    indices = np.linspace(0, len(frames) - 1, 8).astype(int)
    fig, axes = plt.subplots(1, len(indices), figsize=(18, 3), constrained_layout=True)
    for ax, index in zip(axes, indices):
        ax.imshow(
            _to_grid(frames[index][:samples], columns=samples),
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        ax.set_title(f"t={index / (len(frames) - 1):.2f}")
        ax.axis("off")
    fig.suptitle(
        "ODE sampling trajectory: noise gradually becomes digit images",
        fontweight="bold",
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


@torch.no_grad()
def plot_predicted_clean(model, frames, device, path, samples=8, labels=None):
    indices = np.linspace(0, len(frames) - 2, 8).astype(int)
    fig, axes = plt.subplots(1, len(indices), figsize=(18, 3), constrained_layout=True)
    for ax, index in zip(axes, indices):
        state = frames[index][:samples].to(device)
        time_value = index / (len(frames) - 1)
        time = torch.full((len(state), 1), time_value, device=device)
        velocity = (
            model(state, time)
            if labels is None
            else model(state, time, labels[: len(state)])
        )
        clean = state + (1 - time_value) * velocity
        ax.imshow(_to_grid(clean, columns=samples), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"prediction at t={time_value:.2f}")
        ax.axis("off")
    fig.suptitle("Predicted clean images at different times", fontweight="bold")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_image_animation(frames, path, samples=16):
    keep = np.linspace(0, len(frames) - 1, min(50, len(frames))).astype(int)
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    artist = ax.imshow(
        _to_grid(frames[0][:samples], columns=4), cmap="gray", vmin=0, vmax=1
    )
    ax.axis("off")
    title = ax.set_title("")

    def update(frame_number):
        index = keep[frame_number]
        artist.set_data(_to_grid(frames[index][:samples], columns=4))
        title.set_text(
            f"Image Flow Matching sampling, t={index / (len(frames) - 1):.2f}"
        )
        return artist, title

    FuncAnimation(fig, update, frames=len(keep), interval=90).save(
        path, writer=PillowWriter(fps=12)
    )
    plt.close(fig)
