import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


def _grid(images, nrow=10):
    return (
        make_grid((images.detach().cpu().clamp(-1, 1) + 1) / 2, nrow=nrow, padding=2)
        .permute(1, 2, 0)
        .numpy()
    )


def plot_vae_reconstruction(real, reconstructed, path):
    count = min(20, len(real))
    combined = torch.cat([real[:count], reconstructed[:count]])
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    ax.imshow(_grid(combined, nrow=10), cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    ax.set_title(
        "Top rows: real MNIST | Bottom rows: VAE reconstruction", fontweight="bold"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_latent_channels(latent, path, sample=0):
    channels = latent[sample].detach().cpu()
    cols = 4
    rows = (len(channels) + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(10, 2.5 * rows), constrained_layout=True
    )
    axes = axes.ravel()
    for index, ax in enumerate(axes):
        if index < len(channels):
            ax.imshow(channels[index], cmap="coolwarm")
            ax.set_title(f"channel {index}")
        ax.axis("off")
    fig.suptitle("VAE latent channels (7×7 spatial representation)", fontweight="bold")
    fig.savefig(path, dpi=180)
    plt.close(fig)
