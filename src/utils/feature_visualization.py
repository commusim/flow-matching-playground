import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
from torchvision.utils import make_grid


def _pca_fit_transform(matrix, components):
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    basis = vectors[:components].T
    return centered @ basis, basis, matrix.mean(axis=0, keepdims=True)


def _spatial_pyramid(feature, levels=(1, 2, 4)):
    pooled = [
        functional.adaptive_avg_pool2d(feature, (level, level)).flatten(1)
        for level in levels
    ]
    return torch.cat(pooled, dim=1)


@torch.no_grad()
def extract_feature_trajectory(
    model, frames, labels, device, layer="block_3", levels=(1, 2, 4)
):
    vectors = []
    feature_maps = []
    for index, frame in enumerate(frames):
        image = frame.to(device)
        time = torch.full((len(image), 1), index / (len(frames) - 1), device=device)
        _, features = model(image, time, labels, return_features=True)
        selected = features[layer]
        vectors.append(_spatial_pyramid(selected, levels).cpu())
        feature_maps.append(selected.cpu())
    return torch.stack(vectors), torch.stack(feature_maps)


def plot_feature_pca_trajectory(vectors, labels, path):
    time_count, sample_count, dimension = vectors.shape
    matrix = vectors.numpy().reshape(time_count * sample_count, dimension)
    projected, _, _ = _pca_fit_transform(matrix, 2)
    projected = projected.reshape(time_count, sample_count, 2)
    labels_np = labels.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    class_count = int(labels.max().item()) + 1
    colors = plt.cm.get_cmap("tab20", class_count)(np.arange(class_count))
    for sample in range(sample_count):
        label = labels_np[sample]
        line = projected[:, sample]
        ax.plot(line[:, 0], line[:, 1], color=colors[label], alpha=0.7, linewidth=1.5)
        ax.scatter(line[0, 0], line[0, 1], color="#94A3B8", s=18)
        ax.scatter(line[-1, 0], line[-1, 1], color=colors[label], s=32)
        ax.text(line[-1, 0], line[-1, 1], str(label), fontsize=8)
    ax.set(title="Spatial-pyramid feature trajectories", xlabel="PCA 1", ylabel="PCA 2")
    ax.grid(alpha=0.25)
    ax.text(
        0.02,
        0.02,
        "gray=start noise; colored endpoint=label-conditioned result",
        transform=ax.transAxes,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_spatial_feature_pca(feature_maps, labels, path, sample_index=0):
    selected = feature_maps[:, sample_index]
    time_count, channels, height, width = selected.shape
    matrix = selected.permute(0, 2, 3, 1).reshape(-1, channels).numpy()
    projected, _, _ = _pca_fit_transform(matrix, 3)
    projected = projected.reshape(time_count, height, width, 3)
    low = np.percentile(projected, 1, axis=(0, 1, 2), keepdims=True)
    high = np.percentile(projected, 99, axis=(0, 1, 2), keepdims=True)
    projected = np.clip((projected - low) / (high - low + 1e-8), 0, 1)
    indices = np.linspace(0, time_count - 1, 8).astype(int)
    fig, axes = plt.subplots(1, len(indices), figsize=(18, 3), constrained_layout=True)
    label = int(labels[sample_index])
    for ax, index in zip(axes, indices):
        ax.imshow(projected[index])
        ax.set_title(f"t={index / (time_count - 1):.2f}")
        ax.axis("off")
    fig.suptitle(
        f"Spatial feature PCA map for label {label} (shared PCA basis across time)",
        fontweight="bold",
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_feature_activation(feature_maps, labels, path, sample_index=0):
    activation = torch.linalg.vector_norm(feature_maps[:, sample_index], dim=1).numpy()
    indices = np.linspace(0, len(activation) - 1, 8).astype(int)
    low, high = np.percentile(activation, (2, 98))
    fig, axes = plt.subplots(1, len(indices), figsize=(18, 3), constrained_layout=True)
    label = int(labels[sample_index])
    for ax, index in zip(axes, indices):
        ax.imshow(activation[index], cmap="magma", vmin=low, vmax=high)
        ax.set_title(f"t={index / (len(activation) - 1):.2f}")
        ax.axis("off")
    fig.suptitle(f"Feature activation strength for label {label}", fontweight="bold")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_label_conditioned_samples(images, labels, path, samples_per_class):
    order = torch.argsort(labels.detach().cpu())
    sorted_images = images.detach().cpu()[order].clamp(-1, 1)
    grid = make_grid((sorted_images + 1) / 2, nrow=samples_per_class, padding=2)
    fig, ax = plt.subplots(
        figsize=(max(6, samples_per_class * 1.5), 14), constrained_layout=True
    )
    ax.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    ax.set_title(
        f"Label-conditioned generation: rows 0 through {int(labels.max())}",
        fontweight="bold",
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)
