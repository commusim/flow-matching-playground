import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _fit_pca(matrix, components=2):
    mean = matrix.mean(axis=0, keepdims=True)
    centered = matrix - mean
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    return mean, vectors[:components].T


def _project(matrix, mean, basis):
    return (matrix - mean) @ basis


def real_class_centroids(real_features, real_labels, class_count):
    return np.stack(
        [
            real_features[real_labels == label].mean(axis=0)
            for label in range(class_count)
        ]
    )


def semantic_dynamics(feature_matrix, labels, times, target_centroids):
    time_count = len(times)
    sample_count = len(labels)
    sequence = feature_matrix.reshape(time_count, sample_count, -1)
    delta_t = np.diff(times)
    velocity = np.diff(sequence, axis=0) / delta_t[:, None, None]
    speed = np.linalg.norm(velocity, axis=2).mean(axis=1)
    target = target_centroids[labels]
    direction = target[None, :, :] - sequence[:-1]
    numerator = (velocity * direction).sum(axis=2)
    denominator = np.linalg.norm(velocity, axis=2) * np.linalg.norm(direction, axis=2)
    alignment = (numerator / np.maximum(denominator, 1e-8)).mean(axis=1)
    distance = np.linalg.norm(sequence - target[None, :, :], axis=2).mean(axis=1)
    return {
        "sequence": sequence,
        "velocity": velocity,
        "speed": speed,
        "alignment": alignment,
        "distance": distance,
    }


def _real_background(ax, real_points, real_labels, class_count):
    colors = plt.cm.get_cmap("tab20", class_count)(np.arange(class_count))
    for label in range(class_count):
        points = real_points[real_labels == label]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            color=colors[label],
            s=8,
            alpha=0.12,
            edgecolors="none",
        )
        center = points.mean(axis=0)
        ax.text(
            center[0],
            center[1],
            str(label),
            color=colors[label],
            fontsize=10,
            fontweight="bold",
        )
    return colors


def plot_semantic_pca_velocity_fields(
    dynamics,
    raw_features,
    real_features,
    real_labels,
    labels,
    times,
    path,
):
    names = list(dynamics)
    combined = np.concatenate([real_features] + [raw_features[name] for name in names])
    mean, basis = _fit_pca(combined)
    real_projected = _project(real_features, mean, basis)
    all_generated = [_project(raw_features[name], mean, basis) for name in names]
    all_points = np.concatenate([real_projected] + all_generated)
    margin_x = (all_points[:, 0].max() - all_points[:, 0].min()) * 0.05
    margin_y = (all_points[:, 1].max() - all_points[:, 1].min()) * 0.05
    limits = (
        all_points[:, 0].min() - margin_x,
        all_points[:, 0].max() + margin_x,
        all_points[:, 1].min() - margin_y,
        all_points[:, 1].max() + margin_y,
    )
    fig, axes = plt.subplots(2, 2, figsize=(15, 13), constrained_layout=True)
    class_count = int(labels.max()) + 1
    for ax, name in zip(axes.ravel(), names):
        colors = _real_background(ax, real_projected, real_labels, class_count)
        sequence = dynamics[name]["sequence"]
        for label in range(class_count):
            mask = labels == label
            centroid_high = sequence[:, mask].mean(axis=1)
            centroid = _project(centroid_high, mean, basis)
            ax.plot(
                centroid[:, 0],
                centroid[:, 1],
                color=colors[label],
                alpha=0.5,
                linewidth=1,
            )
            for index in range(len(times) - 1):
                change = centroid[index + 1] - centroid[index]
                ax.quiver(
                    centroid[index, 0],
                    centroid[index, 1],
                    change[0],
                    change[1],
                    color=colors[label],
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    width=0.003,
                    alpha=0.75,
                )
            ax.scatter(
                centroid[0, 0], centroid[0, 1], color=colors[label], marker="^", s=35
            )
            ax.text(centroid[-1, 0], centroid[-1, 1], str(label), fontsize=8)
        ax.set(
            title=name,
            xlim=limits[:2],
            ylim=limits[2:],
            xlabel="classifier feature PCA 1",
            ylabel="classifier feature PCA 2",
        )
        ax.grid(alpha=0.15)
    fig.suptitle(
        "Classifier-feature semantic velocity fields (linear PCA projection)",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_dynamics_curves(curves, x_values, path, ylabel, title, horizontal_zero=False):
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for name, values in curves.items():
        ax.plot(x_values, values, marker="o", markersize=3, label=name)
    if horizontal_zero:
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set(xlabel="Flow time t", ylabel=ylabel, title=title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)
