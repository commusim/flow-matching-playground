import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def joint_tsne(feature_sets, seed=42, perplexity=30):
    names = list(feature_sets)
    sizes = [len(feature_sets[name]) for name in names]
    combined = np.concatenate([feature_sets[name] for name in names], axis=0)
    reducer = TSNE(
        n_components=2,
        perplexity=min(perplexity, max(5, (len(combined) - 1) // 3)),
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    embedding = reducer.fit_transform(combined)
    result = {}
    offset = 0
    for name, size in zip(names, sizes):
        result[name] = embedding[offset : offset + size]
        offset += size
    return result


def plot_joint_trajectories(embeddings, labels, time_count, path):
    names = list(embeddings)
    columns = 2
    rows = int(np.ceil(len(names) / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(14, 6.5 * rows), squeeze=False, constrained_layout=True
    )
    colors = plt.cm.tab10(np.arange(10))
    all_points = np.concatenate(list(embeddings.values()), axis=0)
    limits = (
        all_points[:, 0].min(),
        all_points[:, 0].max(),
        all_points[:, 1].min(),
        all_points[:, 1].max(),
    )
    sample_count = len(labels)
    for ax, name in zip(axes.ravel(), names):
        points = embeddings[name].reshape(time_count, sample_count, 2)
        for sample, label in enumerate(labels):
            trajectory = points[:, sample]
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color=colors[label],
                alpha=0.5,
                linewidth=1.1,
            )
            ax.scatter(trajectory[0, 0], trajectory[0, 1], color="#64748B", s=15)
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color=colors[label], s=28)
            ax.text(trajectory[-1, 0], trajectory[-1, 1], str(label), fontsize=7)
        ax.set_title(name, fontweight="bold")
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.grid(alpha=0.18)
        ax.set_xlabel("shared t-SNE 1")
        ax.set_ylabel("shared t-SNE 2")
    for ax in axes.ravel()[len(names) :]:
        ax.axis("off")
    fig.suptitle(
        "Classifier-semantic trajectories using one jointly fitted t-SNE",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_semantic_curves(curves, times, path, ylabel, title):
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for name, values in curves.items():
        ax.plot(times, values, marker="o", markersize=3, label=name)
    ax.set(xlabel="Flow time t", ylabel=ylabel, title=title)
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_endpoint_embedding(embeddings, labels, time_count, path):
    names = list(embeddings)
    sample_count = len(labels)
    fig, axes = plt.subplots(
        1, len(names), figsize=(5 * len(names), 4.5), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    all_endpoints = []
    for name in names:
        points = embeddings[name].reshape(time_count, sample_count, 2)[-1]
        all_endpoints.append(points)
    combined = np.concatenate(all_endpoints)
    limits = (
        combined[:, 0].min(),
        combined[:, 0].max(),
        combined[:, 1].min(),
        combined[:, 1].max(),
    )
    for ax, name, points in zip(axes, names, all_endpoints):
        scatter = ax.scatter(
            points[:, 0], points[:, 1], c=labels, cmap="tab10", vmin=0, vmax=9, s=28
        )
        for point, label in zip(points, labels):
            ax.text(point[0], point[1], str(label), fontsize=7)
        ax.set(title=name, xlim=limits[:2], ylim=limits[2:])
        ax.grid(alpha=0.18)
    axes[-1].legend(
        *scatter.legend_elements(),
        title="label",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    fig.suptitle(
        "Final semantic features in the same t-SNE coordinate system", fontweight="bold"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_label_centroid_trajectories(embeddings, labels, time_count, path):
    names = list(embeddings)
    columns = 2
    rows = int(np.ceil(len(names) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(14, 6.5 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    colors = plt.cm.tab10(np.arange(10))
    all_points = np.concatenate(list(embeddings.values()), axis=0)
    limits = (
        all_points[:, 0].min(),
        all_points[:, 0].max(),
        all_points[:, 1].min(),
        all_points[:, 1].max(),
    )
    sample_count = len(labels)
    unique_labels = np.unique(labels)
    for ax, name in zip(axes.ravel(), names):
        points = embeddings[name].reshape(time_count, sample_count, 2)
        for label in unique_labels:
            mask = labels == label
            centroid = points[:, mask].mean(axis=1)
            ax.plot(
                centroid[:, 0],
                centroid[:, 1],
                color=colors[label],
                linewidth=2.0,
                marker="o",
                markersize=3,
                label=str(label),
            )
            ax.scatter(centroid[0, 0], centroid[0, 1], color="#475569", s=24)
            ax.text(centroid[-1, 0], centroid[-1, 1], str(label), fontsize=9)
        ax.set_title(name, fontweight="bold")
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.grid(alpha=0.18)
        ax.set_xlabel("shared t-SNE 1")
        ax.set_ylabel("shared t-SNE 2")
    for ax in axes.ravel()[len(names) :]:
        ax.axis("off")
    fig.suptitle(
        "Label-centroid semantic trajectories in one shared t-SNE",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)
