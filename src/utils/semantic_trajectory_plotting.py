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


def _limits(*point_sets):
    points = np.concatenate(point_sets, axis=0)
    margin_x = max(1.0, (points[:, 0].max() - points[:, 0].min()) * 0.05)
    margin_y = max(1.0, (points[:, 1].max() - points[:, 1].min()) * 0.05)
    return (
        points[:, 0].min() - margin_x,
        points[:, 0].max() + margin_x,
        points[:, 1].min() - margin_y,
        points[:, 1].max() + margin_y,
    )


def _real_background(ax, real_points, real_labels, class_count):
    colors = plt.cm.get_cmap("tab20", class_count)(np.arange(class_count))
    for label in range(class_count):
        selected = real_points[real_labels == label]
        ax.scatter(
            selected[:, 0],
            selected[:, 1],
            color=colors[label],
            s=8,
            alpha=0.12,
            edgecolors="none",
        )
        centroid = selected.mean(axis=0)
        ax.text(
            centroid[0],
            centroid[1],
            str(label),
            color=colors[label],
            fontsize=11,
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.55, "edgecolor": "none"},
        )
    return colors


def plot_centroid_trajectories_with_reference(
    embeddings,
    real_embedding,
    real_labels,
    expected_labels,
    final_predictions,
    final_known,
    time_count,
    path,
):
    names = list(embeddings)
    columns = 2
    rows = int(np.ceil(len(names) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(15, 7 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    class_count = int(max(real_labels.max(), expected_labels.max())) + 1
    all_generated = np.concatenate(list(embeddings.values()), axis=0)
    limits = _limits(real_embedding, all_generated)
    sample_count = len(expected_labels)
    unique_labels = np.unique(expected_labels)
    for ax, name in zip(axes.ravel(), names):
        colors = _real_background(ax, real_embedding, real_labels, class_count)
        points = embeddings[name].reshape(time_count, sample_count, 2)
        predictions = final_predictions[name]
        known = final_known[name]
        for label in unique_labels:
            mask = expected_labels == label
            centroid = points[:, mask].mean(axis=1)
            ax.plot(
                centroid[:, 0],
                centroid[:, 1],
                color=colors[label],
                linewidth=2.3,
                marker="o",
                markersize=3,
            )
            ax.scatter(
                centroid[0, 0],
                centroid[0, 1],
                color=colors[label],
                marker="^",
                s=42,
                edgecolors="black",
                linewidths=0.5,
                zorder=5,
            )
            known_predictions = predictions[mask & known]
            if len(known_predictions):
                majority = int(
                    np.bincount(known_predictions, minlength=class_count).argmax()
                )
                terminal_text = f"target {label}→pred {majority}"
            else:
                terminal_text = f"target {label}→unknown"
            ax.text(centroid[-1, 0], centroid[-1, 1], terminal_text, fontsize=7)
        ax.set(
            title=name,
            xlim=limits[:2],
            ylim=limits[2:],
            xlabel="shared t-SNE 1",
            ylabel="shared t-SNE 2",
        )
        ax.grid(alpha=0.15)
    for ax in axes.ravel()[len(names) :]:
        ax.axis("off")
    fig.suptitle(
        "Label-colored starts and classifier-semantic trajectories against real clusters",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_final_predictions_with_reference(
    embeddings,
    real_embedding,
    real_labels,
    expected_labels,
    predictions,
    known_masks,
    time_count,
    path,
):
    names = list(embeddings)
    fig, axes = plt.subplots(
        1,
        len(names),
        figsize=(5.2 * len(names), 5),
        squeeze=False,
        constrained_layout=True,
    )
    class_count = int(max(real_labels.max(), expected_labels.max())) + 1
    all_endpoints = [
        embeddings[name].reshape(time_count, len(expected_labels), 2)[-1]
        for name in names
    ]
    limits = _limits(real_embedding, *all_endpoints)
    for ax, name, endpoints in zip(axes.ravel(), names, all_endpoints):
        colors = _real_background(ax, real_embedding, real_labels, class_count)
        predicted = predictions[name]
        known = known_masks[name]
        correct = known & (predicted == expected_labels)
        incorrect = known & (predicted != expected_labels)
        for mask, marker, size in ((correct, "o", 38), (incorrect, "s", 42)):
            if mask.any():
                ax.scatter(
                    endpoints[mask, 0],
                    endpoints[mask, 1],
                    c=colors[predicted[mask]],
                    marker=marker,
                    s=size,
                    edgecolors="black" if marker == "s" else "none",
                    linewidths=0.7,
                    alpha=0.9,
                )
        unknown = ~known
        if unknown.any():
            ax.scatter(
                endpoints[unknown, 0],
                endpoints[unknown, 1],
                color="black",
                marker="x",
                s=45,
                linewidths=1.2,
            )
        ax.set(
            title=name,
            xlim=limits[:2],
            ylim=limits[2:],
            xlabel="shared t-SNE 1",
            ylabel="shared t-SNE 2",
        )
        ax.grid(alpha=0.15)
    fig.suptitle(
        "Final generated samples versus real clusters: circle=correct, square=wrong, x=unknown",
        fontsize=14,
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


def plot_unknown_confusions(confusions, path):
    names = list(confusions)
    fig, axes = plt.subplots(
        1,
        len(names),
        figsize=(5.4 * len(names), 4.8),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, name in zip(axes.ravel(), names):
        matrix = confusions[name].astype(float)
        normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("classifier prediction (last column=unknown)")
        ax.set_ylabel("target label")
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_xticklabels([str(i) for i in range(matrix.shape[1] - 1)] + ["?"])
        ax.set_yticks(range(matrix.shape[0]))
        fig.colorbar(image, ax=ax, shrink=0.75)
    fig.suptitle(
        "Unknown-aware conditional generation confusion matrices", fontweight="bold"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)
