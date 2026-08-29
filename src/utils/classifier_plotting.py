import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def plot_confusion_matrix(matrix, path):
    classes = matrix.shape[0]
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for row in range(classes):
        for column in range(classes):
            ax.text(
                column,
                row,
                f"{normalized[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
            )
    ax.set(title="Classifier confusion matrix", xlabel="predicted", ylabel="true")
    ax.set_xticks(range(classes))
    ax.set_yticks(range(classes))
    fig.colorbar(image, ax=ax)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_tsne_features(features, labels, path, seed=42):
    embedding = TSNE(
        n_components=2,
        perplexity=min(30, max(5, len(features) // 20)),
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(features)
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    scatter = ax.scatter(
        embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10", s=9, alpha=0.7
    )
    ax.set(
        title="Supervised semantic features (t-SNE)",
        xlabel="t-SNE 1",
        ylabel="t-SNE 2",
    )
    ax.legend(
        *scatter.legend_elements(),
        title="digit",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    ax.grid(alpha=0.15)
    fig.savefig(path, dpi=180)
    plt.close(fig)
