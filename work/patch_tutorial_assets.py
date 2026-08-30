from pathlib import Path

path = Path(r"C:\Code\SIQA\ImageTest\scripts\build_tutorial_assets.py")
text = path.read_text(encoding="utf-8")
text = text.replace("from torch import nn\n", "from torch import nn\nfrom sklearn.manifold import TSNE\n")
text = text.replace("TIMES = np.linspace(0, 1, 11)", "TIMES = np.linspace(0, 1, 41)")
text = text.replace(
    "    base = torch.randn(1, 8, 7, 7, generator=generator)\n    latent = base.repeat(10, 1, 1, 1)",
    "    class_count = int(labels.max()) + 1\n    samples = len(labels) // class_count\n    base = torch.randn(samples, 8, 7, 7, generator=generator)\n    latent = base.repeat(class_count, 1, 1, 1)",
)
text = text.replace(
    "    canvas = np.zeros((10 * 28, 11 * 28), dtype=np.uint8)",
    "    canvas = np.zeros((10 * 28, len(TIMES) * 28), dtype=np.uint8)",
)
start = text.index("def build_manifold_assets():")
end = text.index('\n\nif __name__ == "__main__":', start)
new_function = r'''def build_manifold_assets():
    reference_output = ROOT / "outputs/semantic_trajectory_comparison/20260830_114100_seed42"
    reference = np.load(reference_output / "classifier_semantic_features.npz")
    real_features = reference["real_features"]
    real_labels = reference["real_labels"]
    keep = np.concatenate([np.where(real_labels == label)[0][:50] for label in range(10)])
    classifier = load_classifier(
        ROOT / "outputs/mnist_classifier/20260829_154612_seed42/checkpoint.pt",
        DEVICE,
    )
    labels = torch.arange(10, dtype=torch.long).repeat_interleave(3)
    generator = torch.Generator().manual_seed(2048)
    base_noise = torch.randn(3, 1, 28, 28, generator=generator)
    shared_noise = base_noise.repeat(10, 1, 1, 1)
    additive, _ = load_pixel_conditional(
        ROOT / "outputs/mnist_conditional_flow/20260829_135059_seed42/checkpoint.pt",
        DEVICE,
    )
    adagn, _ = load_pixel_conditional(
        ROOT / "outputs/mnist_conditional_flow/20260829_145627_seed42/checkpoint.pt",
        DEVICE,
    )
    unet, _ = load_unet(
        ROOT / "outputs/mnist_unet_flow/20260829_165551_seed42/checkpoint.pt",
        DEVICE,
    )
    latent, vae, _ = load_latent_flow(
        ROOT / "outputs/mnist_latent_flow/20260829_160611_seed42/checkpoint.pt",
        ROOT / "outputs/mnist_vae/20260829_155108_seed42/checkpoint.pt",
        DEVICE,
    )
    frame_sets = {
        "additive_condition": pixel_frames(additive, shared_noise, labels),
        "adagn_condition": pixel_frames(adagn, shared_noise, labels),
        "latent_flow": latent_frames(latent, vae, labels),
        "conditional_unet": unet_frames(unet, shared_noise, labels),
    }
    frame_indices = (TIMES * 80).round().astype(int)
    semantic_features = {}
    with torch.no_grad():
        for name, frames in frame_sets.items():
            batches = []
            for index in frame_indices:
                _, features = classifier(frames[index], return_features=True)
                batches.append(features.cpu().numpy())
            semantic_features[name] = np.stack(batches)
    model_names = list(semantic_features)
    combined_for_tsne = np.concatenate(
        [real_features[keep]]
        + [semantic_features[name].reshape(-1, 128) for name in model_names]
    )
    embedding = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=42,
    ).fit_transform(combined_for_tsne)
    offset = len(keep)
    manifold = {
        "times": rounded(TIMES),
        "real": {
            "points": rounded(embedding[:offset]),
            "labels": real_labels[keep].tolist(),
        },
        "expected_labels": labels.tolist(),
        "models": {},
    }
    for name in model_names:
        size = len(TIMES) * len(labels)
        manifold["models"][name] = rounded(
            embedding[offset : offset + size].reshape(len(TIMES), len(labels), 2)
        )
        offset += size
    (ASSETS / "semantic_tsne.json").write_text(
        json.dumps(manifold, separators=(",", ":")), encoding="utf-8"
    )
    all_semantic = np.concatenate(
        [real_features]
        + [semantic_features[name].reshape(-1, 128) for name in model_names]
    )
    mean = all_semantic.mean(axis=0, keepdims=True)
    _, _, vectors = np.linalg.svd(all_semantic - mean, full_matrices=False)
    basis = vectors[:2].T

    def project(value):
        return (value - mean) @ basis

    real_projected = project(real_features)
    velocity_asset = {
        "times": rounded(TIMES),
        "real": {
            "points": rounded(real_projected[keep]),
            "labels": real_labels[keep].tolist(),
        },
        "models": {},
    }
    labels_numpy = labels.numpy()
    real_centroids = np.stack(
        [real_features[real_labels == label].mean(axis=0) for label in range(10)]
    )
    for name in model_names:
        sequence = semantic_features[name]
        centroids = np.stack(
            [sequence[:, labels_numpy == label].mean(axis=1) for label in range(10)],
            axis=1,
        )
        projected = project(centroids.reshape(-1, 128)).reshape(len(TIMES), 10, 2)
        velocity = np.diff(projected, axis=0) / np.diff(TIMES)[:, None, None]
        target = real_centroids[labels_numpy]
        high_velocity = np.diff(sequence, axis=0) / np.diff(TIMES)[:, None, None]
        direction = target[None] - sequence[:-1]
        alignment = (
            (high_velocity * direction).sum(axis=2)
            / np.maximum(
                np.linalg.norm(high_velocity, axis=2)
                * np.linalg.norm(direction, axis=2),
                1e-8,
            )
        ).mean(axis=1)
        distance = np.linalg.norm(sequence - target[None], axis=2).mean(axis=1)
        speed = np.linalg.norm(high_velocity, axis=2).mean(axis=1)
        velocity_asset["models"][name] = {
            "centroids": rounded(projected),
            "velocity": rounded(velocity),
            "speed": rounded(speed),
            "alignment": rounded(alignment),
            "distance": rounded(distance),
        }
    (ASSETS / "semantic_velocity.json").write_text(
        json.dumps(velocity_asset, separators=(",", ":")), encoding="utf-8"
    )
'''
text = text[:start] + new_function + text[end:]
path.write_text(text, encoding="utf-8")
