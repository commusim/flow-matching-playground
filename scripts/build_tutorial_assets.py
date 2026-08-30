"""Build static assets for the interactive GitHub Pages tutorial."""
# ruff: noqa: E402

import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image
from torch import nn
from sklearn.manifold import TSNE
from umap import UMAP

from src.modules.velocity import ConditionalVelocityMLP
from src.modules.trajectory_model_loader import (
    load_latent_flow,
    load_pixel_conditional,
    load_unet,
)
from src.utils.semantic_evaluation import load_classifier

ASSETS = ROOT / "docs" / "assets"
DEVICE = torch.device("cpu")
TIMES = np.linspace(0, 1, 41)


def rounded(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value).round(4).tolist()


class LegacyTimeEmbedding(nn.Module):
    def __init__(self, frequencies=8):
        super().__init__()
        self.register_buffer("freq", 2.0 ** torch.arange(frequencies).float() * math.pi)

    def forward(self, time):
        angles = time * self.freq[None, :]
        return torch.cat([time, torch.sin(angles), torch.cos(angles)], dim=1)


class LegacyUnconditional(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.time_embedding = LegacyTimeEmbedding()
        self.net = nn.Sequential(
            nn.Linear(19, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, points, time):
        return self.net(torch.cat([points, self.time_embedding(time)], dim=1))


class LegacyConditional(nn.Module):
    def __init__(self, hidden=128, condition_dim=16):
        super().__init__()
        self.condition_embedding = nn.Embedding(2, condition_dim)
        self.net = nn.Sequential(
            nn.Linear(3 + condition_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, points, time, condition):
        return self.net(
            torch.cat([points, time, self.condition_embedding(condition)], dim=1)
        )


def sample_moons(n, generator):
    first = n // 2
    a1 = torch.rand(first, generator=generator) * math.pi
    a2 = torch.rand(n - first, generator=generator) * math.pi
    moon1 = torch.stack([torch.cos(a1), torch.sin(a1)], dim=1)
    moon2 = torch.stack([1 - torch.cos(a2), 0.5 - torch.sin(a2)], dim=1)
    points = torch.cat([moon1, moon2])
    points += 0.07 * torch.randn(points.shape, generator=generator)
    points[:, 0] -= 0.5
    points[:, 1] -= 0.25
    return points * 1.65


def sample_ring(n, generator):
    angle = torch.rand(n, generator=generator) * 2 * math.pi
    radius = 1.35 + 0.08 * torch.randn(n, generator=generator)
    return torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], dim=1)


@torch.no_grad()
def integrate_2d(model, initial, condition=None, steps=100):
    point = initial.clone()
    frames = [point.clone()]
    dt = 1 / steps
    for index in range(steps):
        time = torch.full((len(point), 1), index / steps)
        if condition is None:
            velocity = model(point, time)
        else:
            labels = torch.full((len(point),), condition, dtype=torch.long)
            velocity = model(point, time, labels)
        point = point + dt * velocity
        frames.append(point.clone())
    return frames


def build_2d_assets():
    generator = torch.Generator().manual_seed(42)
    initial = torch.randn(140, 2, generator=generator) * 1.15
    target_moons = sample_moons(300, generator)
    target_ring = sample_ring(300, generator)
    unconditional_data = torch.load(
        ROOT / "outputs/unconditional_2d/checkpoint.pt", map_location="cpu"
    )
    unconditional = LegacyUnconditional(unconditional_data["args"]["hidden"])
    unconditional.load_state_dict(unconditional_data["model"])
    conditional = ConditionalVelocityMLP(hidden=128)
    conditional.load_state_dict(
        torch.load(
            ROOT / "outputs/conditional_2d/20260830_135438_seed42/checkpoint.pt",
            map_location="cpu",
        )
    )
    models = {
        "unconditional_moons": (unconditional, None, target_moons),
        "conditional_moons": (conditional, 0, target_moons),
        "conditional_ring": (conditional, 1, target_ring),
    }
    gx, gy = np.meshgrid(np.linspace(-2.7, 2.7, 15), np.linspace(-2.4, 2.4, 13))
    grid = torch.tensor(np.c_[gx.ravel(), gy.ravel()], dtype=torch.float32)
    result = {"times": rounded(TIMES), "grid": rounded(grid), "modes": {}}
    for name, (model, condition, target) in models.items():
        frames = integrate_2d(model, initial, condition, 100)
        frame_indices = (TIMES * 100).round().astype(int)
        positions = [frames[index] for index in frame_indices]
        velocities = []
        with torch.no_grad():
            for time_value in TIMES:
                time = torch.full((len(grid), 1), float(time_value))
                if condition is None:
                    velocity = model(grid, time)
                else:
                    labels = torch.full((len(grid),), condition, dtype=torch.long)
                    velocity = model(grid, time, labels)
                velocities.append(velocity)
        result["modes"][name] = {
            "positions": rounded(torch.stack(positions)),
            "velocities": rounded(torch.stack(velocities)),
            "target": rounded(target),
        }
    (ASSETS / "flow_2d.json").write_text(
        json.dumps(result, separators=(",", ":")), encoding="utf-8"
    )


@torch.no_grad()
def pixel_frames(model, initial, labels, steps=80):
    image = initial.clone()
    frames = [image.cpu()]
    dt = 1 / steps
    for index in range(steps):
        time = torch.full((len(image), 1), index / steps)
        image = image + dt * model(image, time, labels)
        frames.append(image.cpu())
    return frames


@torch.no_grad()
def unet_frames(model, initial, labels, steps=80, guidance=3.0):
    image = initial.clone()
    frames = [image.cpu()]
    dt = 1 / steps
    for index in range(steps):
        time = torch.full((len(image), 1), index / steps)
        conditional = model(image, time, labels)
        null = torch.full_like(labels, model.null_label)
        unconditional = model(image, time, null)
        image = image + dt * (unconditional + guidance * (conditional - unconditional))
        frames.append(image.cpu())
    return frames


@torch.no_grad()
def latent_frames(flow, vae, labels, steps=80):
    generator = torch.Generator().manual_seed(2026)
    class_count = int(labels.max()) + 1
    samples = len(labels) // class_count
    base = torch.randn(samples, 8, 7, 7, generator=generator)
    latent = base.repeat(class_count, 1, 1, 1)
    frames = [vae.decode(latent).cpu()]
    dt = 1 / steps
    for index in range(steps):
        time = torch.full((len(latent), 1), index / steps)
        latent = latent + dt * flow(latent, time, labels)
        frames.append(vae.decode(latent).cpu())
    return frames


def save_sprite(frames, path):
    selected = [frames[int(round(time * 80))] for time in TIMES]
    canvas = np.zeros((10 * 28, len(TIMES) * 28), dtype=np.uint8)
    for column, batch in enumerate(selected):
        values = ((batch[:, 0].clamp(-1, 1) + 1) * 127.5).numpy().astype(np.uint8)
        for row in range(10):
            canvas[row * 28 : (row + 1) * 28, column * 28 : (column + 1) * 28] = values[
                row
            ]
    Image.fromarray(canvas, mode="L").save(path)


def save_condition_grid(images, path):
    canvas = np.zeros((10 * 28, 4 * 28), dtype=np.uint8)
    values = (
        ((images[:, 0].clamp(-1, 1) + 1) * 127.5)
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )
    for label in range(10):
        for sample in range(4):
            index = label * 4 + sample
            canvas[label * 28 : (label + 1) * 28, sample * 28 : (sample + 1) * 28] = (
                values[index]
            )
    Image.fromarray(canvas, mode="L").save(path)


def build_mnist_assets():
    classifier = load_classifier(
        ROOT / "outputs/mnist_classifier/20260829_154612_seed42/checkpoint.pt", DEVICE
    )
    labels = torch.arange(10, dtype=torch.long)
    generator = torch.Generator().manual_seed(2026)
    base_noise = torch.randn(1, 1, 28, 28, generator=generator)
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
        ROOT / "outputs/mnist_unet_flow/20260829_165551_seed42/checkpoint.pt", DEVICE
    )
    latent, vae, _ = load_latent_flow(
        ROOT / "outputs/mnist_latent_flow/20260829_160611_seed42/checkpoint.pt",
        ROOT / "outputs/mnist_vae/20260829_155108_seed42/checkpoint.pt",
        DEVICE,
    )
    frame_sets = {
        "additive": pixel_frames(additive, shared_noise, labels),
        "adagn": pixel_frames(adagn, shared_noise, labels),
        "latent": latent_frames(latent, vae, labels),
        "unet": unet_frames(unet, shared_noise, labels),
    }
    predictions = {"times": rounded(TIMES), "models": {}}
    for name, frames in frame_sets.items():
        save_sprite(frames, ASSETS / f"mnist_{name}_sprites.png")
        model_predictions = []
        for time_value in TIMES:
            frame = frames[int(round(time_value * 80))]
            logits = classifier(frame)
            probabilities = logits.softmax(dim=1)
            confidence, predicted = probabilities.max(dim=1)
            model_predictions.append(
                {
                    "predicted": predicted.tolist(),
                    "confidence": rounded(confidence),
                    "target_confidence": rounded(probabilities[range(10), labels]),
                }
            )
        predictions["models"][name] = model_predictions
    (ASSETS / "mnist_predictions.json").write_text(
        json.dumps(predictions, separators=(",", ":")), encoding="utf-8"
    )
    condition_labels = torch.arange(10, dtype=torch.long).repeat_interleave(4)
    condition_generator = torch.Generator().manual_seed(2030)
    condition_noise = torch.randn(4, 1, 28, 28, generator=condition_generator).repeat(
        10, 1, 1, 1
    )
    condition_sets = {
        "additive": pixel_frames(additive, condition_noise, condition_labels),
        "adagn": pixel_frames(adagn, condition_noise, condition_labels),
        "latent": latent_frames(latent, vae, condition_labels),
        "unet": unet_frames(unet, condition_noise, condition_labels),
    }
    for name, frames in condition_sets.items():
        save_condition_grid(frames[-1], ASSETS / f"mnist_{name}_conditions.png")


def build_manifold_assets():
    reference_output = (
        ROOT / "outputs/semantic_trajectory_comparison/20260830_114100_seed42"
    )
    reference = np.load(reference_output / "classifier_semantic_features.npz")
    real_features = reference["real_features"]
    real_labels = reference["real_labels"]
    keep = np.concatenate(
        [np.where(real_labels == label)[0][:150] for label in range(10)]
    )
    classifier = load_classifier(
        ROOT / "outputs/mnist_classifier/20260829_154612_seed42/checkpoint.pt",
        DEVICE,
    )
    labels = torch.arange(10, dtype=torch.long).repeat_interleave(10)
    generator = torch.Generator().manual_seed(2048)
    base_noise = torch.randn(10, 1, 28, 28, generator=generator)
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
    all_semantic = np.concatenate(
        [real_features]
        + [semantic_features[name].reshape(-1, 128) for name in model_names]
    )
    mean = all_semantic.mean(axis=0, keepdims=True)
    _, _, vectors = np.linalg.svd(all_semantic - mean, full_matrices=False)
    basis = vectors[:2].T

    def project(value):
        return (value - mean) @ basis

    combined = np.concatenate(
        [real_features[keep]]
        + [semantic_features[name].reshape(-1, 128) for name in model_names]
    )
    pca_embedding = project(combined)
    umap_embedding = UMAP(
        n_neighbors=30,
        min_dist=0.08,
        metric="euclidean",
        random_state=42,
        n_jobs=1,
    ).fit_transform(combined)
    tsne_embedding = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        method="barnes_hut",
        random_state=42,
    ).fit_transform(combined)

    def split_embedding(embedding):
        offset = len(keep)
        reduction = {
            "real": {
                "points": rounded(embedding[:offset]),
                "labels": real_labels[keep].tolist(),
            },
            "models": {},
        }
        for name in model_names:
            size = len(TIMES) * len(labels)
            reduction["models"][name] = rounded(
                embedding[offset : offset + size].reshape(len(TIMES), len(labels), 2)
            )
            offset += size
        return reduction

    reductions = {
        "times": rounded(TIMES),
        "expected_labels": labels.tolist(),
        "samples_per_class": 10,
        "reducers": {
            "pca": split_embedding(pca_embedding),
            "umap": split_embedding(umap_embedding),
            "tsne": split_embedding(tsne_embedding),
        },
    }
    (ASSETS / "semantic_reductions.json").write_text(
        json.dumps(reductions, separators=(",", ":")), encoding="utf-8"
    )
    old_tsne = ASSETS / "semantic_tsne.json"
    if old_tsne.exists():
        old_tsne.unlink()

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


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    build_2d_assets()
    build_mnist_assets()
    build_manifold_assets()
    print(f"Tutorial assets written to {ASSETS}")
