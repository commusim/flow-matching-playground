from pathlib import Path
import numpy as np
import torch
from src.modules.trajectory_model_loader import (
    load_latent_flow,
    load_pixel_conditional,
    load_unet,
)
from src.utils.common import (
    create_run_dir,
    get_device,
    project_root,
    save_run_metadata,
    seed_everything,
)
from src.utils.image_data import create_image_loader
from src.utils.ode import euler_integrate
from src.utils.semantic_evaluation import load_classifier
from src.utils.semantic_trajectory_plotting import (
    joint_tsne,
    plot_centroid_trajectories_with_reference,
    plot_final_predictions_with_reference,
    plot_semantic_curves,
    plot_unknown_confusions,
)

DEFAULTS = {
    "pipeline": "semantic_trajectory_comparison",
    "seed": 42,
    "device": "auto",
    "dataset": "mnist",
    "image_size": 28,
    "num_classes": 10,
    "samples_per_class": 10,
    "real_samples_per_class": 100,
    "ode_steps": 80,
    "time_points": 11,
    "guidance_scale": 3.0,
    "confidence_threshold": 0.8,
    "classifier_checkpoint": None,
    "experiments": {},
    "output_root": None,
}


def _resolve(path):
    if path is None:
        return None
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root() / candidate


def _guided_unet_frames(model, initial, labels, steps, scale):
    image = initial.clone()
    frames = [image.cpu()]
    dt = 1 / steps
    with torch.no_grad():
        for index in range(steps):
            time = torch.full((len(image), 1), index / steps, device=image.device)
            conditional = model(image, time, labels)
            null = torch.full_like(labels, model.null_label)
            unconditional = model(image, time, null)
            image = image + dt * (unconditional + scale * (conditional - unconditional))
            frames.append(image.cpu())
    return frames


def _latent_frames(flow, vae, labels, steps, image_size, latent_channels, device):
    class_count = int(labels.max()) + 1
    samples = len(labels) // class_count
    latent_size = image_size // 4
    base = torch.randn(
        samples, latent_channels, latent_size, latent_size, device=device
    )
    latent = base.repeat(class_count, 1, 1, 1)
    frames = []
    dt = 1 / steps
    with torch.no_grad():
        frames.append(vae.decode(latent).cpu())
        for index in range(steps):
            time = torch.full((len(latent), 1), index / steps, device=device)
            latent = latent + dt * flow(latent, time, labels)
            frames.append(vae.decode(latent).cpu())
    return frames


def _collect_real_features(classifier, cfg, device):
    loader = create_image_loader(
        project_root() / "data",
        256,
        dataset=cfg["dataset"],
        train=False,
        image_size=cfg["image_size"],
        input_channels=1,
        download=False,
    )
    features = []
    labels = []
    counts = np.zeros(cfg["num_classes"], dtype=int)
    with torch.no_grad():
        for images, batch_labels in loader:
            _, batch_features = classifier(images.to(device), return_features=True)
            for feature, label in zip(
                batch_features.cpu().numpy(), batch_labels.numpy()
            ):
                if (
                    label < cfg["num_classes"]
                    and counts[label] < cfg["real_samples_per_class"]
                ):
                    features.append(feature)
                    labels.append(label)
                    counts[label] += 1
            if np.all(counts >= cfg["real_samples_per_class"]):
                break
    return np.asarray(features), np.asarray(labels)


def _semantic_sequence(classifier, frames, labels, indices, threshold, device):
    features = []
    accuracies = []
    confidences = []
    known_rates = []
    predictions = []
    known_masks = []
    expected = labels.to(device)
    with torch.no_grad():
        for index in indices:
            logits, semantic = classifier(
                frames[index].to(device), return_features=True
            )
            probabilities = logits.softmax(1)
            maximum, predicted = probabilities.max(1)
            known = maximum >= threshold
            strict = known & (predicted == expected)
            features.append(semantic.cpu().numpy())
            accuracies.append(float(strict.float().mean()))
            confidences.append(float(probabilities.gather(1, expected[:, None]).mean()))
            known_rates.append(float(known.float().mean()))
            predictions.append(predicted.cpu().numpy())
            known_masks.append(known.cpu().numpy())
    return (
        np.concatenate(features),
        accuracies,
        confidences,
        known_rates,
        predictions,
        known_masks,
    )


def run(config):
    cfg = {**DEFAULTS, **config}
    if not cfg["classifier_checkpoint"]:
        raise ValueError("classifier_checkpoint is required")
    if not cfg["experiments"]:
        raise ValueError("At least one experiment is required")
    seed_everything(cfg["seed"])
    device = get_device(cfg["device"])
    run_dir = create_run_dir(cfg["pipeline"], cfg["seed"], cfg["output_root"])
    classifier = load_classifier(_resolve(cfg["classifier_checkpoint"]), device)
    real_features, real_labels = _collect_real_features(classifier, cfg, device)
    samples = cfg["samples_per_class"]
    labels = torch.arange(cfg["num_classes"], device=device).repeat_interleave(samples)
    base_noise = torch.randn(
        samples, 1, cfg["image_size"], cfg["image_size"], device=device
    )
    shared_noise = base_noise.repeat(cfg["num_classes"], 1, 1, 1)
    all_frames = {}
    metadata = {}
    for name, experiment in cfg["experiments"].items():
        kind = experiment["kind"]
        checkpoint = _resolve(experiment["checkpoint"])
        if kind in {"conditional_additive", "conditional_adagn"}:
            model, info = load_pixel_conditional(checkpoint, device)
            frames = euler_integrate(model, shared_noise, cfg["ode_steps"], labels)
        elif kind == "unet":
            model, info = load_unet(checkpoint, device)
            frames = _guided_unet_frames(
                model,
                shared_noise,
                labels,
                cfg["ode_steps"],
                experiment.get("guidance_scale", cfg["guidance_scale"]),
            )
        elif kind == "latent_flow":
            flow, vae, info = load_latent_flow(
                checkpoint, _resolve(experiment["vae_checkpoint"]), device
            )
            frames = _latent_frames(
                flow,
                vae,
                labels,
                cfg["ode_steps"],
                cfg["image_size"],
                info["latent_channels"],
                device,
            )
        else:
            raise ValueError(f"Unknown experiment kind: {kind}")
        all_frames[name] = frames
        metadata[name] = info
        print(f"loaded {name}: {info}")
    indices = np.linspace(0, cfg["ode_steps"], cfg["time_points"]).round().astype(int)
    times = indices / cfg["ode_steps"]
    feature_sets = {"real_reference": real_features}
    accuracy_curves = {}
    confidence_curves = {}
    known_curves = {}
    final_predictions = {}
    final_known = {}
    for name, frames in all_frames.items():
        values = _semantic_sequence(
            classifier, frames, labels, indices, cfg["confidence_threshold"], device
        )
        features, accuracy, confidence, known_rate, predictions, known_masks = values
        feature_sets[name] = features
        accuracy_curves[name] = accuracy
        confidence_curves[name] = confidence
        known_curves[name] = known_rate
        final_predictions[name] = predictions[-1]
        final_known[name] = known_masks[-1]
    all_embeddings = joint_tsne(feature_sets, cfg["seed"])
    real_embedding = all_embeddings.pop("real_reference")
    labels_np = labels.cpu().numpy()
    plot_centroid_trajectories_with_reference(
        all_embeddings,
        real_embedding,
        real_labels,
        labels_np,
        final_predictions,
        final_known,
        len(indices),
        run_dir / "semantic_trajectories_with_real_clusters.png",
    )
    plot_final_predictions_with_reference(
        all_embeddings,
        real_embedding,
        real_labels,
        labels_np,
        final_predictions,
        final_known,
        len(indices),
        run_dir / "final_predictions_with_real_clusters.png",
    )
    plot_semantic_curves(
        accuracy_curves,
        times,
        run_dir / "strict_accuracy_over_time.png",
        "Strict target accuracy",
        "Target accuracy with low-confidence samples counted as unknown",
    )
    plot_semantic_curves(
        known_curves,
        times,
        run_dir / "known_rate_over_time.png",
        "Known rate",
        "Fraction above classifier confidence threshold",
    )
    plot_semantic_curves(
        confidence_curves,
        times,
        run_dir / "target_confidence_over_time.png",
        "Mean target confidence",
        "Target-label confidence along Flow time",
    )
    confusions = {}
    metrics = {}
    for name in all_frames:
        matrix = np.zeros((cfg["num_classes"], cfg["num_classes"] + 1), dtype=int)
        pred = final_predictions[name]
        known = final_known[name]
        for target, prediction, is_known in zip(labels_np, pred, known):
            matrix[target, prediction if is_known else cfg["num_classes"]] += 1
        confusions[name] = matrix
        strict = known & (pred == labels_np)
        known_correct = float(strict.sum() / max(known.sum(), 1))
        metrics[name] = {
            "strict_accuracy": float(strict.mean()),
            "known_rate": float(known.mean()),
            "accuracy_among_known": known_correct,
            "unknown_rate": float((~known).mean()),
            "target_confidence": confidence_curves[name][-1],
            "metadata": metadata[name],
        }
    plot_unknown_confusions(confusions, run_dir / "unknown_aware_confusions.png")
    np.savez_compressed(
        run_dir / "shared_tsne_embeddings.npz",
        real_embedding=real_embedding,
        real_labels=real_labels,
        expected_labels=labels_np,
        times=times,
        **all_embeddings,
    )
    save_run_metadata(run_dir, cfg, metrics)
    lines = [
        "# Semantic trajectories with real class references",
        "",
        f"Confidence threshold: {cfg['confidence_threshold']}",
        f"Generated samples per target Label: {samples}",
        f"Real reference samples per class: {cfg['real_samples_per_class']}",
        "",
        "## Final unknown-aware results",
        "",
    ]
    for name, result in metrics.items():
        lines.append(
            f"- **{name}**: strict accuracy={result['strict_accuracy']:.3f}, known rate={result['known_rate']:.3f}, accuracy among known={result['accuracy_among_known']:.3f}, unknown rate={result['unknown_rate']:.3f}"
        )
    (run_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"outputs: {run_dir}")
    return run_dir
