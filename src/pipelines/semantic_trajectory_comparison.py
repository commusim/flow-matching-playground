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
from src.utils.ode import euler_integrate
from src.utils.semantic_evaluation import load_classifier
from src.utils.semantic_trajectory_plotting import (
    joint_tsne,
    plot_endpoint_embedding,
    plot_label_centroid_trajectories,
    plot_joint_trajectories,
    plot_semantic_curves,
)

DEFAULTS = {
    "pipeline": "semantic_trajectory_comparison",
    "seed": 42,
    "device": "auto",
    "image_size": 28,
    "samples_per_class": 2,
    "num_classes": 10,
    "ode_steps": 80,
    "time_points": 13,
    "guidance_scale": 3.0,
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
            null_labels = torch.full_like(labels, model.null_label)
            unconditional = model(image, time, null_labels)
            image = image + dt * (unconditional + scale * (conditional - unconditional))
            frames.append(image.cpu())
    return frames


def _latent_frames(flow, vae, labels, steps, image_size, latent_channels, device):
    latent_size = image_size // 4
    latent = torch.randn(
        len(labels), latent_channels, latent_size, latent_size, device=device
    )
    frames = []
    dt = 1 / steps
    with torch.no_grad():
        frames.append(vae.decode(latent).cpu())
        for index in range(steps):
            time = torch.full((len(latent), 1), index / steps, device=device)
            latent = latent + dt * flow(latent, time, labels)
            frames.append(vae.decode(latent).cpu())
    return frames


def _semantic_features(classifier, frames, labels, indices, device):
    features = []
    accuracies = []
    confidences = []
    expected = labels.to(device)
    with torch.no_grad():
        for index in indices:
            logits, semantic = classifier(
                frames[index].to(device), return_features=True
            )
            probabilities = logits.softmax(dim=1)
            predictions = probabilities.argmax(dim=1)
            features.append(semantic.cpu().numpy())
            accuracies.append(float((predictions == expected).float().mean()))
            confidences.append(float(probabilities.gather(1, expected[:, None]).mean()))
    return np.concatenate(features, axis=0), accuracies, confidences


def run(config):
    cfg = {**DEFAULTS, **config}
    if not cfg["classifier_checkpoint"]:
        raise ValueError("classifier_checkpoint is required")
    if not cfg["experiments"]:
        raise ValueError("At least one experiment checkpoint is required")
    seed_everything(cfg["seed"])
    device = get_device(cfg["device"])
    run_dir = create_run_dir(cfg["pipeline"], cfg["seed"], cfg["output_root"])
    classifier = load_classifier(_resolve(cfg["classifier_checkpoint"]), device)
    samples = cfg["samples_per_class"]
    labels = torch.arange(cfg["num_classes"], device=device).repeat_interleave(samples)
    shared_noise = torch.randn(
        samples, 1, cfg["image_size"], cfg["image_size"], device=device
    ).repeat(cfg["num_classes"], 1, 1, 1)
    all_frames = {}
    experiment_metadata = {}
    for name, experiment in cfg["experiments"].items():
        kind = experiment["kind"]
        checkpoint = _resolve(experiment["checkpoint"])
        if kind in {"conditional_additive", "conditional_adagn"}:
            model, metadata = load_pixel_conditional(checkpoint, device)
            frames = euler_integrate(model, shared_noise, cfg["ode_steps"], labels)
        elif kind == "unet":
            model, metadata = load_unet(checkpoint, device)
            scale = experiment.get("guidance_scale", cfg["guidance_scale"])
            frames = _guided_unet_frames(
                model, shared_noise, labels, cfg["ode_steps"], scale
            )
        elif kind == "latent_flow":
            flow, vae, metadata = load_latent_flow(
                checkpoint, _resolve(experiment["vae_checkpoint"]), device
            )
            frames = _latent_frames(
                flow,
                vae,
                labels,
                cfg["ode_steps"],
                cfg["image_size"],
                metadata["latent_channels"],
                device,
            )
        else:
            raise ValueError(f"Unknown experiment kind: {kind}")
        all_frames[name] = frames
        experiment_metadata[name] = metadata
        print(f"loaded {name}: {metadata}")
    indices = np.linspace(0, cfg["ode_steps"], cfg["time_points"]).round().astype(int)
    times = indices / cfg["ode_steps"]
    feature_sets = {}
    accuracy_curves = {}
    confidence_curves = {}
    for name, frames in all_frames.items():
        features, accuracy, confidence = _semantic_features(
            classifier, frames, labels, indices, device
        )
        feature_sets[name] = features
        accuracy_curves[name] = accuracy
        confidence_curves[name] = confidence
    embeddings = joint_tsne(feature_sets, cfg["seed"])
    labels_numpy = labels.cpu().numpy()
    plot_joint_trajectories(
        embeddings,
        labels_numpy,
        len(indices),
        run_dir / "shared_tsne_semantic_trajectories.png",
    )
    plot_label_centroid_trajectories(
        embeddings,
        labels_numpy,
        len(indices),
        run_dir / "shared_tsne_label_centroid_trajectories.png",
    )
    np.savez_compressed(
        run_dir / "shared_tsne_embeddings.npz",
        labels=labels_numpy,
        times=times,
        **embeddings,
    )
    plot_endpoint_embedding(
        embeddings,
        labels_numpy,
        len(indices),
        run_dir / "shared_tsne_final_embeddings.png",
    )
    plot_semantic_curves(
        accuracy_curves,
        times,
        run_dir / "semantic_accuracy_over_time.png",
        "Target-label accuracy",
        "Classifier semantic accuracy along the generation trajectory",
    )
    plot_semantic_curves(
        confidence_curves,
        times,
        run_dir / "semantic_confidence_over_time.png",
        "Mean target-label confidence",
        "Classifier target confidence along the generation trajectory",
    )
    metrics = {
        name: {
            "final_accuracy": accuracy_curves[name][-1],
            "final_confidence": confidence_curves[name][-1],
            "metadata": experiment_metadata[name],
        }
        for name in all_frames
    }
    save_run_metadata(run_dir, cfg, metrics)
    report_lines = [
        "# Classifier-semantic trajectory comparison",
        "",
        "All intermediate images are encoded by the same frozen supervised classifier.",
        "All experiments and time points are concatenated before fitting one shared t-SNE.",
        "",
        "## Final results",
        "",
    ]
    for name, result in metrics.items():
        report_lines.append(
            f"- **{name}**: accuracy={result['final_accuracy']:.3f}, "
            f"target confidence={result['final_confidence']:.3f}"
        )
    report_lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `shared_tsne_semantic_trajectories.png`: same t-SNE coordinate system for every experiment.",
            "- `shared_tsne_label_centroid_trajectories.png`: cleaner per-Label centroid trajectories.",
            "- `shared_tsne_final_embeddings.png`: final semantic feature locations.",
            "- `semantic_accuracy_over_time.png`: target-label accuracy versus Flow time.",
            "- `semantic_confidence_over_time.png`: target confidence versus Flow time.",
            "",
            "t-SNE is jointly fitted for comparability, but its global distances and path lengths should not be interpreted quantitatively.",
        ]
    )
    (run_dir / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"outputs: {run_dir}")
    return run_dir
