import numpy as np
import torch
from src.modules.flow import flow_matching_loss, linear_flow_batch
from src.modules.image_velocity import ConditionalImageUNet
from src.utils.common import (
    create_run_dir,
    get_device,
    project_root,
    save_run_metadata,
    seed_everything,
)
from src.utils.feature_visualization import (
    extract_feature_trajectory,
    plot_feature_pca_trajectory,
    plot_label_conditioned_samples,
    plot_spatial_feature_pca,
)
from src.utils.image_data import create_image_loader, dataset_spec, infinite_batches
from src.utils.image_plotting import (
    make_image_animation,
    plot_image_overview,
    plot_image_trajectory,
    plot_predicted_clean,
)
from src.utils.plotting import plot_loss
from src.utils.semantic_evaluation import evaluate_conditional_images, load_classifier

DEFAULTS = {
    "pipeline": "mnist_unet_flow",
    "dataset": "mnist",
    "image_size": 28,
    "input_channels": None,
    "num_classes": None,
    "seed": 42,
    "device": "auto",
    "steps": 5000,
    "batch_size": 128,
    "lr": 0.001,
    "base_channels": 32,
    "samples_per_class": 4,
    "ode_steps": 80,
    "subset_size": 60000,
    "label_dropout": 0.1,
    "guidance_scale": 3.0,
    "animation": True,
    "classifier_checkpoint": None,
    "download": False,
    "data_root": None,
    "output_root": None,
}


def guided_integrate(model, initial, labels, steps, guidance):
    x = initial.clone()
    frames = [x.cpu()]
    dt = 1 / steps
    model.eval()
    with torch.no_grad():
        for index in range(steps):
            time = torch.full((len(x), 1), index / steps, device=x.device)
            conditional = model(x, time, labels)
            null = torch.full_like(labels, model.null_label)
            unconditional = model(x, time, null)
            x = x + dt * (unconditional + guidance * (conditional - unconditional))
            frames.append(x.cpu())
    return frames


def run(config):
    cfg = {**DEFAULTS, **config}
    seed_everything(cfg["seed"])
    device = get_device(cfg["device"])
    run_dir = create_run_dir(cfg["pipeline"], cfg["seed"], cfg["output_root"])
    spec = dataset_spec(cfg["dataset"])
    cfg["input_channels"] = cfg["input_channels"] or spec["channels"]
    cfg["num_classes"] = cfg["num_classes"] or spec["classes"]
    if cfg["image_size"] % 4 != 0:
        raise ValueError("image_size must be divisible by 4 for the two-level U-Net")
    loader = create_image_loader(
        cfg["data_root"] or project_root() / "data",
        cfg["batch_size"],
        dataset=cfg["dataset"],
        train=True,
        image_size=cfg["image_size"],
        input_channels=cfg["input_channels"],
        subset_size=cfg["subset_size"],
        download=cfg["download"],
    )
    batches = infinite_batches(loader)
    model = ConditionalImageUNet(
        cfg["input_channels"], cfg["base_channels"], cfg["num_classes"] + 1
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    losses = []
    for step in range(1, cfg["steps"] + 1):
        images, labels = next(batches)
        data, labels = images.to(device), labels.to(device)
        drop = torch.rand(len(labels), device=device) < cfg["label_dropout"]
        train_labels = labels.clone()
        train_labels[drop] = model.null_label
        noise = torch.randn_like(data)
        time = torch.rand(len(data), 1, device=device)
        state, target = linear_flow_batch(data, noise, time[:, :, None, None])
        loss = flow_matching_loss(model(state, time, train_labels), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step == 1 or step % max(1, cfg["steps"] // 10) == 0:
            print(f"step {step}/{cfg['steps']} loss {np.mean(losses[-100:]):.5f}")
    samples = cfg["samples_per_class"]
    labels = torch.arange(cfg["num_classes"], device=device).repeat_interleave(samples)
    noise = torch.randn(
        samples,
        cfg["input_channels"],
        cfg["image_size"],
        cfg["image_size"],
        device=device,
    ).repeat(cfg["num_classes"], 1, 1, 1)
    frames = guided_integrate(
        model, noise, labels, cfg["ode_steps"], cfg["guidance_scale"]
    )
    real, _ = next(iter(loader))
    real = real[: len(labels)].to(device)
    torch.save(model.state_dict(), run_dir / "checkpoint.pt")
    plot_loss(losses, run_dir / "loss_curve.png")
    plot_image_overview(real, noise, frames[-1], run_dir / "overview.png")
    plot_label_conditioned_samples(
        frames[-1], labels, run_dir / "label_grid.png", samples
    )
    plot_image_trajectory(
        frames, run_dir / "trajectory.png", samples=min(cfg["num_classes"], len(labels))
    )
    plot_predicted_clean(
        model,
        frames,
        device,
        run_dir / "predicted_clean.png",
        samples=min(cfg["num_classes"], len(labels)),
        labels=labels,
    )
    vectors, maps = extract_feature_trajectory(
        model, frames, labels, device, layer="bottleneck", levels=(1, 2)
    )
    plot_feature_pca_trajectory(vectors, labels, run_dir / "bottleneck_feature_pca.png")
    plot_spatial_feature_pca(maps, labels, run_dir / "bottleneck_spatial_pca.png")
    if cfg["animation"]:
        make_image_animation(
            frames, run_dir / "sampling_animation.gif", min(16, len(labels))
        )
    metrics = {
        "final_loss": losses[-1],
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": str(device),
    }
    if cfg["classifier_checkpoint"]:
        classifier = load_classifier(cfg["classifier_checkpoint"], device)
        semantic, _, _ = evaluate_conditional_images(
            classifier, frames[-1].to(device), labels
        )
        metrics.update(semantic)
    save_run_metadata(run_dir, cfg, metrics)
    print(f"outputs: {run_dir}")
    return run_dir
