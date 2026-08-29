import numpy as np
import torch
from src.modules.flow import flow_matching_loss, linear_flow_batch
from src.modules.image_velocity import ConditionalMNISTVelocityCNN
from src.utils.common import (
    create_run_dir,
    get_device,
    project_root,
    save_run_metadata,
    seed_everything,
)
from src.utils.feature_visualization import (
    extract_feature_trajectory,
    plot_feature_activation,
    plot_feature_pca_trajectory,
    plot_label_conditioned_samples,
    plot_spatial_feature_pca,
)
from src.utils.image_data import create_mnist_loader, infinite_batches
from src.utils.image_plotting import (
    make_image_animation,
    plot_image_overview,
    plot_image_trajectory,
    plot_predicted_clean,
)
from src.utils.ode import euler_integrate
from src.utils.plotting import plot_loss

DEFAULTS = {
    "pipeline": "mnist_conditional_flow",
    "seed": 42,
    "device": "auto",
    "steps": 3000,
    "batch_size": 128,
    "lr": 0.001,
    "hidden": 64,
    "samples_per_class": 4,
    "ode_steps": 80,
    "subset_size": 60000,
    "download": False,
    "num_workers": 0,
    "animation": True,
    "feature_layer": "block_3",
    "checkpoint_path": None,
    "output_root": None,
}


def run(config):
    cfg = {**DEFAULTS, **config}
    seed_everything(cfg["seed"])
    device = get_device(cfg["device"])
    run_dir = create_run_dir(cfg["pipeline"], cfg["seed"], cfg["output_root"])
    loader = create_mnist_loader(
        project_root() / "data",
        cfg["batch_size"],
        train=True,
        subset_size=cfg["subset_size"],
        download=cfg["download"],
        num_workers=cfg["num_workers"],
    )
    batches = infinite_batches(loader)
    model = ConditionalMNISTVelocityCNN(hidden=cfg["hidden"]).to(device)
    if cfg["checkpoint_path"]:
        state = torch.load(cfg["checkpoint_path"], map_location=device)
        model.load_state_dict(state)
        print(f"loaded checkpoint: {cfg['checkpoint_path']}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    losses = []
    model.train()
    for step in range(1, cfg["steps"] + 1):
        images, labels = next(batches)
        data = images.to(device)
        labels = labels.to(device)
        noise = torch.randn_like(data)
        time = torch.rand(len(data), 1, device=device)
        state, target = linear_flow_batch(data, noise, time.view(-1, 1, 1, 1))
        loss = flow_matching_loss(model(state, time, labels), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step == 1 or step % max(1, cfg["steps"] // 10) == 0:
            print(f"step {step}/{cfg['steps']} loss {np.mean(losses[-100:]):.5f}")

    model.eval()
    samples_per_class = cfg["samples_per_class"]
    labels = torch.arange(10, device=device).repeat_interleave(samples_per_class)
    shared_noise = torch.randn(samples_per_class, 1, 28, 28, device=device)
    initial_noise = shared_noise.repeat(10, 1, 1, 1)
    frames = euler_integrate(model, initial_noise, cfg["ode_steps"], labels)
    real_images, _ = next(iter(loader))
    real_images = real_images[: len(labels)].to(device)
    torch.save(model.state_dict(), run_dir / "checkpoint.pt")
    if losses:
        plot_loss(losses, run_dir / "loss_curve.png")
    plot_image_overview(
        real_images, initial_noise, frames[-1], run_dir / "overview.png"
    )
    plot_label_conditioned_samples(
        frames[-1], labels, run_dir / "label_grid.png", samples_per_class
    )
    plot_image_trajectory(
        frames, run_dir / "trajectory.png", samples=min(10, len(labels))
    )
    plot_predicted_clean(
        model,
        frames,
        device,
        run_dir / "predicted_clean.png",
        samples=min(10, len(labels)),
        labels=labels,
    )
    feature_vectors, feature_maps = extract_feature_trajectory(
        model, frames, labels, device, layer=cfg["feature_layer"]
    )
    plot_feature_pca_trajectory(
        feature_vectors, labels, run_dir / "feature_pca_trajectory.png"
    )
    plot_spatial_feature_pca(feature_maps, labels, run_dir / "spatial_feature_pca.png")
    plot_feature_activation(feature_maps, labels, run_dir / "feature_activation.png")
    if cfg["animation"]:
        make_image_animation(
            frames, run_dir / "sampling_animation.gif", samples=min(16, len(labels))
        )
    probe = torch.randn(16, 1, 28, 28, device=device)
    probe_time = torch.full((len(probe), 1), 0.05, device=device)
    label_velocities = []
    with torch.no_grad():
        for label_value in range(10):
            probe_labels = torch.full(
                (len(probe),), label_value, dtype=torch.long, device=device
            )
            label_velocities.append(model(probe, probe_time, probe_labels))
    pairwise_differences = []
    for first in range(10):
        for second in range(first + 1, 10):
            difference = (label_velocities[first] - label_velocities[second]).pow(2)
            pairwise_differences.append(float(difference.mean().sqrt().cpu()))
    label_velocity_separation = float(np.mean(pairwise_differences))
    generated = frames[-1]
    save_run_metadata(
        run_dir,
        cfg,
        {
            "final_loss": losses[-1] if losses else None,
            "device": str(device),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "generated_mean": float(generated.mean()),
            "generated_std": float(generated.std()),
            "label_velocity_separation_t005": label_velocity_separation,
        },
    )
    print(f"outputs: {run_dir}")
    return run_dir
