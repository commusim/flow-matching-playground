import matplotlib.pyplot as plt
import numpy as np
import torch
from src.modules.flow import sinkhorn_coupling
from src.modules.training import train_conditional
from src.modules.velocity import ConditionalVelocityMLP
from src.utils.common import (
    create_run_dir,
    get_device,
    save_run_metadata,
    seed_everything,
)
from src.utils.data_2d import sample_moons, sample_noise, sample_ring
from src.utils.ode import euler_integrate
from src.utils.plotting import (
    COLORS,
    plot_loss,
    plot_trajectory,
    plot_velocity_times,
    scatter,
    style_axis,
)

DEFAULTS = {
    "pipeline": "ot_flow",
    "seed": 42,
    "device": "auto",
    "steps": 2500,
    "batch_size": 256,
    "lr": 0.002,
    "hidden": 128,
    "particles": 700,
    "ode_steps": 80,
    "ot_regularization": 0.12,
    "ot_iterations": 40,
    "output_root": None,
}


def run(config):
    cfg = {**DEFAULTS, **config}
    seed_everything(cfg["seed"])
    device = get_device(cfg["device"])
    run_dir = create_run_dir(cfg["pipeline"], cfg["seed"], cfg["output_root"])
    model = ConditionalVelocityMLP(cfg["hidden"]).to(device)

    def build_batch(size, device):
        labels = torch.randint(0, 2, (size,), device=device)
        noise = sample_noise(size, device)
        data = torch.empty_like(noise)
        distances = []
        for condition, sampler in ((0, sample_moons), (1, sample_ring)):
            mask = labels == condition
            if int(mask.sum()):
                source = noise[mask]
                candidate = sampler(len(source), device)
                paired, cost, _ = sinkhorn_coupling(
                    source, candidate, cfg["ot_regularization"], cfg["ot_iterations"]
                )
                data[mask] = paired
                distances.append(float(cost.mean().sqrt().cpu()))
        return data, noise, labels, float(np.mean(distances))

    losses, distances = train_conditional(model, build_batch, device, cfg)
    initial = sample_noise(cfg["particles"], device)
    labels = [
        torch.full((cfg["particles"],), value, dtype=torch.long, device=device)
        for value in (0, 1)
    ]
    frames = [
        euler_integrate(model, initial, cfg["ode_steps"], label) for label in labels
    ]
    targets = [
        sample_moons(cfg["particles"], device),
        sample_ring(cfg["particles"], device),
    ]
    torch.save(model.state_dict(), run_dir / "checkpoint.pt")
    plot_loss(losses, run_dir / "loss_curve.png")
    plot_velocity_times(
        model, device, run_dir / "velocity_fields.png", conditions=(0, 1)
    )
    plot_trajectory(
        frames[0],
        targets[0],
        run_dir / "trajectory_moons.png",
        color="#EF4444",
        title="OT condition 0: moons",
    )
    plot_trajectory(
        frames[1],
        targets[1],
        run_dir / "trajectory_ring.png",
        color="#2563EB",
        title="OT condition 1: ring",
    )
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for row, (name, color) in enumerate((("moons", "#EF4444"), ("ring", "#2563EB"))):
        scatter(axes[row, 0], initial, COLORS["noise"])
        style_axis(axes[row, 0], "OT source")
        scatter(axes[row, 1], targets[row], color)
        style_axis(axes[row, 1], f"{name} target")
        scatter(axes[row, 2], frames[row][-1], color)
        style_axis(axes[row, 2], f"{name} generated")
    fig.suptitle("Optimal-Transport Conditional Flow Matching", fontweight="bold")
    fig.savefig(run_dir / "overview.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(distances, alpha=0.7)
    ax.set(
        title="Mini-batch OT transport distance", xlabel="step", ylabel="mean distance"
    )
    ax.grid(alpha=0.25)
    fig.savefig(run_dir / "transport_distance.png", dpi=180)
    plt.close(fig)
    save_run_metadata(
        run_dir,
        cfg,
        {
            "final_loss": losses[-1],
            "mean_transport_distance": float(np.mean(distances)),
            "device": str(device),
        },
    )
    print(f"outputs: {run_dir}")
    return run_dir
