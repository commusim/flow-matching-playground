import matplotlib.pyplot as plt
import torch
from src.modules.training import train_conditional
from src.modules.velocity import ConditionalVelocityMLP
from src.utils.common import (
    create_run_dir,
    get_device,
    save_run_metadata,
    seed_everything,
)
from src.utils.data_2d import (
    sample_by_condition,
    sample_moons,
    sample_noise,
    sample_ring,
)
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
    "pipeline": "conditional_2d",
    "seed": 42,
    "device": "auto",
    "steps": 4000,
    "batch_size": 768,
    "lr": 0.002,
    "hidden": 128,
    "particles": 1000,
    "ode_steps": 80,
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
        return (
            sample_by_condition(labels, device),
            sample_noise(size, device),
            labels,
            None,
        )

    losses, _ = train_conditional(model, build_batch, device, cfg)
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
        title="Condition 0: moons",
    )
    plot_trajectory(
        frames[1],
        targets[1],
        run_dir / "trajectory_ring.png",
        color="#2563EB",
        title="Condition 1: ring",
    )
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for row, (name, color) in enumerate((("moons", "#EF4444"), ("ring", "#2563EB"))):
        scatter(axes[row, 0], initial, COLORS["noise"])
        style_axis(axes[row, 0], f"same noise → {name}")
        scatter(axes[row, 1], targets[row], color)
        style_axis(axes[row, 1], f"{name} target")
        scatter(axes[row, 2], frames[row][-1], color)
        style_axis(axes[row, 2], f"{name} generated")
    fig.suptitle("Conditional Flow Matching", fontweight="bold")
    fig.savefig(run_dir / "overview.png", dpi=180)
    plt.close(fig)
    save_run_metadata(run_dir, cfg, {"final_loss": losses[-1], "device": str(device)})
    print(f"outputs: {run_dir}")
    return run_dir
