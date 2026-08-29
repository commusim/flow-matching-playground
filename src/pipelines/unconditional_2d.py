import matplotlib.pyplot as plt
import torch
from src.modules.training import train_unconditional
from src.modules.velocity import VelocityMLP
from src.utils.common import (
    create_run_dir,
    get_device,
    save_run_metadata,
    seed_everything,
)
from src.utils.data_2d import sample_moons, sample_noise
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
    "pipeline": "unconditional_2d",
    "seed": 42,
    "device": "auto",
    "steps": 4000,
    "batch_size": 1024,
    "lr": 0.002,
    "hidden": 128,
    "particles": 1400,
    "ode_steps": 80,
    "output_root": None,
}


def run(config):
    cfg = {**DEFAULTS, **config}
    seed_everything(cfg["seed"])
    device = get_device(cfg["device"])
    run_dir = create_run_dir(cfg["pipeline"], cfg["seed"], cfg["output_root"])
    model = VelocityMLP(cfg["hidden"]).to(device)
    losses = train_unconditional(model, sample_moons, sample_noise, device, cfg)
    initial = sample_noise(cfg["particles"], device)
    target = sample_moons(cfg["particles"], device)
    frames = euler_integrate(model, initial, cfg["ode_steps"])
    torch.save(model.state_dict(), run_dir / "checkpoint.pt")
    plot_loss(losses, run_dir / "loss_curve.png")
    plot_trajectory(frames, target, run_dir / "trajectory.png")
    plot_velocity_times(model, device, run_dir / "velocity_fields.png")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    scatter(axes[0], initial, COLORS["noise"])
    style_axis(axes[0], "Gaussian noise")
    scatter(axes[1], target, COLORS["data"])
    style_axis(axes[1], "Target moons")
    scatter(axes[2], target, COLORS["data"], alpha=0.15)
    scatter(axes[2], frames[-1], COLORS["generated"])
    style_axis(axes[2], "Generated")
    fig.suptitle("Unconditional 2D Flow Matching", fontweight="bold")
    fig.savefig(run_dir / "overview.png", dpi=180)
    plt.close(fig)
    metrics = {
        "final_loss": losses[-1],
        "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    save_run_metadata(run_dir, cfg, metrics)
    print(f"outputs: {run_dir}")
    return run_dir
