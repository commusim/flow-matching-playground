import numpy as np
import torch
from src.modules.flow import flow_matching_loss, linear_flow_batch
from src.modules.image_model_registry import create_mnist_model, load_mnist_checkpoint
from src.utils.common import (
    create_run_dir,
    get_device,
    project_root,
    save_run_metadata,
    seed_everything,
)
from src.utils.image_data import create_image_loader, dataset_spec, infinite_batches
from src.utils.image_plotting import (
    make_image_animation,
    plot_image_overview,
    plot_image_trajectory,
    plot_predicted_clean,
)
from src.utils.ode import euler_integrate
from src.utils.plotting import plot_loss

DEFAULTS = {
    "pipeline": "mnist_flow",
    "dataset": "mnist",
    "image_size": 28,
    "input_channels": None,
    "seed": 42,
    "device": "auto",
    "steps": 3000,
    "batch_size": 128,
    "lr": 0.001,
    "hidden": 64,
    "particles": 32,
    "ode_steps": 80,
    "subset_size": 60000,
    "animation": True,
    "model_variant": "unconditional",
    "checkpoint_path": None,
    "download": False,
    "data_root": None,
    "output_root": None,
}


def run(config):
    cfg = {**DEFAULTS, **config}
    seed_everything(cfg["seed"])
    device = get_device(cfg["device"])
    run_dir = create_run_dir(cfg["pipeline"], cfg["seed"], cfg["output_root"])
    spec = dataset_spec(cfg["dataset"])
    cfg["input_channels"] = cfg["input_channels"] or spec["channels"]
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
    if cfg["checkpoint_path"]:
        model, variant, dimensions = load_mnist_checkpoint(
            cfg["checkpoint_path"], device, cfg.get("model_variant", "auto")
        )
        if variant != "unconditional":
            raise ValueError(
                "A conditional checkpoint must use a conditional image pipeline"
            )
        cfg.update(dimensions)
        cfg["model_variant"] = variant
    else:
        model = create_mnist_model(
            "unconditional", hidden=cfg["hidden"], input_channels=cfg["input_channels"]
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    losses = []
    for step in range(1, cfg["steps"] + 1):
        images, _ = next(batches)
        data = images.to(device)
        noise = torch.randn_like(data)
        time = torch.rand(len(data), 1, device=device)
        state, target = linear_flow_batch(data, noise, time[:, :, None, None])
        loss = flow_matching_loss(model(state, time), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step == 1 or step % max(1, cfg["steps"] // 10) == 0:
            print(f"step {step}/{cfg['steps']} loss {np.mean(losses[-100:]):.5f}")
    real, _ = next(iter(loader))
    real = real[: cfg["particles"]].to(device)
    noise = torch.randn(
        cfg["particles"],
        cfg["input_channels"],
        cfg["image_size"],
        cfg["image_size"],
        device=device,
    )
    frames = euler_integrate(model, noise, cfg["ode_steps"])
    torch.save(model.state_dict(), run_dir / "checkpoint.pt")
    if losses:
        plot_loss(losses, run_dir / "loss_curve.png")
    plot_image_overview(real, noise, frames[-1], run_dir / "overview.png")
    plot_image_trajectory(frames, run_dir / "trajectory.png")
    plot_predicted_clean(model, frames, device, run_dir / "predicted_clean.png")
    if cfg["animation"]:
        make_image_animation(frames, run_dir / "sampling_animation.gif")
    save_run_metadata(
        run_dir,
        cfg,
        {
            "final_loss": losses[-1] if losses else None,
            "parameters": sum(p.numel() for p in model.parameters()),
            "device": str(device),
        },
    )
    print(f"outputs: {run_dir}")
    return run_dir
