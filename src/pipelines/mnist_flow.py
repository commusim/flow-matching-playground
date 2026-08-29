import numpy as np
import torch
from src.modules.flow import flow_matching_loss, linear_flow_batch
from src.modules.image_velocity import MNISTVelocityCNN
from src.utils.common import (
    create_run_dir,
    get_device,
    project_root,
    save_run_metadata,
    seed_everything,
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
    "pipeline": "mnist_flow",
    "seed": 42,
    "device": "auto",
    "steps": 3000,
    "batch_size": 128,
    "lr": 0.001,
    "hidden": 64,
    "particles": 32,
    "ode_steps": 80,
    "subset_size": 60000,
    "download": False,
    "num_workers": 0,
    "animation": True,
    "output_root": None,
}


def run(config):
    cfg = {**DEFAULTS, **config}
    seed_everything(cfg["seed"])
    device = get_device(cfg["device"])
    run_dir = create_run_dir(cfg["pipeline"], cfg["seed"], cfg["output_root"])
    data_root = project_root() / "data"
    loader = create_mnist_loader(
        data_root=data_root,
        batch_size=cfg["batch_size"],
        train=True,
        subset_size=cfg["subset_size"],
        download=cfg["download"],
        num_workers=cfg["num_workers"],
    )
    batches = infinite_batches(loader)
    model = MNISTVelocityCNN(hidden=cfg["hidden"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    losses = []
    model.train()
    for step in range(1, cfg["steps"] + 1):
        images, _ = next(batches)
        data = images.to(device)
        noise = torch.randn_like(data)
        time = torch.rand(len(data), 1, device=device)
        state, target_velocity = linear_flow_batch(data, noise, time.view(-1, 1, 1, 1))
        predicted_velocity = model(state, time)
        loss = flow_matching_loss(predicted_velocity, target_velocity)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step == 1 or step % max(1, cfg["steps"] // 10) == 0:
            print(f"step {step}/{cfg['steps']} loss {np.mean(losses[-100:]):.5f}")

    model.eval()
    real_images, _ = next(iter(loader))
    real_images = real_images[: cfg["particles"]].to(device)
    initial_noise = torch.randn(cfg["particles"], 1, 28, 28, device=device)
    frames = euler_integrate(model, initial_noise, cfg["ode_steps"])
    torch.save(model.state_dict(), run_dir / "checkpoint.pt")
    plot_loss(losses, run_dir / "loss_curve.png")
    plot_image_overview(
        real_images, initial_noise, frames[-1], run_dir / "overview.png"
    )
    plot_image_trajectory(frames, run_dir / "trajectory.png")
    plot_predicted_clean(model, frames, device, run_dir / "predicted_clean.png")
    if cfg["animation"]:
        make_image_animation(frames, run_dir / "sampling_animation.gif")
    generated = frames[-1]
    metrics = {
        "final_loss": losses[-1],
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "generated_mean": float(generated.mean()),
        "generated_std": float(generated.std()),
        "training_samples": min(cfg["subset_size"], 60000),
    }
    save_run_metadata(run_dir, cfg, metrics)
    print(f"outputs: {run_dir}")
    return run_dir
