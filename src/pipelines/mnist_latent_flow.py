import numpy as np
import torch
from src.modules.flow import flow_matching_loss, linear_flow_batch
from src.modules.vae import LatentConditionalVelocityCNN, MNISTVAE
from src.utils.common import (
    create_run_dir,
    get_device,
    project_root,
    save_run_metadata,
    seed_everything,
)
from src.utils.image_data import create_mnist_loader, infinite_batches
from src.utils.image_plotting import plot_image_overview, plot_image_trajectory
from src.utils.feature_visualization import plot_label_conditioned_samples
from src.utils.plotting import plot_loss
from src.utils.semantic_evaluation import evaluate_conditional_images, load_classifier

DEFAULTS = {
    "pipeline": "mnist_latent_flow",
    "seed": 42,
    "device": "auto",
    "steps": 4000,
    "batch_size": 128,
    "lr": 0.001,
    "hidden": 64,
    "latent_channels": 8,
    "samples_per_class": 4,
    "ode_steps": 80,
    "subset_size": 60000,
    "vae_checkpoint": None,
    "classifier_checkpoint": None,
    "output_root": None,
}


def run(config):
    cfg = {**DEFAULTS, **config}
    seed_everything(cfg["seed"])
    device = get_device(cfg["device"])
    run_dir = create_run_dir(cfg["pipeline"], cfg["seed"], cfg["output_root"])
    if not cfg["vae_checkpoint"]:
        raise ValueError("mnist_latent_flow requires vae_checkpoint")
    vae = MNISTVAE(cfg["latent_channels"]).to(device)
    vae.load_state_dict(torch.load(cfg["vae_checkpoint"], map_location=device))
    vae.eval()
    [p.requires_grad_(False) for p in vae.parameters()]
    loader = create_mnist_loader(
        project_root() / "data", cfg["batch_size"], True, cfg["subset_size"], False, 0
    )
    batches = infinite_batches(loader)
    model = LatentConditionalVelocityCNN(cfg["latent_channels"], cfg["hidden"]).to(
        device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    losses = []
    for step in range(1, cfg["steps"] + 1):
        images, labels = next(batches)
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            data = vae.encode(images, sample=False)
        noise = torch.randn_like(data)
        time = torch.rand(len(data), 1, device=device)
        state, target = linear_flow_batch(data, noise, time[:, :, None, None])
        loss = flow_matching_loss(model(state, time, labels), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step == 1 or step % max(1, cfg["steps"] // 10) == 0:
            print(f"step {step}/{cfg['steps']} loss {np.mean(losses[-100:]):.5f}")
    samples = cfg["samples_per_class"]
    labels = torch.arange(10, device=device).repeat_interleave(samples)
    latent = torch.randn(len(labels), cfg["latent_channels"], 7, 7, device=device)
    latent_frames = [latent.cpu()]
    dt = 1 / cfg["ode_steps"]
    model.eval()
    with torch.no_grad():
        for index in range(cfg["ode_steps"]):
            time = torch.full((len(latent), 1), index / cfg["ode_steps"], device=device)
            latent = latent + dt * model(latent, time, labels)
            latent_frames.append(latent.cpu())
        decoded_frames = [vae.decode(frame.to(device)).cpu() for frame in latent_frames]
    real, _ = next(iter(loader))
    real = real[: len(labels)].to(device)
    generated = decoded_frames[-1].to(device)
    torch.save(model.state_dict(), run_dir / "checkpoint.pt")
    plot_loss(losses, run_dir / "loss_curve.png")
    plot_image_overview(real, decoded_frames[0], generated, run_dir / "overview.png")
    plot_label_conditioned_samples(
        generated, labels, run_dir / "label_grid.png", samples
    )
    plot_image_trajectory(
        decoded_frames, run_dir / "decoded_trajectory.png", samples=min(10, len(labels))
    )
    metrics = {
        "final_loss": losses[-1],
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": str(device),
    }
    if cfg["classifier_checkpoint"]:
        classifier = load_classifier(cfg["classifier_checkpoint"], device)
        semantic, _, _ = evaluate_conditional_images(classifier, generated, labels)
        metrics.update(semantic)
    save_run_metadata(run_dir, cfg, metrics)
    print(f"outputs: {run_dir}")
    return run_dir
