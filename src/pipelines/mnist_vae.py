import numpy as np
import torch
from src.modules.vae import ImageVAE
from src.utils.common import (
    create_run_dir,
    get_device,
    project_root,
    save_run_metadata,
    seed_everything,
)
from src.utils.image_data import create_image_loader, dataset_spec
from src.utils.plotting import plot_loss
from src.utils.vae_plotting import plot_latent_channels, plot_vae_reconstruction

DEFAULTS = {
    "pipeline": "mnist_vae",
    "dataset": "mnist",
    "image_size": 28,
    "input_channels": None,
    "seed": 42,
    "device": "auto",
    "epochs": 10,
    "batch_size": 128,
    "lr": 0.001,
    "latent_channels": 8,
    "kl_weight": 0.0001,
    "subset_size": 60000,
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
    if cfg["image_size"] % 4 != 0:
        raise ValueError("image_size must be divisible by 4 for ImageVAE")
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
    model = ImageVAE(cfg["latent_channels"], cfg["input_channels"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    losses = []
    for epoch in range(cfg["epochs"]):
        model.train()
        for images, _ in loader:
            images = images.to(device)
            recon, mean, logvar = model(images)
            recon_loss = torch.nn.functional.mse_loss(recon, images)
            kl = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
            loss = recon_loss + cfg["kl_weight"] * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(
            f"epoch {epoch + 1}/{cfg['epochs']} loss {np.mean(losses[-len(loader) :]):.5f}"
        )
    model.eval()
    images, _ = next(iter(loader))
    images = images.to(device)
    with torch.no_grad():
        recon, mean, _ = model(images)
    torch.save(model.state_dict(), run_dir / "checkpoint.pt")
    plot_loss(losses, run_dir / "loss_curve.png")
    plot_vae_reconstruction(images, recon, run_dir / "reconstruction.png")
    plot_latent_channels(mean, run_dir / "latent_channels.png")
    save_run_metadata(
        run_dir,
        cfg,
        {
            "final_loss": losses[-1],
            "reconstruction_mse": float(torch.nn.functional.mse_loss(recon, images)),
            "parameters": sum(p.numel() for p in model.parameters()),
            "device": str(device),
            "latent_size": [
                cfg["latent_channels"],
                cfg["image_size"] // 4,
                cfg["image_size"] // 4,
            ],
        },
    )
    print(f"outputs: {run_dir}")
    return run_dir
