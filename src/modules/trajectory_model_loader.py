from pathlib import Path
import torch
from src.modules.image_model_registry import load_mnist_checkpoint
from src.modules.image_velocity import ConditionalImageUNet
from src.modules.vae import ImageVAE, LatentConditionalVelocityCNN


def load_pixel_conditional(path, device):
    model, variant, dimensions = load_mnist_checkpoint(path, device, "auto")
    if variant == "unconditional":
        raise ValueError("Semantic label trajectories require a conditional checkpoint")
    model.eval()
    return model, {"variant": variant, **dimensions}


def load_unet(path, device):
    state = torch.load(Path(path), map_location=device)
    input_channels = int(state["input.weight"].shape[1])
    base = int(state["input.weight"].shape[0])
    classes_with_null = int(state["label_embedding.weight"].shape[0])
    condition_dim = int(state["condition.0.weight"].shape[0])
    model = ConditionalImageUNet(
        input_channels=input_channels,
        base=base,
        classes=classes_with_null,
        condition_dim=condition_dim,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, {
        "variant": "conditional_unet",
        "input_channels": input_channels,
        "base_channels": base,
        "classes": classes_with_null - 1,
    }


def load_latent_flow(flow_path, vae_path, device):
    flow_state = torch.load(Path(flow_path), map_location=device)
    vae_state = torch.load(Path(vae_path), map_location=device)
    latent_channels = int(flow_state["input.weight"].shape[1])
    hidden = int(flow_state["input.weight"].shape[0])
    classes = int(flow_state["label_embedding.weight"].shape[0])
    input_channels = int(vae_state["encoder.0.weight"].shape[1])
    flow = LatentConditionalVelocityCNN(
        latent_channels=latent_channels,
        hidden=hidden,
        classes=classes,
    ).to(device)
    flow.load_state_dict(flow_state, strict=True)
    flow.eval()
    vae = ImageVAE(latent_channels, input_channels).to(device)
    vae.load_state_dict(vae_state, strict=True)
    vae.eval()
    return (
        flow,
        vae,
        {
            "variant": "latent_flow",
            "latent_channels": latent_channels,
            "hidden": hidden,
            "classes": classes,
            "input_channels": input_channels,
        },
    )
