from pathlib import Path
import torch
from src.modules.image_velocity import (
    AdaGNConditionalMNISTVelocityCNN,
    AdditiveConditionalMNISTVelocityCNN,
    UnconditionalMNISTVelocityCNN,
)

MODEL_VARIANTS = {
    "unconditional": UnconditionalMNISTVelocityCNN,
    "conditional_additive": AdditiveConditionalMNISTVelocityCNN,
    "conditional_adagn": AdaGNConditionalMNISTVelocityCNN,
}


def _unwrap_state(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def infer_model_variant(state):
    keys = set(state)
    if "label_embedding.weight" not in keys:
        return "unconditional"
    if "condition_projection.0.weight" in keys:
        return "conditional_adagn"
    if "label_projection.0.weight" in keys:
        return "conditional_additive"
    raise ValueError("Unable to infer MNIST model variant from checkpoint keys")


def infer_model_dimensions(state):
    dimensions = {"hidden": int(state["input.weight"].shape[0])}
    if "label_embedding.weight" in state:
        dimensions["classes"] = int(state["label_embedding.weight"].shape[0])
        dimensions["label_dim"] = int(state["label_embedding.weight"].shape[1])
    if "time_embedding.frequencies" in state:
        dimensions["time_dim"] = int(state["time_embedding.frequencies"].numel() * 2)
    return dimensions


def create_mnist_model(variant, hidden=64, time_dim=32, label_dim=32, classes=10):
    if variant not in MODEL_VARIANTS:
        raise ValueError(
            f"Unknown model variant: {variant}. Available: {', '.join(MODEL_VARIANTS)}"
        )
    model_class = MODEL_VARIANTS[variant]
    if variant == "unconditional":
        return model_class(hidden=hidden, time_dim=time_dim)
    return model_class(
        hidden=hidden,
        time_dim=time_dim,
        label_dim=label_dim,
        classes=classes,
    )


def load_mnist_checkpoint(path, device, variant="auto"):
    checkpoint = torch.load(Path(path), map_location=device)
    state = _unwrap_state(checkpoint)
    detected = infer_model_variant(state)
    if variant != "auto" and variant != detected:
        raise ValueError(
            f"Checkpoint is '{detected}', but config requested '{variant}'. "
            "Use model_variant: auto or select the matching architecture."
        )
    dimensions = infer_model_dimensions(state)
    model = create_mnist_model(detected, **dimensions).to(device)
    model.load_state_dict(state, strict=True)
    return model, detected, dimensions
