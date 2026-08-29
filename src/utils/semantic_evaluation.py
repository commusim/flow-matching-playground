import torch
from src.modules.classifier import ImageClassifier


def load_classifier(path, device):
    state = torch.load(path, map_location=device)
    input_channels = int(state["encoder.0.weight"].shape[1])
    feature_dim = int(state["classifier.weight"].shape[1])
    classes = int(state["classifier.weight"].shape[0])
    pool_flat = int(state["feature_projection.1.weight"].shape[1])
    pool_size = int(round((pool_flat / 64) ** 0.5))
    model = ImageClassifier(input_channels, feature_dim, classes, pool_size).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def evaluate_conditional_images(classifier, images, expected_labels):
    model_device = next(classifier.parameters()).device
    logits, features = classifier(images.to(model_device), return_features=True)
    probabilities = logits.softmax(1)
    predictions = probabilities.argmax(1)
    expected = expected_labels.to(model_device)
    accuracy = float((predictions == expected).float().mean())
    confidence = float(probabilities.gather(1, expected[:, None]).mean())
    return (
        {
            "conditional_accuracy": accuracy,
            "target_confidence": confidence,
        },
        features.cpu(),
        predictions.cpu(),
    )
