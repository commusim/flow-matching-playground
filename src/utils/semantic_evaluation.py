import torch
from src.modules.classifier import MNISTClassifier


def load_classifier(path, device, feature_dim=128):
    model = MNISTClassifier(feature_dim).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def evaluate_conditional_images(classifier, images, expected_labels):
    logits, features = classifier(
        images.to(next(classifier.parameters()).device), return_features=True
    )
    probabilities = logits.softmax(1)
    predictions = probabilities.argmax(1)
    accuracy = float(
        (predictions == expected_labels.to(predictions.device)).float().mean()
    )
    confidence = float(
        probabilities.gather(
            1, expected_labels.to(probabilities.device)[:, None]
        ).mean()
    )
    return (
        {"conditional_accuracy": accuracy, "target_confidence": confidence},
        features.cpu(),
        predictions.cpu(),
    )
