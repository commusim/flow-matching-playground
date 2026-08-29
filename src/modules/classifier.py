from torch import nn


class MNISTClassifier(nn.Module):
    """Small supervised CNN used as a semantic feature extractor and evaluator."""

    def __init__(self, feature_dim=128, classes=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.feature_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, feature_dim),
            nn.SiLU(),
        )
        self.classifier = nn.Linear(feature_dim, classes)

    def forward(self, images, return_features=False):
        features = self.feature_projection(self.encoder(images))
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits
