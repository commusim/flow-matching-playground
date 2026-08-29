from torch import nn


class ImageClassifier(nn.Module):
    """Supervised CNN with configurable channels/classes and size-independent pooling."""

    def __init__(self, input_channels=1, feature_dim=128, classes=10, pool_size=7):
        super().__init__()
        self.pool_size = pool_size
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((pool_size, pool_size)),
        )
        self.feature_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * pool_size * pool_size, feature_dim),
            nn.SiLU(),
        )
        self.classifier = nn.Linear(feature_dim, classes)

    def forward(self, images, return_features=False):
        features = self.feature_projection(self.encoder(images))
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


MNISTClassifier = ImageClassifier
