import numpy as np
import torch
from src.modules.classifier import MNISTClassifier
from src.utils.classifier_plotting import plot_confusion_matrix, plot_tsne_features
from src.utils.common import (
    create_run_dir,
    get_device,
    project_root,
    save_run_metadata,
    seed_everything,
)
from src.utils.image_data import create_mnist_loader
from src.utils.plotting import plot_loss

DEFAULTS = {
    "pipeline": "mnist_classifier",
    "seed": 42,
    "device": "auto",
    "epochs": 5,
    "batch_size": 128,
    "lr": 0.001,
    "feature_dim": 128,
    "subset_size": 60000,
    "test_subset_size": 10000,
    "tsne_samples": 1500,
    "checkpoint_path": None,
    "output_root": None,
}


def run(config):
    cfg = {**DEFAULTS, **config}
    seed_everything(cfg["seed"])
    device = get_device(cfg["device"])
    run_dir = create_run_dir(cfg["pipeline"], cfg["seed"], cfg["output_root"])
    train_loader = create_mnist_loader(
        project_root() / "data", cfg["batch_size"], True, cfg["subset_size"], False, 0
    )
    test_loader = create_mnist_loader(
        project_root() / "data",
        cfg["batch_size"],
        False,
        cfg["test_subset_size"],
        False,
        0,
    )
    model = MNISTClassifier(cfg["feature_dim"]).to(device)
    if cfg["checkpoint_path"]:
        model.load_state_dict(torch.load(cfg["checkpoint_path"], map_location=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    losses = []
    for epoch in range(cfg["epochs"]):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            loss = torch.nn.functional.cross_entropy(model(images), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(
            f"epoch {epoch + 1}/{cfg['epochs']} loss {np.mean(losses[-len(train_loader) :]):.5f}"
        )
    model.eval()
    confusion = np.zeros((10, 10), dtype=np.int64)
    correct = total = 0
    all_features = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            logits, features = model(images.to(device), return_features=True)
            predictions = logits.argmax(1).cpu()
            for true, pred in zip(labels.numpy(), predictions.numpy()):
                confusion[true, pred] += 1
            correct += int((predictions == labels).sum())
            total += len(labels)
            if sum(len(item) for item in all_labels) < cfg["tsne_samples"]:
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.numpy())
    accuracy = correct / total
    torch.save(model.state_dict(), run_dir / "checkpoint.pt")
    plot_loss(losses, run_dir / "loss_curve.png")
    plot_confusion_matrix(confusion, run_dir / "confusion_matrix.png")
    features = np.concatenate(all_features)[: cfg["tsne_samples"]]
    labels = np.concatenate(all_labels)[: cfg["tsne_samples"]]
    plot_tsne_features(
        features, labels, run_dir / "semantic_features_tsne.png", cfg["seed"]
    )
    save_run_metadata(
        run_dir,
        cfg,
        {
            "test_accuracy": accuracy,
            "parameters": sum(p.numel() for p in model.parameters()),
            "device": str(device),
        },
    )
    print(f"accuracy: {accuracy:.4f}")
    print(f"outputs: {run_dir}")
    return run_dir
