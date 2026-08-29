import numpy as np
import torch
from src.modules.classifier import ImageClassifier
from src.utils.classifier_plotting import plot_confusion_matrix, plot_tsne_features
from src.utils.common import (
    create_run_dir,
    get_device,
    project_root,
    save_run_metadata,
    seed_everything,
)
from src.utils.image_data import create_image_loader, dataset_spec
from src.utils.plotting import plot_loss

DEFAULTS = {
    "pipeline": "mnist_classifier",
    "dataset": "mnist",
    "image_size": 28,
    "input_channels": None,
    "num_classes": None,
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
    cfg["num_classes"] = cfg["num_classes"] or spec["classes"]
    common = {
        "data_root": cfg["data_root"] or project_root() / "data",
        "batch_size": cfg["batch_size"],
        "dataset": cfg["dataset"],
        "image_size": cfg["image_size"],
        "input_channels": cfg["input_channels"],
        "download": False,
        "num_workers": 0,
    }
    train_loader = create_image_loader(
        train=True, subset_size=cfg["subset_size"], **common
    )
    test_loader = create_image_loader(
        train=False, subset_size=cfg["test_subset_size"], **common
    )
    model = ImageClassifier(
        cfg["input_channels"], cfg["feature_dim"], cfg["num_classes"]
    ).to(device)
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
    confusion = np.zeros((cfg["num_classes"], cfg["num_classes"]), dtype=np.int64)
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
