from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, FashionMNIST, ImageFolder, MNIST

DATASETS = {
    "mnist": (MNIST, 1, 10),
    "fashion_mnist": (FashionMNIST, 1, 10),
    "cifar10": (CIFAR10, 3, 10),
}


def dataset_spec(name):
    if name == "image_folder":
        return {"channels": None, "classes": None}
    if name not in DATASETS:
        raise ValueError(
            f"Unknown dataset: {name}. Available: {', '.join(DATASETS)} or image_folder"
        )
    _, channels, classes = DATASETS[name]
    return {"channels": channels, "classes": classes}


def create_image_loader(
    data_root,
    batch_size,
    dataset="mnist",
    train=True,
    image_size=28,
    input_channels=None,
    subset_size=None,
    download=False,
    num_workers=0,
):
    root = Path(data_root)
    if dataset == "image_folder":
        if input_channels is None:
            input_channels = 3
        dataset_class = ImageFolder
        default_channels = input_channels
    else:
        dataset_class, default_channels, _ = DATASETS[dataset]
        input_channels = default_channels if input_channels is None else input_channels
    operations = [transforms.Resize((image_size, image_size))]
    if input_channels != default_channels:
        operations.append(transforms.Grayscale(num_output_channels=input_channels))
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * input_channels, (0.5,) * input_channels),
        ]
    )
    transform = transforms.Compose(operations)
    if dataset == "image_folder":
        split = "train" if train else "test"
        source = root / split if (root / split).exists() else root
        image_dataset = dataset_class(source, transform=transform)
    else:
        image_dataset = dataset_class(
            root=root, train=train, transform=transform, download=download
        )
    if subset_size is not None and subset_size < len(image_dataset):
        image_dataset = Subset(image_dataset, range(subset_size))
    return DataLoader(
        image_dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=num_workers,
    )


def create_mnist_loader(
    data_root, batch_size, train=True, subset_size=None, download=False, num_workers=0
):
    return create_image_loader(
        data_root=data_root,
        batch_size=batch_size,
        dataset="mnist",
        train=train,
        image_size=28,
        input_channels=1,
        subset_size=subset_size,
        download=download,
        num_workers=num_workers,
    )


def infinite_batches(loader):
    while True:
        for batch in loader:
            yield batch
