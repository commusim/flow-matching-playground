from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST


def create_mnist_loader(
    data_root, batch_size, train=True, subset_size=None, download=False, num_workers=0
):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    dataset = MNIST(
        root=Path(data_root), train=train, transform=transform, download=download
    )
    if subset_size is not None and subset_size < len(dataset):
        dataset = Subset(dataset, range(subset_size))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=num_workers,
    )


def infinite_batches(loader):
    while True:
        for batch in loader:
            yield batch
