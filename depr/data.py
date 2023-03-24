import torchvision

from torch.utils.data import DataLoader, Subset
from torchvision import transforms

CIFAR10_CLASSES = (
    'plane', 'car', 'bird', 'cat',
    'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
)
DEFAULT_ROOT = './data/'


def load_federated_cifar10(
        train_indices: list[int],
        test_indices: list[int],
        batch_size: int = 64
) -> tuple[DataLoader, DataLoader]:
    # Load the respective dataloader.
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_transforms = transforms.Compose([
        torchvision.transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=DEFAULT_ROOT, train=True, download=True, transform=train_transforms)
    test_set = torchvision.datasets.CIFAR10(
        root=DEFAULT_ROOT, train=False, download=True, transform=test_transforms)

    train_set = Subset(train_set, train_indices)
    test_set = Subset(test_set, test_indices)

    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader
