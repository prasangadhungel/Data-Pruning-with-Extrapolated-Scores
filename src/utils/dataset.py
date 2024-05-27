import inspect

import torchvision.transforms as transforms
# import Dataset
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, CIFAR100


class IndexDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        for name, param in dataset.__dict__.items():
            self.__setattr__(name, param)
        for method in inspect.getmembers(dataset, predicate=inspect.ismethod):
            self.__setattr__(method[0], method[1])
        for method in inspect.getmembers(dataset, predicate=inspect.isfunction):
            self.__setattr__(method[0], method[1])
        self.dataset = dataset

    def __len__(self) -> int:
        return self.dataset.__len__()

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the image and the mapped class index.
        """
        item = self.dataset.__getitem__(idx)
        return (item[0], item[1], idx)


def get_transforms(mean, std):
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return transform_train, transform_test


def get_dataset(dataset_name: str, batch_size: int = 128):
    if dataset_name == "CIFAR10":
        mean_cifar10 = (0.4914, 0.4822, 0.4465)
        std_cifar10 = (0.2470, 0.2435, 0.2616)
        transform_train, transform_test = get_transforms(mean_cifar10, std_cifar10)
        trainset = CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        trainloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        testset = CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
        testloader = DataLoader(
            testset, batch_size=2 * batch_size, shuffle=False, num_workers=2
        )

    elif dataset_name == "CIFAR100":
        mean_cifar100 = (0.5071, 0.4865, 0.4409)
        std_cifar100 = (0.2673, 0.2564, 0.2761)
        transform_train, transform_test = get_transforms(mean_cifar100, std_cifar100)
        trainset = CIFAR100(
            root="./data", train=True, download=True, transform=transform_train
        )
        trainloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2
        )

        testset = CIFAR100(
            root="./data", train=False, download=True, transform=transform_test
        )
        testloader = DataLoader(
            testset, batch_size=2 * batch_size, shuffle=False, num_workers=2
        )

    return trainloader, testloader
