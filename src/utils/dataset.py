import inspect
import os

import numpy as np
import torchvision.transforms as transforms
from torch import from_numpy
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, CIFAR100, Places365


class NpToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return from_numpy(image) / 255.0


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


class IndexDatasetWithLabels(Dataset):
    def __init__(self, dataset: Dataset, labels) -> None:
        for name, param in dataset.__dict__.items():
            self.__setattr__(name, param)
        for method in inspect.getmembers(dataset, predicate=inspect.ismethod):
            self.__setattr__(method[0], method[1])
        for method in inspect.getmembers(dataset, predicate=inspect.isfunction):
            self.__setattr__(method[0], method[1])
        self.dataset = dataset
        self.labels = labels


class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None, reshape=False):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.reshape = reshape

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.reshape:
            image = image.reshape(3, 64, 64).transpose(1, 2, 0)

        if self.transform:
            image = self.transform(image)

        return image, label


class CustomDatasetWithIndices(Dataset):
    def __init__(self, images, labels, indices, transform=None, reshape=False):
        self.images = images
        self.labels = labels
        self.indices = indices
        self.transform = transform
        self.reshape = reshape

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        index = self.indices[idx]

        if self.reshape:
            image = image.reshape(3, 64, 64).transpose(1, 2, 0)

        if self.transform:
            image = self.transform(image)

        return image, label, index


def get_transforms(mean, std, from_numpy=False, dataset_name="CIFAR10"):
    if dataset_name == "PLACES_365":
        # transform_train = transforms.Compose(
        #     [
        #         transforms.RandomResizedCrop(224),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean, std),
        #     ]
        # )

        # transform_test = transforms.Compose(
        #     [
        #         transforms.Resize(256),
        #         transforms.CenterCrop(224),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean, std),
        #     ]
        # )

        # TRANSFORMATION1
        transform_train = transforms.Compose(
            [
                transforms.Resize(80),
                transforms.RandomCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.Resize(80),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        # TRANSFORMATION2
        # transform_train = transforms.Compose([
        #     transforms.RandomResizedCrop(64),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean, std),
        # ])

        # transform_test = transforms.Compose([
        #     transforms.Resize(64),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean, std),
        # ])

    elif dataset_name == "IMAGENET":
        transform_train = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    else:
        if from_numpy:
            transform_train = transforms.Compose(
                [
                    NpToTensor(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize(mean, std),
                ]
            )

            transform_test = transforms.Compose(
                [
                    NpToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )

        else:
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


def get_dataset(dataset_name: str):
    if dataset_name == "CIFAR10":
        mean_cifar10 = (0.4914, 0.4822, 0.4465)
        std_cifar10 = (0.2470, 0.2435, 0.2616)
        transform_train, transform_test = get_transforms(mean_cifar10, std_cifar10)
        trainset = CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        testset = CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
        trainset = IndexDataset(trainset)

    elif dataset_name == "CIFAR100":
        mean_cifar100 = (0.5071, 0.4865, 0.4409)
        std_cifar100 = (0.2673, 0.2564, 0.2761)
        transform_train, transform_test = get_transforms(mean_cifar100, std_cifar100)
        trainset = CIFAR100(
            root="./data", train=True, download=True, transform=transform_train
        )
        testset = CIFAR100(
            root="./data", train=False, download=True, transform=transform_test
        )
        trainset = IndexDataset(trainset)

    elif dataset_name == "PLACES_365":
        mean_places365 = (0.485, 0.456, 0.406)
        std_places365 = (0.229, 0.224, 0.225)
        root_dir = "/ceph/ssd/shared/datasets/places-365"
        transform_train, transform_test = get_transforms(
            mean_places365,
            std_places365,
            dataset_name=dataset_name,
        )
        trainset = Places365(
            root=root_dir,
            split="train-standard",
            download=False,
            small=True,
            transform=transform_train,
        )
        testset = Places365(
            root=root_dir,
            split="val",
            download=False,
            small=True,
            transform=transform_test,
        )
        trainset = IndexDataset(trainset)

    elif dataset_name == "SYNTHETIC_CIFAR100_1M":
        mean_cifar100_syn = (0.5194, 0.4991, 0.4573)
        std_cifar100_syn = (0.2748, 0.2640, 0.2858)

        data = np.load(
            "/nfs/homedirs/dhp/unsupervised-data-pruning/data/cifar100_1m.npz"
        )

        num_samples = len(data["label"])
        train_images = data["image"]
        train_labels = data["label"]

        indices = np.arange(num_samples)

        transform_train, transform_test = get_transforms(
            mean_cifar100_syn, std_cifar100_syn, from_numpy=True
        )

        trainset = CustomDatasetWithIndices(
            train_images, train_labels, indices, transform=transform_train
        )
        testset = None

    elif dataset_name == "IMAGENET":
        mean_imagenet = (0.449, 0.426, 0.379)
        std_imagenet = (0.285, 0.276, 0.284)

        root_dir = "/ceph/ssd/shared/datasets/imagenet/imagenet64"
        train_file_list = [f"train_data_batch_{i}.npz" for i in range(1, 11)]

        train_data = []
        train_labels = []

        for file_name in train_file_list:
            file_path = os.path.join(root_dir, file_name)
            with np.load(file_path) as data_file:
                train_data.append(data_file["data"])
                train_labels.append(data_file["labels"])

        train_data = np.concatenate(train_data, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        train_labels = train_labels - 1

        num_samples = len(train_labels)
        indices = np.arange(num_samples)

        val_file = "val_data.npz"
        val_data = []
        val_labels = []

        with np.load(os.path.join(root_dir, val_file)) as data_file:
            val_data.append(data_file["data"])
            val_labels.append(data_file["labels"])

        val_data = np.concatenate(val_data, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)
        val_labels = val_labels - 1
        transform_train, transform_test = get_transforms(
            mean_imagenet, std_imagenet, from_numpy=True, dataset_name="IMAGENET"
        )
        trainset = CustomDatasetWithIndices(
            train_data, train_labels, indices, transform=transform_train, reshape=True
        )
        testset = CustomDataset(val_data, val_labels, transform_test, reshape=True)

    return trainset, testset


def get_dataloaders(dataset_name: str, batch_size: int = 128):
    trainset, testset = get_dataset(dataset_name)

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    testloader = DataLoader(
        testset, batch_size=2 * batch_size, shuffle=False, num_workers=2
    )

    return trainloader, testloader


def get_dataloaders_from_dataset(trainset, testset, batch_size: int = 128):
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    testloader = DataLoader(
        testset, batch_size=2 * batch_size, shuffle=False, num_workers=2
    )

    return trainloader, testloader


def prepare_data(dataset_cfg, batch_size):
    if dataset_cfg.name == "SYNTHETIC_CIFAR100_1M":
        trainset, testset = get_dataset(dataset_cfg.name)
        train_loader, _ = get_dataloaders_from_dataset(
            trainset, testset, batch_size=batch_size
        )
        _, test_loader = get_dataloaders("CIFAR100", batch_size=batch_size)
        num_train_examples = len(trainset)

    else:
        trainset, testset = get_dataset(dataset_cfg.name)
        train_loader, test_loader = get_dataloaders_from_dataset(
            trainset, testset, batch_size=batch_size
        )
        num_train_examples = len(trainset)
    return trainset, train_loader, test_loader, num_train_examples
