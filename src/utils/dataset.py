import inspect
import json

import numpy as np
import torchvision.transforms as transforms

# import Dataset
from torch import from_numpy
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, CIFAR100


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
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class CustomDatasetWithIndices(Dataset):
    def __init__(self, images, labels, indices, transform=None):
        self.images = images
        self.labels = labels
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        index = self.indices[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, index


def get_transforms(mean, std, from_numpy=False):
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


def get_dataset(dataset_name: str, partial=False, subset_idxs=[0]):
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

    elif dataset_name == "SYNTHETIC_CIFAR100_50":
        mean_cifar100_syn = (0.5321, 0.5066, 0.4586)
        std_cifar100_syn = (0.2673, 0.2564, 0.2761)

        data = np.load(
            "/ceph/ssd/shared/datasets/cifar100_synthetic/cifar100_50m_part1.npz"
        )

        num_samples = len(data["label"])
        np.random.seed(40)
        if not partial:
            indices = np.random.permutation(num_samples)

            train_images = data["image"][indices[: int(0.8 * num_samples)]]
            train_labels = data["label"][indices[: int(0.8 * num_samples)]]

            test_images = data["image"][indices[int(0.8 * num_samples) :]]
            test_labels = data["label"][indices[int(0.8 * num_samples) :]]

            mean_cifar100_syn = (0.5321, 0.5066, 0.4586)
            std_cifar100_syn = (0.2673, 0.2564, 0.2761)

            transform_train, transform_test = get_transforms(
                mean_cifar100_syn, std_cifar100_syn, from_numpy=True
            )

            trainset = CustomDatasetWithIndices(
                train_images, train_labels, indices, transform=transform_train
            )
            testset = CustomDatasetWithIndices(
                test_images, test_labels, indices, transform=transform_test
            )

        else:
            data_files = [
                "/ceph/ssd/shared/datasets/cifar100_synthetic/cifar100_50m_part1.npz",
                "/ceph/ssd/shared/datasets/cifar100_synthetic/cifar100_50m_part2.npz",
                "/ceph/ssd/shared/datasets/cifar100_synthetic/cifar100_50m_part3.npz",
                "/ceph/ssd/shared/datasets/cifar100_synthetic/cifar100_50m_part4.npz",
            ]
            images = []
            labels = []
            for file in data_files:
                data = np.load(file)
                images.append(data["image"])
                labels.append(data["label"])
            images = np.concatenate(images)
            labels = np.concatenate(labels)

            # read json /nfs/homedirs/dhp/unsupervised-data-pruning/data/subset_indices_synthetic_cifar_1M_total_10.0_percentage.json
            with open(
                "/nfs/homedirs/dhp/unsupervised-data-pruning/data/subset_indices_synthetic_cifar_50M_total_2_percentage.json",
                "r",
            ) as f:
                indices_dict = json.load(f)

            choosen_indices = []
            for subset_idx in subset_idxs:
                choosen_indices.extend(indices_dict[str(subset_idx)])

            train_images = images[choosen_indices]
            train_labels = labels[choosen_indices]

            test_images = images[indices_dict["test"]]
            test_labels = labels[indices_dict["test"]]

            mean_cifar100_syn = (0.5321, 0.5066, 0.4586)
            std_cifar100_syn = (0.2673, 0.2564, 0.2761)

            transform_train, transform_test = get_transforms(
                mean_cifar100_syn, std_cifar100_syn, from_numpy=True
            )

            trainset = CustomDatasetWithIndices(
                train_images, train_labels, choosen_indices, transform=transform_train
            )
            testset = CustomDataset(test_images, test_labels, transform=transform_test)

    elif dataset_name == "SYNTHETIC_CIFAR100_1M":
        mean_cifar100_syn = (0.5194, 0.4991, 0.4573)
        std_cifar100_syn = (0.2748, 0.2640, 0.2858)

        data = np.load("/nfs/homedirs/dhp/unsupervised-data-pruning/data/1m.npz")

        num_samples = len(data["label"])
        np.random.seed(40)
        if not partial:
            indices = np.random.permutation(num_samples)

            train_images = data["image"][indices[: int(0.8 * num_samples)]]
            train_labels = data["label"][indices[: int(0.8 * num_samples)]]

            test_images = data["image"][indices[int(0.8 * num_samples) :]]
            test_labels = data["label"][indices[int(0.8 * num_samples) :]]

            mean_cifar100_syn = (0.5194, 0.4991, 0.4573)
            std_cifar100_syn = (0.2748, 0.2640, 0.2858)

            transform_train, transform_test = get_transforms(
                mean_cifar100_syn, std_cifar100_syn, from_numpy=True
            )

            trainset = CustomDatasetWithIndices(
                train_images, train_labels, indices, transform=transform_train
            )
            testset = CustomDatasetWithIndices(
                test_images, test_labels, indices, transform=transform_test
            )

        else:
            data = np.load("/nfs/homedirs/dhp/unsupervised-data-pruning/data/1m.npz")

            images = data["image"]
            labels = data["label"]

            # read json /nfs/homedirs/dhp/unsupervised-data-pruning/data/subset_indices_synthetic_cifar_1M_total_10.0_percentage.json
            with open(
                "/nfs/homedirs/dhp/unsupervised-data-pruning/data/subset_indices_synthetic_cifar_1M_total_99_percentage_pruned_knn.json",
                "r",
            ) as f:
                indices_dict = json.load(f)

            choosen_indices = []
            for subset_idx in subset_idxs:
                choosen_indices.extend(indices_dict[str(subset_idx)])

            train_images = images[choosen_indices]
            train_labels = labels[choosen_indices]

            test_images = images[indices_dict["test"]]
            test_labels = labels[indices_dict["test"]]

            mean_cifar100_syn = (0.5194, 0.4991, 0.4573)
            std_cifar100_syn = (0.2748, 0.2640, 0.2858)

            transform_train, transform_test = get_transforms(
                mean_cifar100_syn, std_cifar100_syn, from_numpy=True
            )

            trainset = CustomDatasetWithIndices(
                train_images, train_labels, choosen_indices, transform=transform_train
            )
            testset = CustomDataset(test_images, test_labels, transform=transform_test)

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
