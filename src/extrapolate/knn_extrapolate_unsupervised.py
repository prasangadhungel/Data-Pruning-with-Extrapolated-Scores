import datetime
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from loguru import logger
from omegaconf import OmegaConf
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from torch_cluster.knn import knn
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.dataset import prepare_data
from utils.helpers import parse_config
from utils.models import load_model_by_name

logger.remove()
logger.add(sys.stdout, format="{time:MM-DD HH:mm} - {message}")


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def _convert_image_to_rgb(image):
    if torch.is_tensor(image):
        return image
    else:
        return image.convert("RGB")


def _safe_to_tensor(x):
    if torch.is_tensor(x):
        return x
    else:
        return transforms.ToTensor()(x)


def get_default_transforms():
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            _convert_image_to_rgb,
            _safe_to_tensor,
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )


class IndexDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return (item[0], item[1], idx)


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


def get_datasets(dataset, transform, root_dir="./data"):
    data_path = os.path.join(root_dir, "datasets")

    if dataset == "CIFAR10":
        train_dataset = dsets.CIFAR10(
            root=data_path, train=True, transform=transform, download=True
        )
        val_dataset = dsets.CIFAR10(
            root=data_path, train=False, transform=transform, download=True
        )
    elif dataset == "CIFAR100":
        train_dataset = dsets.CIFAR100(
            root=data_path, train=True, transform=transform, download=True
        )
        val_dataset = dsets.CIFAR100(
            root=data_path, train=False, transform=transform, download=True
        )
    elif dataset == "SYNTHETIC_CIFAR100_1M":
        data = np.load(
            "/nfs/homedirs/dhp/unsupervised-data-pruning/data/cifar100_1m.npz"
        )

        num_samples = len(data["label"])
        train_images = data["image"]
        train_labels = data["label"]

        indices = np.arange(num_samples)
        transform = get_default_transforms()
        train_dataset = CustomDatasetWithIndices(
            train_images, train_labels, indices, transform=transform
        )
        val_dataset = dsets.CIFAR100(
            root=data_path, train=False, transform=transform, download=True
        )
        val_dataset = IndexDataset(val_dataset)
        return train_dataset, val_dataset

    train_dataset = IndexDataset(train_dataset)
    val_dataset = IndexDataset(val_dataset)

    return train_dataset, val_dataset


def get_dataloaders(dataset, transform, batch_size, root_dir="data"):
    if transform is None:
        transform = get_default_transforms()
    train_dataset, val_dataset = get_datasets(dataset, transform, root_dir)
    trainloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=10
    )
    valloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=10
    )
    return trainloader, valloader


def get_dataloader_from_trainset(trainset, batch_size, shuffle=False):
    return DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=10)


def get_features(dataloader, model, device):
    features_dict = {}
    with torch.no_grad():
        for x, y, idx in tqdm(dataloader):
            features = model(x.to(device)).detach().cpu().numpy()
            for i, index in enumerate(idx.numpy()):
                features_dict[index] = features[i]

    sorted_indices = sorted(features_dict.keys())
    sorted_features = np.array([features_dict[i] for i in sorted_indices])
    # convert it to torch tensor
    sorted_features = torch.tensor(sorted_features, dtype=torch.float32)
    return sorted_features


def run_representation(args, device="cuda"):
    torch.hub.set_dir(args.model.path)
    model = torch.hub.load(args.model.torch_hub, args.model.version).to(device)
    model.eval()
    logger.info(
        f"Model parameters: {np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}"
    )
    preprocess = None

    trainloader, valloader = get_dataloaders(
        args.dataset.name,
        preprocess,
        args.training.batch_size_repr,
        args.dataset.root_dir,
    )
    feats_train = get_features(trainloader, model, device)
    feats_val = get_features(valloader, model, device)

    return feats_train, feats_val


def get_correlation(
    embeddings, subset_scores_dict, k, full_scores_dict, distance_metric, device
):
    all_samples = list(range(len(embeddings)))
    seed_samples = [int(key) for key in subset_scores_dict.keys()]

    unseeded_samples = [s for s in all_samples if s not in seed_samples]

    unseed_embeddings = torch.stack(
        [embeddings[s].flatten() for s in unseeded_samples], dim=0
    ).to(device)

    seed_embeddings = torch.stack(
        [embeddings[s].flatten() for s in seed_samples], dim=0
    ).to(device)

    seed_scores_we_have = torch.tensor(
        [subset_scores_dict[str(s)] for s in seed_samples],
        dtype=torch.float,
        device=device,
    )

    unseed_scores_true = torch.tensor(
        [full_scores_dict[str(u)] for u in unseeded_samples],
        dtype=torch.float,
        device=device,
    )

    use_cosine = distance_metric == "cosine"

    unseed_idx, seed_idx = knn(
        x=seed_embeddings,  # source
        y=unseed_embeddings,  # target
        k=k,
        cosine=use_cosine,
    )

    neighbor_scores = seed_scores_we_have[seed_idx]  # shape: [k * U]

    U = len(unseeded_samples)
    sum_unweighted = torch.zeros(U, dtype=torch.float, device=device)
    sum_weighted = torch.zeros(U, dtype=torch.float, device=device)
    sum_weights = torch.zeros(U, dtype=torch.float, device=device)

    # diffs = seed_embeddings[seed_idx] - unseed_embeddings[unseed_idx]
    # dists = diffs.norm(p=2, dim=1)  # shape: [k * U]
    # weights = torch.exp(-dists)     # shape: [k * U]
    # This will create CUDA OOM, so we need to compute in chunks

    B = seed_idx.shape[0]
    chunk_size = 20000

    for i in range(0, B, chunk_size):
        end = min(i + chunk_size, B)
        si = seed_idx[i:end]
        ui = unseed_idx[i:end]

        diffs = seed_embeddings[si] - unseed_embeddings[ui]
        chunk_dists = diffs.norm(p=2, dim=1)  # shape: [chunk_size]
        chunk_weights = torch.exp(-chunk_dists)  # shape: [chunk_size]

        chunk_scores = neighbor_scores[i:end]  # shape: [chunk_size]
        sum_unweighted.index_add_(0, ui, chunk_scores)
        sum_weighted.index_add_(0, ui, chunk_scores * chunk_weights)
        sum_weights.index_add_(0, ui, chunk_weights)

    knn_avg_scores = sum_unweighted / k
    knn_weighted_scores = sum_weighted / sum_weights

    unseed_scores_np = unseed_scores_true.cpu().numpy()
    knn_avg_scores_np = knn_avg_scores.cpu().numpy()
    knn_weighted_scores_np = knn_weighted_scores.cpu().numpy()

    corr_avg = np.corrcoef(unseed_scores_np, knn_avg_scores_np)[0, 1]
    corr_weighted = np.corrcoef(unseed_scores_np, knn_weighted_scores_np)[0, 1]

    spearman_avg = spearmanr(unseed_scores_np, knn_avg_scores_np).correlation
    spearman_weighted = spearmanr(unseed_scores_np, knn_weighted_scores_np).correlation

    logger.info(f"Average Correlation: {corr_avg}, Spearman: {spearman_avg}")
    logger.info(f"Weighted Correlation: {corr_weighted}, Spearman: {spearman_weighted}")

    knn_dict_weighted = {}
    knn_dict_avg = {}

    for i, sample_id in enumerate(unseeded_samples):
        knn_dict_weighted[str(sample_id)] = float(knn_weighted_scores_np[i])
        knn_dict_avg[str(sample_id)] = float(knn_avg_scores_np[i])

    for sample_id in seed_samples:
        knn_dict_weighted[str(sample_id)] = float(subset_scores_dict[str(sample_id)])
        knn_dict_avg[str(sample_id)] = float(subset_scores_dict[str(sample_id)])

    return (
        corr_avg,
        corr_weighted,
        spearman_avg,
        spearman_weighted,
        knn_dict_weighted,
        knn_dict_avg,
    )


def main(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)
    cfg = cfg.CIFAR10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dataset: {cfg.dataset.name}, Device: {device}")

    with open(cfg.scores.original_scores_file) as f:
        full_scores_dict = json.load(f)

    with open(cfg.scores.subset_scores_file) as f:
        subset_scores_dict = json.load(f)

    logger.info(f"Number of samples in subset: {len(subset_scores_dict)}")
    subset_scores_np = np.array(
        [
            subset_scores_dict[str(i)]
            for i in range(len(full_scores_dict.keys()))
            if str(i) in subset_scores_dict
        ]
    )
    full_scores_np = np.array(
        [
            full_scores_dict[str(i)]
            for i in range(len(full_scores_dict.keys()))
            if str(i) in subset_scores_dict
        ]
    )

    # check if the scores contain NaNs
    if np.isnan(subset_scores_np).any():
        logger.error("Subset scores contain NaNs")

    if np.isnan(full_scores_np).any():
        logger.error("Full scores contain NaNs")

    corr = np.corrcoef(full_scores_np, subset_scores_np)[0, 1]
    spearman = spearmanr(full_scores_np, subset_scores_np).correlation
    mse = np.mean((full_scores_np - subset_scores_np) ** 2)
    logger.info(f"Max achievable correlation: {corr} Spearman: {spearman} MSE: {mse}")

    if cfg.scores.normalize_scores:
        # min-max normalization
        logger.info("Normalizing scores with min-max normalization")
        subset_scores_dict = {
            key: (value - np.min(subset_scores_np))
            / (np.max(subset_scores_np) - np.min(subset_scores_np))
            for key, value in subset_scores_dict.items()
        }
        full_scores_dict = {
            key: (value - np.min(full_scores_np))
            / (np.max(full_scores_np) - np.min(full_scores_np))
            for key, value in full_scores_dict.items()
        }

        subset_scores_np = (subset_scores_np - np.min(subset_scores_np)) / (
            np.max(subset_scores_np) - np.min(subset_scores_np)
        )
        full_scores_np = (full_scores_np - np.min(full_scores_np)) / (
            np.max(full_scores_np) - np.min(full_scores_np)
        )

        corr = np.corrcoef(full_scores_np, subset_scores_np)[0, 1]
        spearman = spearmanr(full_scores_np, subset_scores_np).correlation
        mse = np.mean((full_scores_np - subset_scores_np) ** 2)
        logger.info(
            f"Max achievable correlation: {corr} Spearman: {spearman} MSE: {mse}"
        )

    models, ks, num_seeds = [], [], []

    for model_name in tqdm(cfg.models.names):
        if cfg.checkpoints.read_embeddings:
            embeddings = torch.load(
                cfg.checkpoints.embeddings_file, map_location=device
            )
            logger.info(f"Loaded embeddings from {cfg.checkpoints.embeddings_file}")

        else:
            torch.hub.set_dir(cfg.models.path)
            model = torch.hub.load(cfg.models.torch_hub, cfg.models.version).to(device)
            model.eval()
            preprocess = None

            trainloader, _ = get_dataloaders(
                cfg.dataset.name,
                preprocess,
                cfg.training.batch_size_repr,
                cfg.dataset.root_dir,
            )

            embeddings = get_features(trainloader, model, device)

            if cfg.checkpoints.save_embeddings:
                torch.save(embeddings, cfg.checkpoints.embeddings_file)
                logger.info(f"Saved embeddings to {cfg.checkpoints.embeddings_file}")

        for k in tqdm(cfg.hyperparams.k_values):
            distance = cfg.hyperparams.distance
            (
                corr_avg,
                corr_weighted,
                spearman_avg,
                spearman_weighted,
                knn_dict_weighted,
                knn_dict_avg,
            ) = get_correlation(
                embeddings,
                subset_scores_dict,
                k,
                full_scores_dict,
                distance_metric=distance,
                device=device,
            )
            num_seed = len(subset_scores_dict)

            models.append(model_name)
            ks.append(k)
            num_seeds.append(num_seed)

            logger.info(
                f"k: {k}, num_seed: {num_seed}, distance_metric: {distance}, corr_avg: {corr_avg}, corr_weighted: {corr_weighted}, spearman_avg: {spearman_avg}, spearman_weighted: {spearman_weighted}",
            )

            date = datetime.datetime.now()
            curr_datetime = f"_{date.month}_{date.day}"

            filename_weighted = f"{cfg.output.knn_dict_path}_{cfg.scores.type}_{cfg.dataset.name}_weighted_{model_name}_k_{k}_seed_{num_seed}_{distance}_{curr_datetime}.json"
            filename_avg = f"{cfg.output.knn_dict_path}_{cfg.scores.type}_{cfg.dataset.name}_avg_{model_name}_k_{k}_seed_{num_seed}_{distance}_{curr_datetime}.json"

            with open(filename_weighted, "w") as f:
                json.dump(knn_dict_weighted, f)

            logger.info(f"Saved weighted scores to {filename_weighted}")

            with open(filename_avg, "w") as f:
                json.dump(knn_dict_avg, f)

            logger.info(f"Saved average scores to {filename_avg}")


if __name__ == "__main__":
    default_config_path = os.path.join(
        os.path.dirname(__file__), "configs", "knn_config_unsupervised.yaml"
    )
    config_path = parse_config(
        default_config=default_config_path, description="Run KNN Extrapolation"
    )
    main(cfg_path=config_path)
