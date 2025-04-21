import datetime
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from loguru import logger
from omegaconf import OmegaConf
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCNConv, knn_graph
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from torchvision.datasets import Places365

from utils.helpers import parse_config, seed_everything
from utils.dataset import prepare_data

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

    elif dataset == "PLACES_365":
        root_dir = "/ceph/ssd/shared/datasets/places-365"
        transform = get_default_transforms()

        train_dataset = Places365(
            root=root_dir,
            split="train-standard",
            download=False,
            small=True,
            transform=transform,
        )
        val_dataset = Places365(
            root=root_dir, split="val", download=False, small=True, transform=transform
        )

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


class GNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.5):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, 512)
        self.conv2 = GCNConv(512, 256)
        self.conv3 = GCNConv(256, output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index, edge_attr)
        return x


def get_edges_and_attributes(
    embeddings,
    k=10,
    distance="euclidean",
    read_knn=False,
    save_knn=True,
    knn_file=None,
    read_edge_attr=False,
    save_edge_attr=True,
    edge_attr_file=None,
    device=torch.device("cuda"),
):

    if distance == "cosine":
        embeddings = F.normalize(embeddings, p=2, dim=1)

    # knn_graph will return edge_index with shape [2, E]
    if read_knn:
        edge_index = torch.load(knn_file, map_location=device)
        logger.info(f"Loaded edge_index from {knn_file}")

    else:
        logger.info("Computing kNN graph")
        edge_index = knn_graph(embeddings, k, loop=False)
        logger.info("Finished computing kNN graph")

        if save_knn:
            torch.save(edge_index, knn_file)
            logger.info(f"Saved edge_index to {knn_file}")

    # Compute distances for each edge:
    src, dst = edge_index

    if read_edge_attr:
        edge_attr = torch.load(edge_attr_file, map_location=device)
        logger.info(f"Loaded edge_attr from {edge_attr_file}")

    else:
        logger.info(f"Computing edge attributes using {distance} distance")
        logger.info(f"Length of src: {len(src)}")
        logger.info(f"Length of dst: {len(dst)}")
        # only do this if you have large CUDA memory
        # dist = (embeddings[src] - embeddings[dst]).pow(2).sum(dim=-1).sqrt()

        # if you have small CUDA memory, do this instead
        chunk_size = 10000
        for i in range(0, len(src), chunk_size):
            # if i % 200 == 0:
            #     logger.info(f"Processing chunk {i} to {i + chunk_size}")

            chunk_src = src[i : i + chunk_size]
            chunk_dst = dst[i : i + chunk_size]
            chunk_dist = (
                (embeddings[chunk_src] - embeddings[chunk_dst])
                .pow(2)
                .sum(dim=-1)
                .sqrt()
            )
            if i == 0:
                dist = chunk_dist
            else:
                dist = torch.cat((dist, chunk_dist), dim=0)

        edge_attr = torch.exp(-dist)
        if save_edge_attr:
            torch.save(edge_attr, edge_attr_file)
            logger.info(f"Saved edge_attr to {edge_attr_file}")

    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    edge_attr = torch.sqrt(edge_attr)

    logger.info("Finished computing edge attributes")
    return edge_index, edge_attr


def prepare_data_graph(
    embeddings,
    seed_samples,
    training_dict,
    k,
    read_knn=False,
    save_knn=True,
    knn_file=None,
    read_edge_attr=False,
    save_edge_attr=True,
    edge_attr_file=None,
    val_frac=0.1,
    distance="euclidean",
    device="cuda",
):
    samples_list = list(range(len(embeddings)))

    y = torch.tensor(
        [training_dict[str(i)] for i in samples_list],
        dtype=torch.float,
        device=device,
    )

    # Compute edges and attrs using GPU
    edge_index, edge_attr = get_edges_and_attributes(
        embeddings,
        k,
        distance,
        read_knn,
        save_knn,
        knn_file,
        read_edge_attr,
        save_edge_attr,
        edge_attr_file,
        device=device,
    )

    x = torch.tensor(
        embeddings,
        dtype=torch.float,
        device=device,
    )
    val_idxs = random.sample(seed_samples, int(val_frac * len(seed_samples)))
    train_idxs = [i for i in seed_samples if i not in val_idxs]
    test_idx = [i for i in samples_list if i not in seed_samples]

    train_mask = torch.zeros(y.size(0), dtype=torch.bool)
    train_mask[train_idxs] = True
    val_mask = torch.zeros(y.size(0), dtype=torch.bool)
    val_mask[val_idxs] = True
    test_mask = torch.zeros(y.size(0), dtype=torch.bool)
    test_mask[test_idx] = True

    logger.info("Finished preparing data")
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


@torch.no_grad()
def evaluate(
    model,
    test_loader,
    num_nodes,
    device,
    orig_train,
    orig_val,
    orig_test,
    train_mask,
    val_mask,
    test_mask,
):
    model.eval()
    all_preds = torch.empty(num_nodes, device=device)
    for sub_data in test_loader:
        sub_data = sub_data.to(device)
        out_sub = model(sub_data.x, sub_data.edge_index, sub_data.edge_attr).squeeze()
        out_root = out_sub[: sub_data.batch_size]
        node_ids = sub_data.n_id[: sub_data.batch_size]
        all_preds[node_ids] = out_root

    pred_train = all_preds[train_mask].detach().cpu().numpy()
    corr_train = np.corrcoef(orig_train, pred_train)[0, 1]
    spearman_train = spearmanr(orig_train, pred_train).correlation
    mse_train = np.mean((orig_train - pred_train) ** 2)

    pred_val = all_preds[val_mask].detach().cpu().numpy()
    corr_val = np.corrcoef(orig_val, pred_val)[0, 1]
    spearman_val = spearmanr(orig_val, pred_val).correlation
    mse_val = np.mean((orig_val - pred_val) ** 2)

    pred_test = all_preds[test_mask].detach().cpu().numpy()
    corr_test = np.corrcoef(orig_test, pred_test)[0, 1]
    spearman_test = spearmanr(orig_test, pred_test).correlation
    mse_test = np.mean((orig_test - pred_test) ** 2)

    return (
        all_preds,
        corr_train,
        spearman_train,
        mse_train,
        corr_val,
        spearman_val,
        mse_val,
        corr_test,
        spearman_test,
        mse_test,
    )


def main(cfg_path: str):
    seed_everything(42)

    cfg = OmegaConf.load(cfg_path)
    cfg = cfg.CIFAR10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, _, num_samples = prepare_data(cfg.dataset, 1024)
    logger.info(f"Loaded dataset: {cfg.dataset.name}, Device: {device}")

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

    seed_samples = [int(key) for key in subset_scores_dict.keys()]
    num_seed = len(seed_samples)
    unseed_samples = [i for i in range(num_samples) if i not in seed_samples]

    # full scores dict are just used for evaluation
    # we won't have access to them during training
    # but we will have access to subset scores dict
    training_dict = subset_scores_dict.copy()
    for samples in unseed_samples:
        training_dict[str(samples)] = 0.0

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

            data = prepare_data_graph(
                embeddings,
                seed_samples,
                training_dict,
                k,
                read_knn=cfg.checkpoints.read_knn,
                save_knn=cfg.checkpoints.save_knn,
                knn_file=cfg.checkpoints.knn_path + "knn_k_" + str(k) + ".pth",
                read_edge_attr=cfg.checkpoints.read_edge_attr,
                save_edge_attr=cfg.checkpoints.save_edge_attr,
                edge_attr_file=cfg.checkpoints.edge_attr_path
                + "edge_attr_k_"
                + str(k)
                + ".pth",
                distance=cfg.hyperparams.distance,
                device=device,
            ).to(device)

            logger.info(f"Finished preparing data for k={k}")

            neighbor_samples = [
                10,
                10,
            ]

            train_loader = NeighborLoader(
                data,
                num_neighbors=neighbor_samples,
                batch_size=cfg.hyperparams.batch_size,
                shuffle=True,
                input_nodes=data.train_mask,
            )

            all_loader = NeighborLoader(
                data,
                num_neighbors=[-1],
                batch_size=1024,
                shuffle=False,
                input_nodes=None,
            )

            model = GNN(input_dim=data.num_node_features, output_dim=1).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=cfg.hyperparams.lr,
            )

            orig_train = np.array(
                [
                    subset_scores_dict[str(i)]
                    for i in range(num_samples)
                    if data.train_mask[i]
                ]
            )
            orig_val = np.array(
                [
                    subset_scores_dict[str(i)]
                    for i in range(num_samples)
                    if data.val_mask[i]
                ]
            )
            orig_test = np.array(
                [
                    full_scores_dict[str(i)]
                    for i in range(num_samples)
                    if data.test_mask[i]
                ]
            )

            (
                all_preds,
                corr_train,
                spearman_train,
                mse_train,
                corr_val,
                spearman_val,
                mse_val,
                corr_test,
                spearman_test,
                mse_test,
            ) = evaluate(
                model,
                all_loader,
                data.num_nodes,
                device,
                orig_train,
                orig_val,
                orig_test,
                data.train_mask,
                data.val_mask,
                data.test_mask,
            )

            logger.info(
                f"[Before training] Train - Corr: {corr_train:3f} Spearman: {spearman_train:3f} MSE: {mse_train:4f}"
            )
            logger.info(
                f"[Before training] Val - Corr: {corr_val:3f} Spearman: {spearman_val:3f} MSE: {mse_val:4f}"
            )
            logger.info(
                f"[Before training] Test - Corr: {corr_test:3f} Spearman: {spearman_test:3f} MSE: {mse_test:4f}"
            )

            best_val_corr, best_val_spearman, best_test_corr, best_test_spearman = (
                -1,
                -1,
                -1,
                -1,
            )

            for epoch in range(cfg.hyperparams.epochs):
                model.train()
                train_losses = []

                for batch_idx, batch_data in enumerate(train_loader):
                    batch_data = batch_data.to(device)
                    optimizer.zero_grad()
                    out = model(
                        batch_data.x, batch_data.edge_index, batch_data.edge_attr
                    ).squeeze()
                    out_root = out[: batch_data.batch_size]
                    y_root = batch_data.y[: batch_data.batch_size]
                    loss = F.mse_loss(out_root, y_root)
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss)

                    if batch_idx % 1000 == 0 and batch_idx > 0:
                        (
                            _,
                            corr_train,
                            spearman_train,
                            mse_train,
                            corr_val,
                            spearman_val,
                            mse_val,
                            corr_test,
                            spearman_test,
                            mse_test,
                        ) = evaluate(
                            model,
                            all_loader,
                            data.num_nodes,
                            device,
                            orig_train,
                            orig_val,
                            orig_test,
                            data.train_mask,
                            data.val_mask,
                            data.test_mask,
                        )
                        logger.info(
                            f"Batch={batch_idx} Train: {corr_train:3f} val: {corr_val:3f} test: {corr_test:3f}"
                        )

                logger.info(
                    f"GNN Epoch: {epoch}, Loss: {torch.stack(train_losses).mean().item():.5f}"
                )

                (
                    all_preds,
                    corr_train,
                    spearman_train,
                    mse_train,
                    corr_val,
                    spearman_val,
                    mse_val,
                    corr_test,
                    spearman_test,
                    mse_test,
                ) = evaluate(
                    model,
                    all_loader,
                    data.num_nodes,
                    device,
                    orig_train,
                    orig_val,
                    orig_test,
                    data.train_mask,
                    data.val_mask,
                    data.test_mask,
                )

                logger.info(
                    f"Step={epoch} Train - Corr: {corr_train:3f} Spearman: {spearman_train:3f} MSE: {mse_train:4f}"
                )
                logger.info(
                    f"Step={epoch} Val - Corr: {corr_val:3f} Spearman: {spearman_val:3f} MSE: {mse_val:4f}"
                )
                logger.info(
                    f"Step={epoch} Test - Corr: {corr_test:3f} Spearman: {spearman_test:3f} MSE: {mse_test:4f}"
                )

                if corr_val > best_val_corr:
                    logger.info(f"New best val corr: {corr_val}")
                    best_val_corr = corr_val
                    best_val_spearman = spearman_val
                    best_test_corr = corr_test
                    best_test_spearman = spearman_test

                    saved_scores = subset_scores_dict.copy()

                    for i in unseed_samples:
                        saved_scores[str(i)] = all_preds[i].item()

            # calculate correlation
            logger.info(f"Best Val Corr: {best_val_corr} Spearman: {best_val_spearman}")
            logger.info(
                f"Best Test Corr: {best_test_corr} Spearman: {best_test_spearman}"
            )

            extrapolated_scores = np.array(
                [saved_scores[str(i)] for i in range(num_samples)]
            )
            full_scores = np.array(
                [full_scores_dict[str(i)] for i in range(num_samples)]
            )
            corr_extrapolated = np.corrcoef(full_scores, extrapolated_scores)[0, 1]
            spearman_extrapolated = spearmanr(
                full_scores, extrapolated_scores
            ).correlation
            logger.info(
                f"Final Extrapolated Corr: {corr_extrapolated} Spearman: {spearman_extrapolated}"
            )

            filename = f"{cfg.output.gnn_dict_path}_{cfg.scores.type}_{cfg.dataset.name}_{model_name}_k_{k}_seed_{num_seed}_euclidean"

            date = datetime.datetime.now()
            filename += f"_{date.month}_{date.day}"

            with open(
                f"{cfg.output.gnn_dict_path}_{cfg.scores.type}_{cfg.dataset.name}_{model_name}_k_{k}_seed_{num_seed}_euclidean.json",
                "w",
            ) as f:
                json.dump(saved_scores, f)

            logger.info(f"Saved extrapolated scores to {filename}")


if __name__ == "__main__":
    default_config_path = os.path.join(
        os.path.dirname(__file__), "configs", "gnn_config_unsupervised.yaml"
    )
    config_path = parse_config(
        default_config=default_config_path, description="Run GNN Extrapolation"
    )
    main(cfg_path=config_path)
