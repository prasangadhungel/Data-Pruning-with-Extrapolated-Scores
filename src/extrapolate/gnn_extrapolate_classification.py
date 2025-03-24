import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import OmegaConf
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCNConv, knn_graph
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.argparse import parse_config
from utils.dataset import prepare_data
from utils.models import load_model_by_name

logger.remove()
logger.add(sys.stdout, format="{time:MM-DD HH:mm} - {message}")


class GNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.5):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, 512)
        self.conv2 = GCNConv(512, 256)
        self.conv3 = GCNConv(256, output_dim)  # Output dim is now num_classes
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index, edge_attr)
        return x.view(-1, x.size(1))

def get_edges_and_attributes(
    embeddings,
    k=10,
    device=torch.device("cuda"),
):
    edge_index = knn_graph(embeddings, k, loop=False)
    src, dst = edge_index

    logger.info(f"Length of src: {len(src)}")
    logger.info(f"Length of dst: {len(dst)}")
    chunk_size = 10000
    for i in range(0, len(src), chunk_size):
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
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    edge_attr = torch.sqrt(edge_attr)

    logger.info("Finished computing edge attributes")
    return edge_index, edge_attr


def prepare_data_graph(
    embeddings,
    labels_dict,
    num_classes,
    seed_samples,
    training_data,
    k,
    val_frac=0.1,
    device="cuda",
):
    samples_list = list(range(len(embeddings)))
    labels = [labels_dict[i] for i in samples_list]

    y = torch.tensor(training_data, dtype=torch.long, device=device)  # Changed to Long for classification

    edge_index, edge_attr = get_edges_and_attributes(
        embeddings,
        k,
        device=device,
    )

    x = torch.cat((torch.eye(num_classes)[labels].to(device), embeddings), dim=1)

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
    train_mask,
    val_mask,
    test_mask,
    all_labels,
    training_data
):
    model.eval()
    all_preds = torch.empty(num_nodes, dtype=torch.long, device=device)
    for sub_data in test_loader:
        sub_data = sub_data.to(device)
        out_sub = model(sub_data.x, sub_data.edge_index, sub_data.edge_attr)
        pred_sub = out_sub.argmax(dim=1)
        out_root = pred_sub[: sub_data.batch_size]
        node_ids = sub_data.n_id[: sub_data.batch_size]
        all_preds[node_ids] = out_root

    pred_train = all_preds[train_mask].detach().cpu().numpy()
    
    train_indices = train_mask.nonzero(as_tuple=True)[0].cpu().numpy()
    orig_train = np.array([training_data[i] for i in train_indices])

    # comute train accuracy
    acc_train = (pred_train == orig_train).mean()
    
    pred_val = all_preds[val_mask].detach().cpu().numpy()

    val_indices = val_mask.nonzero(as_tuple=True)[0].cpu().numpy()
    orig_val = np.array([training_data[i] for i in val_indices])

    acc_val = (pred_val == orig_val).mean()

    test_indices = test_mask.nonzero(as_tuple=True)[0].cpu().numpy()
    orig_test = np.array([all_labels[i] for i in test_indices])

    pred_test = all_preds[test_mask].detach().cpu().numpy()
    acc_test = (pred_test == orig_test).mean()

    return (
        all_preds,
        acc_train,
        acc_val,
        acc_test
    )


def main(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)
    cfg = cfg.CIFAR10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_bins = 10  # number of bins

    trainset, train_loader, _, num_samples = prepare_data(cfg.dataset, 1024)
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
        ]
    )

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

    bins = np.quantile(subset_scores_np, np.linspace(0, 1, num_bins + 1))
    train_labels = np.digitize(subset_scores_np, bins[1:-1], right=False)

    logger.info(f"Number of unique training labels: {len(np.unique(train_labels))}")
    # also count number of samples in each bins
    num_unique, counts = np.unique(train_labels, return_counts=True)
    logger.info(f"Number of samples in each training bin: {dict(zip(num_unique, counts))}")
    logger.info(train_labels[:10])

    all_labels = np.digitize(full_scores_np, bins[1:-1], right=False)
    logger.info(f"Number of unique test labels: {len(np.unique(all_labels))}")
    logger.info(f"Number of samples in each test bin: {dict(zip(*np.unique(all_labels, return_counts=True)))}")
    logger.info(all_labels[:10])

    labels_tensor = torch.zeros(num_samples, dtype=torch.int64, device=device)
    seed_samples = [int(key) for key in subset_scores_dict.keys()]
    unseed_samples = [i for i in range(num_samples) if i not in seed_samples]

    training_dict = subset_scores_dict.copy()
    for samples in unseed_samples:
        training_dict[str(samples)] = 0.0

    training_dict_np = np.array(
        [
            training_dict[str(i)]
            for i in range(len(full_scores_dict.keys()))
            if str(i) in training_dict
        ]
    )
    training_data = np.digitize(
        training_dict_np, bins[1:-1], right=False
    )

    for model_name in tqdm(cfg.models.names):
        embedding_model = load_model_by_name(
            model_name,
            cfg.dataset.num_classes,
            cfg.dataset.image_size,
            cfg.models.resnet50.path,
            device,
        )
        embedding_model.eval()

        sample_input, _, _ = trainset[0]  # first sample, ignore label & idx
        sample_input = sample_input.unsqueeze(0).to(device)
        with torch.no_grad():
            sample_output = embedding_model(sample_input)
        embedding_dim = sample_output.shape[1]

        embeddings = torch.zeros(num_samples, embedding_dim, device=device)

        for images, labels, sample_idxs in tqdm(
            train_loader, mininterval=20, maxinterval=40
        ):
            images = images.to(device)
            sample_idxs = sample_idxs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                batch_embeddings = embedding_model(images)

            embeddings[sample_idxs] = batch_embeddings
            labels_tensor[sample_idxs] = labels

        for k in tqdm(cfg.hyperparams.k_values):
            data = prepare_data_graph(
                embeddings,
                labels_tensor,
                cfg.dataset.num_classes,
                seed_samples,
                training_data,
                k,
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

            model = GNN(
                input_dim=data.num_node_features, output_dim=num_bins
            ).to(
                device
            )  # Output dim = num_bins
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

            (
                all_preds,
                acc_train,
                acc_val,
                acc_test
            ) = evaluate(
                model,
                all_loader,
                data.num_nodes,
                device,
                data.train_mask,
                data.val_mask,
                data.test_mask,
                all_labels.copy(),
                training_data.copy()
            )

            logger.info(
                f"[Before training] Train - Acc: {acc_train:3f} Val - Acc: {acc_val:3f} Test - Acc: {acc_test:3f}"
            )

            best_val_acc = 0
            best_test_acc = 0


            for epoch in range(cfg.hyperparams.epochs):
                model.train()
                train_losses = []

                for batch_idx, batch_data in enumerate(train_loader):
                    batch_data = batch_data.to(device)
                    optimizer.zero_grad()
                    out = model(
                        batch_data.x, batch_data.edge_index, batch_data.edge_attr
                    )
                    out_root = out[: batch_data.batch_size]
                    y_root = batch_data.y[: batch_data.batch_size]
                    loss = torch.nn.functional.cross_entropy(out_root, y_root)
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss)

                    if batch_idx % 1000 == 0 and batch_idx > 0:
                        (
                            all_preds,
                            acc_train,
                            acc_val,
                            acc_test
                        ) = evaluate(
                            model,
                            all_loader,
                            data.num_nodes,
                            device,
                            data.train_mask,
                            data.val_mask,
                            data.test_mask,
                            all_labels.copy(),
                            training_data.copy()
                        )
                        logger.info(
                            f"Batch={batch_idx} Train: {acc_train:3f} Val: {acc_val:3f} Test: {acc_test:3f}"
                        )

                logger.info(
                    f"GNN Epoch: {epoch}, Loss: {torch.stack(train_losses).mean().item():.5f}"
                )

                (
                    all_preds,
                    acc_train,
                    acc_val,
                    acc_test
                ) = evaluate(
                    model,
                    all_loader,
                    data.num_nodes,
                    device,
                    data.train_mask,
                    data.val_mask,
                    data.test_mask,
                    all_labels.copy(),
                    training_data.copy()
                )

                logger.info(
                    f"Step={epoch} Train: {acc_train:3f} Val: {acc_val:3f} Test: {acc_test:3f}"
                )
                if acc_val > best_val_acc:
                    best_val_acc = acc_val
                    logger.info(f"Best Val Acc: {best_val_acc}")
                    best_test_acc = acc_test
                    logger.info(f"Best Test Acc: {best_test_acc}")

            # calculate correlation
            logger.info(f"Best Val Acc: {best_val_acc} Test Acc: {best_test_acc}")


if __name__ == "__main__":
    default_config_path = os.path.join(
        os.path.dirname(__file__), "configs", "gnn_config.yaml"
    )
    config_path = parse_config(
        default_config=default_config_path, description="Run GNN Extrapolation"
    )
    main(cfg_path=config_path)