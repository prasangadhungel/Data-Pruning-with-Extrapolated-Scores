import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import OmegaConf
from scipy.stats import spearmanr
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCNConv, knn_graph
from tqdm import tqdm

sys.path.append("/nfs/homedirs/dhp/unsupervised-data-pruning/src")

import wandb
from prune.utils.argparse import parse_config
from prune.utils.dataset import prepare_data
from prune.utils.models import load_model_by_name

logger.remove()
logger.add(sys.stdout, format="{time:MM-DD HH:mm} - {message}")


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
        chunk_size = 20000
        for i in range(0, len(src), chunk_size):
            if i % 20 == 0:
                logger.info(f"Processing chunk {i} to {i + chunk_size}")

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
    labels_dict,
    num_classes,
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
    labels = [labels_dict[i] for i in samples_list]

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

    pred_val = all_preds[val_mask].detach().cpu().numpy()
    corr_val = np.corrcoef(orig_val, pred_val)[0, 1]
    spearman_val = spearmanr(orig_val, pred_val).correlation

    pred_test = all_preds[test_mask].detach().cpu().numpy()
    corr_test = np.corrcoef(orig_test, pred_test)[0, 1]
    spearman_test = spearmanr(orig_test, pred_test).correlation

    return (
        all_preds,
        corr_train,
        spearman_train,
        corr_val,
        spearman_val,
        corr_test,
        spearman_test,
    )


def main(cfg_path: str):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    cfg = OmegaConf.load(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainset, train_loader, _, num_samples = prepare_data(cfg.dataset, 1024)

    with open(cfg.dataset.original_scores_file) as f:
        full_scores_dict = json.load(f)

    with open(cfg.dataset.subset_scores_file) as f:
        subset_scores_dict = json.load(f)

    labels_tensor = torch.zeros(num_samples, dtype=torch.int64, device=device)
    seed_samples = [int(key) for key in subset_scores_dict.keys()]
    num_seed = len(seed_samples)
    unseed_samples = [i for i in range(num_samples) if i not in seed_samples]

    # full scores dict are just used for evaluation
    # we won't have access to them during training
    # but we will have access to subset scores dict
    training_dict = full_scores_dict.copy()
    training_dict.update(subset_scores_dict)

    for model_name in tqdm(cfg.models.names):
        if cfg.checkpoints.read_embeddings:
            embeddings = torch.load(
                cfg.checkpoints.embeddings_file, map_location=device
            )
            logger.info(f"Loaded embeddings from {cfg.checkpoints.embeddings_file}")

            for _, labels, sample_idxs in tqdm(
                train_loader, mininterval=20, maxinterval=40
            ):
                labels = labels.to(device)
                sample_idxs = sample_idxs.to(device)
                labels_tensor[sample_idxs] = labels

        else:
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

            if cfg.checkpoints.save_embeddings:
                torch.save(embeddings, cfg.checkpoints.embeddings_file)
                logger.info(f"Saved embeddings to {cfg.checkpoints.embeddings_file}")

        for k in tqdm(cfg.hyperparams.k_values):
            wandb.init(
                project=cfg.wandb.project,
                name=f"k-{k}-model-{model_name}",
            )
            wandb.config.update(OmegaConf.to_container(cfg, resolve=True))

            data = prepare_data_graph(
                embeddings,
                labels_tensor,
                cfg.dataset.num_classes,
                seed_samples,
                training_dict,
                k,
                read_knn=cfg.checkpoints.read_knn,
                save_knn=cfg.checkpoints.save_knn,
                knn_file=cfg.checkpoints.knn_file,
                read_edge_attr=cfg.checkpoints.read_edge_attr,
                save_edge_attr=cfg.checkpoints.save_edge_attr,
                edge_attr_file=cfg.checkpoints.edge_attr_file,
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
                batch_size=128,  # adjust based on your GPU memory
                shuffle=True,
                input_nodes=data.train_mask,  # root nodes = training nodes
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
                lr=0.001,
            )

            orig_train = np.array(
                [
                    full_scores_dict[str(i)]
                    for i in range(num_samples)
                    if data.train_mask[i]
                ]
            )
            orig_val = np.array(
                [
                    full_scores_dict[str(i)]
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
                corr_val,
                spearman_val,
                corr_test,
                spearman_test,
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
                f"[Before training] Corr train: {corr_train} Spearman train: {spearman_train}"
            )
            logger.info(
                f"[Before training] Corr val: {corr_val} Spearman val: {spearman_val}"
            )
            logger.info(
                f"[Before training] Corr test: {corr_test} Spearman test: {spearman_test}"
            )
            wandb.log(
                {
                    "Train Correlation": corr_train,
                    "Train Spearman": spearman_train,
                    "Val Correlation": corr_val,
                    "Val Spearman": spearman_val,
                    "Test Correlation": corr_test,
                    "Test Spearman": spearman_test,
                },
                step=0,
            )

            # Training loop
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
                            corr_val,
                            spearman_val,
                            corr_test,
                            spearman_test,
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
                    corr_val,
                    spearman_val,
                    corr_test,
                    spearman_test,
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
                    f"Step={epoch} Corr train: {corr_train} Spearman train: {spearman_train}"
                )
                logger.info(
                    f"Step={epoch} Corr val: {corr_val} Spearman val: {spearman_val}"
                )
                logger.info(
                    f"Step={epoch} Corr test: {corr_test} Spearman test: {spearman_test}"
                )
                wandb.log(
                    {
                        "Train Correlation": corr_train,
                        "Train Spearman": spearman_train,
                        "Val Correlation": corr_val,
                        "Val Spearman": spearman_val,
                        "Test Correlation": corr_test,
                        "Test Spearman": spearman_test,
                    },
                    step=epoch + 1,
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

            # calculate correlation between saved scores and full scores

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

            with open(
                f"{cfg.output.gnn_dict_path}_{cfg.dataset.name}_{model_name}_k_{k}_seed_{num_seed}_euclidean.json",
                "w",
            ) as f:
                json.dump(saved_scores, f)


if __name__ == "__main__":
    default_config_path = os.path.join(
        os.path.dirname(__file__), "configs", "gnn_config.yaml"
    )
    config_path = parse_config(
        default_config=default_config_path, description="Run GNN Extrapolation"
    )
    main(cfg_path=config_path)
