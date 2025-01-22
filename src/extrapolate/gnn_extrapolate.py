import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import OmegaConf
from scipy.stats import spearmanr
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, knn_graph
from tqdm import tqdm

import wandb
from prune.utils.argparse import parse_config
from prune.utils.dataset import get_dataset
from prune.utils.models import load_model_by_name

logger.remove()
logger.add(sys.stdout, format="{time:MM-DD HH:mm} - {message}")


class GNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, 512)
        self.conv2 = GCNConv(512, 256)
        self.conv3 = GCNConv(256, output_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
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
    x = torch.tensor(embeddings, dtype=torch.float, device=device)

    if distance == "cosine":
        x = F.normalize(x, p=2, dim=1)

    # knn_graph will return edge_index with shape [2, E]
    if read_knn:
        edge_index = torch.load(knn_file, map_location=device)
        logger.info(f"Loaded edge_index from {knn_file}")

    else:
        logger.info("Computing kNN graph")
        edge_index = knn_graph(x, k, loop=False)
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
        dist = (x[src] - x[dst]).pow(2).sum(dim=-1).sqrt()
        edge_attr = torch.exp(-dist)
        if save_edge_attr:
            torch.save(edge_attr, edge_attr_file)
            logger.info(f"Saved edge_attr to {edge_attr_file}")

    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    logger.info("Finished computing edge attributes")
    return edge_index, edge_attr


def prepare_data(
    embeddings_dict,
    labels_dict,
    num_classes,
    seed_samples,
    full_scores_dict,
    k,
    read_knn=False,
    save_knn=True,
    knn_file=None,
    read_edge_attr=False,
    save_edge_attr=True,
    edge_attr_file=None,
    distance="euclidean",
    device="cuda",
):
    samples_list = [int(i) for i in embeddings_dict.keys()]

    seed_samples_pos = [samples_list.index(i) for i in seed_samples]

    # Move embeddings to numpy first, then tensor
    embeddings = np.array(
        [embeddings_dict[i].cpu().numpy().flatten() for i in samples_list]
    )

    labels = [labels_dict[i] for i in samples_list]
    one_hot_labels = np.eye(num_classes)[labels]

    y = [full_scores_dict[str(i)] for i in samples_list]
    y = torch.tensor(y, dtype=torch.float)

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

    # x = torch.tensor(embeddings, dtype=torch.float, device=device)
    x = torch.tensor(one_hot_labels, dtype=torch.float, device=device)
    x = torch.cat(
        (x, torch.tensor(embeddings, dtype=torch.float, device=device)), dim=1
    )

    mask = torch.zeros(y.size(0), dtype=torch.bool)
    mask[seed_samples_pos] = True

    # Move label to device as well
    y = y.to(device)
    logger.info("Finished preparing data")
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, train_mask=mask)


def main(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainset, _ = get_dataset(
        cfg.dataset.name,
        partial=cfg.dataset.partial,
        subset_idxs=cfg.dataset.subset_idxs,
    )

    with open(cfg.dataset.original_scores_file) as f:
        full_scores_dict = json.load(f)

    with open(cfg.dataset.subset_scores_file) as f:
        subset_scores_dict = json.load(f)

    seed_samples = [int(key) for key in subset_scores_dict.keys()]
    results = []

    for model_name in tqdm(cfg.models.names):
        if cfg.checkpoints.read_embeddings:
            embeddings_dict = torch.load(
                cfg.checkpoints.embeddings_file, map_location=device
            )
            logger.info(f"Loaded embeddings from {cfg.checkpoints.embeddings_file}")

            labels_dict = {}
            for i in tqdm(range(len(trainset)), mininterval=10, maxinterval=20):
                _, label, sample_idx = trainset[i]
                labels_dict[sample_idx] = label

        else:
            embedding_model = load_model_by_name(
                model_name,
                cfg.dataset.num_classes,
                cfg.dataset.image_size,
                cfg.models.resnet50.path,
                device,
            )
            embedding_model.eval()

            embeddings_dict = {}
            labels_dict = {}
            for i in tqdm(range(len(trainset)), mininterval=10, maxinterval=20):
                sample, label, sample_idx = trainset[i]
                sample = sample.to(device).unsqueeze(0)
                with torch.no_grad():
                    embedding_val = embedding_model(sample)
                embeddings_dict[sample_idx] = embedding_val.cpu()
                labels_dict[sample_idx] = label

            if cfg.checkpoints.save_embeddings:
                torch.save(embeddings_dict, cfg.checkpoints.embeddings_file)
                logger.info(f"Saved embeddings to {cfg.checkpoints.embeddings_file}")

        for k in tqdm(cfg.hyperparams.k_values):
            wandb.init(
                project=cfg.wandb.project,
                name=f"k-{k}-model-{model_name}",
            )
            wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
            num_seed = len(seed_samples)

            data = prepare_data(
                embeddings_dict,
                labels_dict,
                cfg.dataset.num_classes,
                seed_samples,
                full_scores_dict,
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

            model = GNN(input_dim=data.num_node_features, output_dim=1).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            # Evaluate before training
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index, data.edge_attr).squeeze()

            # Original (true) scores for seed nodes
            orig_seed = data.y[data.train_mask].cpu().numpy()
            pred_seed = out[data.train_mask].cpu().numpy()
            corr_seed = np.corrcoef(orig_seed, pred_seed)[0, 1]
            spearman_seed = spearmanr(orig_seed, pred_seed).correlation
            logger.info(
                f"[Before Training] Seeds: Corr={corr_seed}, Spearman={spearman_seed}"
            )

            # Original (true) scores for unseeded
            orig_unseed = data.y[~data.train_mask].cpu().numpy()
            pred_unseed = out[~data.train_mask].cpu().numpy()
            corr_unseed = np.corrcoef(orig_unseed, pred_unseed)[0, 1]
            spearman_unseed = spearmanr(orig_unseed, pred_unseed).correlation
            logger.info(
                f"[Before Training] Unseeded: Corr={corr_unseed}, Spearman={spearman_unseed}"
            )

            # Training loop
            model.train()
            for epoch in range(cfg.hyperparams.epochs):
                optimizer.zero_grad()
                out = model(data.x, data.edge_index, data.edge_attr).squeeze()
                loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                optimizer.step()

                if epoch % 100 == 0:
                    print(f"GNN Epoch: {epoch}, Loss: {loss.item()}")

                wandb.log({"Train Loss": loss.item()}, step=epoch)

                # Evaluate correlation metrics during training
                pred_seed = out[data.train_mask].detach().cpu().numpy()
                corr_seed = np.corrcoef(orig_seed, pred_seed)[0, 1]
                spearman_seed = spearmanr(orig_seed, pred_seed).correlation
                wandb.log({"Corr Seed": corr_seed * 0.5}, step=epoch)
                wandb.log({"Spearman Seed": spearman_seed * 0.5}, step=epoch)

                pred_unseed = out[~data.train_mask].detach().cpu().numpy()
                corr_unseed = np.corrcoef(orig_unseed, pred_unseed)[0, 1]
                spearman_unseed = spearmanr(orig_unseed, pred_unseed).correlation
                wandb.log({"Corr Unseed": corr_unseed * 0.5}, step=epoch)
                wandb.log({"Spearman Unseed": spearman_unseed * 0.5}, step=epoch)

            # Final evaluation
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index, data.edge_attr).squeeze()

            pred_unseed = out[~data.train_mask].detach().cpu().numpy()
            corr_unseed = np.corrcoef(orig_unseed, pred_unseed)[0, 1]
            spearman_unseed = spearmanr(orig_unseed, pred_unseed).correlation

            pred_seed = out[data.train_mask].detach().cpu().numpy()
            corr_seed = np.corrcoef(orig_seed, pred_seed)[0, 1]
            spearman_seed = spearmanr(orig_seed, pred_seed).correlation

            logger.info(
                f"[After Training] Seeds: Corr={corr_seed}, Spearman={spearman_seed}"
            )
            logger.info(
                f"[After Training] Unseeded: Corr={corr_unseed}, Spearman={spearman_unseed}"
            )
            results.append(
                {
                    "model": model_name,
                    "k": k,
                    "num_seeds": num_seed,
                    "corr_avg": corr_unseed,
                    "spearman_avg": spearman_unseed,
                }
            )
            wandb.finish()

    results_df = pd.DataFrame(results)
    results_df.to_csv(cfg.output.results_file, index=False)


if __name__ == "__main__":
    default_config_path = os.path.join(
        os.path.dirname(__file__), "configs", "gnn_config.yaml"
    )
    config_path = parse_config(
        default_config=default_config_path, description="Run GNN Extrapolation"
    )
    main(cfg_path=config_path)
