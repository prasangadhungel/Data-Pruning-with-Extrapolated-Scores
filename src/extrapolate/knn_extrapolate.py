import datetime
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import OmegaConf
from scipy.stats import spearmanr
from torch_cluster.knn import knn
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.helpers import parse_config
from utils.dataset import prepare_data
from utils.models import load_model_by_name

logger.remove()
logger.add(sys.stdout, format="{time:MM-DD HH:mm} - {message}")


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
        y=unseed_embeddings,  # target, for which we want to find the neighbors
        k=k,
        cosine=use_cosine,
    )

    neighbor_scores = seed_scores_we_have[seed_idx]  # shape: [k * U]

    U = len(unseeded_samples)
    sum_unweighted = torch.zeros(U, dtype=torch.float, device=device)
    sum_weighted = torch.zeros(U, dtype=torch.float, device=device)
    sum_weights = torch.zeros(U, dtype=torch.float, device=device)

    # one caveat of using knn from torch_cluster is that it returns only
    # the indices of the neighbors, not the distances. Average knn doesn't
    # require the distance, but weighted knn does. So we need to compute 
    # the distances ourselves. This would increase the computation time
    # but with better implementation, we can save some computation time

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
    cfg = cfg.IMAGENET

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


    for model_name in tqdm(cfg.models.names):
        if cfg.checkpoints.read_embeddings:
            logger.info("Reading embeddings")
            embeddings = torch.load(
                cfg.checkpoints.embeddings_file, map_location=device
            )
            logger.info(f"Loaded embeddings from {cfg.checkpoints.embeddings_file}")

        else:
            logger.info("Computing embeddings")

            trainset, train_loader, _, num_samples = prepare_data(cfg.dataset, 1024)
            embedding_model = load_model_by_name(
                model_name,
                cfg.dataset.num_classes,
                cfg.dataset.image_size,
                cfg.models.resnet50.path,
                device,
            )
            embedding_model.eval()

            sample_input, _, _ = trainset[0]
            sample_input = sample_input.unsqueeze(0).to(device)
            with torch.no_grad():
                sample_output = embedding_model(sample_input)
            embedding_dim = sample_output.shape[1]

            embeddings = torch.zeros(num_samples, embedding_dim, device=device)

            for images, _, sample_idxs in tqdm(
                train_loader, mininterval=20, maxinterval=40
            ):
                images = images.to(device)
                sample_idxs = sample_idxs.to(device)
                with torch.no_grad():
                    batch_embeddings = embedding_model(images)

                embeddings[sample_idxs] = batch_embeddings

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
        os.path.dirname(__file__), "configs", "knn_config.yaml"
    )
    config_path = parse_config(
        default_config=default_config_path, description="Run KNN Extrapolation"
    )
    main(cfg_path=config_path)
