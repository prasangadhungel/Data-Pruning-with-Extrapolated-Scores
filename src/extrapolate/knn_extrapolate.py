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

from utils.argparse import parse_config
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

    knn_dict_created = {}
    for i, sample_id in enumerate(unseeded_samples):
        knn_dict_created[str(sample_id)] = float(knn_weighted_scores_np[i])

    for sample_id in seed_samples:
        knn_dict_created[str(sample_id)] = float(full_scores_dict[str(sample_id)])

    return corr_avg, corr_weighted, spearman_avg, spearman_weighted, knn_dict_created


def main(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)
    cfg = cfg.SYNTHETIC_CIFAR100_1M

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(cfg.dataset.original_scores_file) as f:
        full_scores_dict = json.load(f)

    with open(cfg.dataset.subset_scores_file) as f:
        subset_scores_dict = json.load(f)

    results = []

    models, ks, num_seeds, distance_metrics = [], [], [], []
    corr_avgs, corr_weighteds, spearman_avgs, spearman_weighteds = [], [], [], []

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
                knn_dict,
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

            distance_metrics.append(distance)
            corr_avgs.append(corr_avg)
            corr_weighteds.append(corr_weighted)
            spearman_avgs.append(spearman_avg)
            spearman_weighteds.append(spearman_weighted)
            logger.info(
                f"k: {k}, num_seed: {num_seed}, distance_metric: {distance}, corr_avg: {corr_avg}, corr_weighted: {corr_weighted}, spearman_avg: {spearman_avg}, spearman_weighted: {spearman_weighted}",
            )

            with open(
                f"{cfg.output.knn_dict_path}_{cfg.dataset.name}_{model_name}_k_{k}_seed_{num_seed}_{distance}.json",
                "w",
            ) as f:
                json.dump(knn_dict, f)

    results = pd.DataFrame(
        {
            "model": models,
            "k": ks,
            "num_seeds": num_seeds,
            "distance_metric": distance_metrics,
            "corr_avg": corr_avgs,
            "corr_weighted": corr_weighteds,
            "spearman_avg": spearman_avgs,
            "spearman_weighted": spearman_weighteds,
        }
    )
    results.to_csv(cfg.output.results_path, index=False)


if __name__ == "__main__":
    default_config_path = os.path.join(
        os.path.dirname(__file__), "configs", "knn_config.yaml"
    )
    config_path = parse_config(
        default_config=default_config_path, description="Run KNN Extrapolation"
    )
    main(cfg_path=config_path)
