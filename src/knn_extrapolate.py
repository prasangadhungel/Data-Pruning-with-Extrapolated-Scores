import json

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from scipy.stats import spearmanr
from tqdm import tqdm

from utils.dataset import get_dataset
from utils.models import load_model_by_name


def get_correlation(
    embeddings_dict,
    number_of_seeds,
    k,
    distance_metric,
    full_scores_dict,
    original_score,
):
    samples_list = [int(i) for i in embeddings_dict.keys()]
    seed_samples = samples_list
    unseeded_samples = [i for i in samples_list if i not in seed_samples]

    unseeded_scores_seed_avg = {}
    unseeded_scores_seed_weighted = {}

    for i in unseeded_samples:
        distances = {}
        for j in seed_samples:
            if distance_metric == "cosine":
                distance = torch.nn.functional.cosine_similarity(
                    embeddings_dict[i], embeddings_dict[j], dim=1
                )
                # if cosine similarity is more distance should be less and vice versa
                distance = 1 - distance
                distances[j] = distance.item()

            elif distance_metric == "euclidean":
                distance = torch.dist(embeddings_dict[i], embeddings_dict[j], p=2)
                distances[j] = distance.item()

        keys_smallest = np.array(list(distances.keys()))[
            np.argsort(list(distances.values()))[:k]
        ]

        neighbors_scores = [full_scores_dict[str(arg)] for arg in keys_smallest]
        distance_neighbors = [distances[arg] + 1e-10 for arg in keys_smallest]
        avg_score = np.mean(neighbors_scores)
        weights = [np.exp(-d) for d in distance_neighbors]
        weighted_avg_score = np.average(neighbors_scores, weights=weights)
        unseeded_scores_seed_avg[i] = avg_score
        unseeded_scores_seed_weighted[i] = weighted_avg_score

    orig_list = [full_scores_dict[str(k)] for k in unseeded_samples]
    avg_list = [unseeded_scores_seed_avg[k] for k in unseeded_samples]
    weighted_list = [unseeded_scores_seed_weighted[k] for k in unseeded_samples]
    true_scores = [original_score[str(k)] for k in unseeded_samples]
    corr_avg = np.corrcoef(true_scores, avg_list)
    corr_avg = corr_avg[0, 1]
    corr_weighted = np.corrcoef(true_scores, weighted_list)
    corr_weighted = corr_weighted[0, 1]
    spearman_avg = spearmanr(orig_list, avg_list).correlation
    spearman_weighted = spearmanr(orig_list, weighted_list).correlation

    knn_dict = {}
    for keys in full_scores_dict:
        if int(keys) in seed_samples:
            knn_dict[keys] = full_scores_dict[keys]
        else:
            knn_dict[keys] = unseeded_scores_seed_weighted[int(keys)]

    return corr_avg, corr_weighted, spearman_avg, spearman_weighted, knn_dict


@hydra.main(config_path="configs", config_name="knn_config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainset, _ = get_dataset(
        cfg.dataset.name,
        partial=cfg.dataset.partial,
        subset_idxs=cfg.dataset.subset_idxs,
    )

    # Load full scores and original scores
    with open(cfg.scores.full_scores_part1) as f:
        full_scores_dict_1 = json.load(f)

    with open(cfg.scores.full_scores_part2) as f:
        full_scores_dict_2 = json.load(f)

    full_scores_dict = {**full_scores_dict_1, **full_scores_dict_2}

    with open(cfg.scores.original_scores) as f:
        original_score = json.load(f)

    # Loop over combinations
    models = []
    ks = []
    num_seeds = []
    distance_metrics = []
    corr_avgs = []
    corr_weighteds = []
    spearman_avgs = []
    spearman_weighteds = []

    for model_name in tqdm(cfg.models.names):
        embedding_model = load_model_by_name(model_name, device, cfg.models.model_path)
        embedding_model.eval()

        embeddings_dict = {}

        for i in tqdm(range(len(trainset)), mininterval=10, maxinterval=20):
            sample = trainset[i][0]
            sample_idx = trainset[i][2]
            sample = sample.to(device)
            sample = sample.unsqueeze(0)
            with torch.no_grad():
                embedding_val = embedding_model(sample)
            embeddings_dict[sample_idx] = embedding_val.cpu()

        for k in tqdm(cfg.knn.k_values):
            for num_seed in tqdm(cfg.knn.num_seeds):
                for distance_metric in tqdm(cfg.knn.distance_metrics):

                    (
                        corr_avg,
                        corr_weighted,
                        spearman_avg,
                        spearman_weighted,
                        knn_dict,
                    ) = get_correlation(
                        embeddings_dict,
                        num_seed,
                        k,
                        distance_metric,
                        full_scores_dict,
                        original_score,
                    )

                    models.append(model_name)
                    ks.append(k)
                    num_seeds.append(num_seed)
                    distance_metrics.append(distance_metric)
                    corr_avgs.append(corr_avg)
                    corr_weighteds.append(corr_weighted)
                    spearman_avgs.append(spearman_avg)
                    spearman_weighteds.append(spearman_weighted)
                    print(
                        f"k: {k}, num_seed: {num_seed}, distance_metric: {distance_metric}, corr_avg: {corr_avg}, corr_weighted: {corr_weighted}, spearman_avg: {spearman_avg}, spearman_weighted: {spearman_weighted}"
                    )

                    with open(
                        f"{cfg.output.knn_dict_path}_{model_name}_{k}_{num_seed}_{distance_metric}.json",
                        "w",
                    ) as f:
                        json.dump(knn_dict, f)

    # Save results
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
    main()
