import json
import numpy as np
import torch
import torchvision
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr
from utils.dataset import get_dataset
from utils.models import load_model, ResNetEmbedding


def get_correlation(
    embeddings_dict, number_of_seeds, k, distance_metric, full_scores_dict
):
    samples_list = [int(i) for i in embeddings_dict.keys()]
    seed_samples = np.random.choice(samples_list, number_of_seeds, replace=False)
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
        weights = [1 / d for d in distance_neighbors]
        weighted_avg_score = np.average(neighbors_scores, weights=weights)
        unseeded_scores_seed_avg[i] = avg_score
        unseeded_scores_seed_weighted[i] = weighted_avg_score

    orig_list = [full_scores_dict[str(k)] for k in unseeded_samples]
    avg_list = [unseeded_scores_seed_avg[k] for k in unseeded_samples]
    weighted_list = [unseeded_scores_seed_weighted[k] for k in unseeded_samples]
    corr_avg = np.corrcoef(orig_list, avg_list)
    corr_avg = corr_avg[0, 1]
    corr_weighted = np.corrcoef(orig_list, weighted_list)
    corr_weighted = corr_weighted[0, 1]
    spearman_avg = spearmanr(orig_list, avg_list).correlation
    spearman_weighted = spearmanr(orig_list, weighted_list).correlation

    return corr_avg, corr_weighted, spearman_avg, spearman_weighted


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainset, _ = get_dataset("SYNTHETIC_CIFAR100_1M", partial=True, subset_idxs=["50k"])

with open(
    "/nfs/homedirs/dhp/unsupervised-data-pruning/scores/SYNTHETIC_CIFAR100_1M_dynamic_uncertainty_train_2.json"
) as f:
    full_scores_dict = json.load(f)

# print keys of full_scores_dict
print(list(full_scores_dict.keys())[:50])

model_combination = ["resnet18", "resnet50", "resnet50-trained-to-compute-score"]
k_combination = [5, 20, 50]
num_seeds_combination = [1000, 10000, 20000, 50000, 100000]
distance_combination = ["cosine", "euclidean"]

models = []
ks = []
num_seeds = []
distance_metrics = []
corr_avgs = []
corr_weighteds = []
spearman_avgs = []
spearman_weighteds = []

for model in tqdm(model_combination):
    if model == "resnet50-trained-to-compute-score":
        model_path = "/nfs/homedirs/dhp/unsupervised-data-pruning/models/SYNTHETIC_CIFAR100_1M_ResNet50.pt"
        model_name = "ResNet50"
        num_classes = 100
        model = load_model(model_name, num_classes, model_path, device)
        embedding_model = ResNetEmbedding(model).to(device)
        embedding_model.eval()
    
    elif model == "resnet18":
        embedding_model = torchvision.models.resnet18(pretrained=True)
        embedding_model = embedding_model.to(device)
        embedding_model.eval()

    elif model == "resnet50":
        embedding_model = torchvision.models.resnet50(pretrained=True)
        embedding_model = embedding_model.to(device)
        embedding_model.eval()
    
    embeddings_dict = {}

    for i in tqdm(range(len(trainset))):
        sample = trainset[i][0]
        sample_idx = trainset[i][2]
        sample = sample.to(device)
        sample = sample.unsqueeze(0)
        embedding_val = embedding_model(sample)
        embeddings_dict[sample_idx] = embedding_val


    for k in tqdm(k_combination):
        for num_seed in tqdm(num_seeds_combination):
            for distance_metric in tqdm(distance_combination):

                corr_avg, corr_weighted, spearman_avg, spearman_weighted = (
                    get_correlation(
                        embeddings_dict, num_seed, k, distance_metric, full_scores_dict
                    )
                )

                models.append(model)
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

# save in /nfs/homedirs/dhp/unsupervised-data-pruning/data/knn_extrapolate_results.csv
results.to_csv(
    "/nfs/homedirs/dhp/unsupervised-data-pruning/data/knn_extrapolate_results_own_model.csv",
    index=False,
)