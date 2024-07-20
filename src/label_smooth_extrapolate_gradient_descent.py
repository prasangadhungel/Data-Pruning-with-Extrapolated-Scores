import json
import numpy as np
import pandas as pd
import torch
import torchvision
from scipy.stats import spearmanr
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from utils.dataset import get_dataset
from utils.models import ResNetEmbedding, load_model

def get_graph_weights(embeddings_dict, k, distance_metric):
    samples_list = list(embeddings_dict.keys())
    embeddings = np.array([embeddings_dict[sample].cpu().numpy().flatten() for sample in samples_list])
    
    if distance_metric == "cosine":
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(embeddings)
    elif distance_metric == "euclidean":
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(embeddings)
    
    distances, indices = nbrs.kneighbors(embeddings)
    graph_weights = {}

    for i, sample in enumerate(samples_list):
        neighbors = indices[i]
        weights = np.exp(-distances[i])
        graph_weights[sample] = {samples_list[neighbors[j]]: weights[j] for j in range(1, len(neighbors)) if neighbors[j] != i}
    
    return graph_weights

def smooth_labels(graph_weights, seed_scores, knn_extrapolated_score, num_iterations=10000):
    scores = {node: knn_extrapolated_score[str(node)] for node in graph_weights.keys()}
    scores.update(seed_scores)
    
    for itr in range(num_iterations):
        new_scores = {}
        for node in graph_weights:
            if node in seed_scores:
                new_scores[node] = seed_scores[node]
            else:
                neighbors = graph_weights[node].keys()
                neighbor_scores = [scores[neighbor] for neighbor in neighbors]
                weights = [graph_weights[node][neighbor] for neighbor in neighbors]
                new_score = np.average(neighbor_scores, weights=weights)

                if new_score < 0:
                    new_score = 0
                if new_score > 1:
                    new_score = 1

                new_scores[node] = new_score
        
        scores = new_scores

    return scores

def get_correlation(embeddings_dict, number_of_seeds, k, distance_metric, full_scores_dict, knn_extrapolated_score):
    samples_list = [int(i) for i in embeddings_dict.keys()]
    seed_samples = np.random.choice(samples_list, number_of_seeds, replace=False)
    unseeded_samples = [i for i in samples_list if i not in seed_samples]
    
    seed_scores = {str(sample): full_scores_dict[str(sample)] for sample in seed_samples}
    
    graph_weights = get_graph_weights(embeddings_dict, k, distance_metric)
    page_rank_result = smooth_labels(graph_weights, seed_scores, knn_extrapolated_score)
    
    unseeded_scores = {sample: page_rank_result[sample] for sample in unseeded_samples}
    
    orig_list = [full_scores_dict[str(k)] for k in unseeded_samples]
    page_rank_list = [unseeded_scores[k] for k in unseeded_samples]
    corr = np.corrcoef(orig_list, page_rank_list)
    corr = corr[0, 1]
    spearman_corr = spearmanr(orig_list, page_rank_list).correlation
    
    result_dict = {**seed_scores, **unseeded_scores}
    
    return corr, spearman_corr, result_dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainset, _ = get_dataset("SYNTHETIC_CIFAR100_1M", partial=True, subset_idxs=["train"])

with open(
    "/nfs/homedirs/dhp/unsupervised-data-pruning/scores/SYNTHETIC_CIFAR100_1M_dynamic_uncertainty_train_2.json"
) as f:
    full_scores_dict = json.load(f)

with open("/nfs/homedirs/dhp/unsupervised-data-pruning/scores/SYNTHETIC_CIFAR100_1M_dynamic_uncertainty_knn_extrapolated.json") as f:
    knn_extrapolated_score = json.load(f)

model_combination = ["resnet50-trained-to-compute-score", "resnet18", "resnet50"]
k_combination = [5, 20, 50]
num_seeds_combination = [100000, 10000,  20000, 50000]

models = []
ks = []
num_seeds = []
distance_metrics = []
corrs = []
spearman_corrs = []

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

    for i in tqdm(range(len(trainset)), mininterval=10, maxinterval=20):
        sample = trainset[i][0]
        sample_idx = trainset[i][2]
        sample = sample.to(device)
        sample = sample.unsqueeze(0)
        with torch.no_grad():
            embedding_val = embedding_model(sample)
        embeddings_dict[sample_idx] = embedding_val.cpu()

    for k in tqdm(k_combination):
        for num_seed in tqdm(num_seeds_combination):

                corr, spearman_corr, result_dict = get_correlation(
                    embeddings_dict, num_seed, k, "euclidean", full_scores_dict, full_scores_dict
                )

                models.append(model)
                ks.append(k)
                num_seeds.append(num_seed)
                corrs.append(corr)
                spearman_corrs.append(spearman_corr)
                print(
                    f"k: {k}, num_seed: {num_seed}, corr: {corr}, spearman_corr: {spearman_corr}"
                )

                with open(
                    f"/nfs/homedirs/dhp/unsupervised-data-pruning/data/label_smoothing_extrapolation_{model}_{k}_{num_seed}_{distance_metric}.json",
                    "w",
                ) as f:
                    json.dump(result_dict, f)

results = pd.DataFrame(
    {
        "model": models,
        "k": ks,
        "num_seeds": num_seeds,
        "corr": corrs,
        "spearman_corr": spearman_corrs,
    }
)

# save in /nfs/homedirs/dhp/unsupervised-data-pruning/data/graph_extrapolate_results.csv
results.to_csv(
    "/nfs/homedirs/dhp/unsupervised-data-pruning/data/graph_extrapolation.csv",
    index=False,
)
