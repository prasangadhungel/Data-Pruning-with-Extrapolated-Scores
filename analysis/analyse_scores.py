import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import OmegaConf
from scipy.stats import spearmanr
from torch_geometric.nn import knn_graph
from torch_cluster.knn import knn
from tqdm import tqdm
from collections import defaultdict
import pandas as pd

sys.path.append("/ceph/hdd/shared/schmidt_schwinn_data_pruning/unsupervised-data-pruning/src")

from utils.helpers import parse_config
from utils.dataset import prepare_data
from utils.models import load_model_by_name

def get_edges_and_attributes(
    embeddings,
    k=10,
    device=torch.device("cuda"),
):
    logger.info("Computing kNN graph")
    edge_index = knn_graph(embeddings, k, loop=False)

    # Compute distances for each edge:
    src, dst = edge_index

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

    edge_index = edge_index.to(device)
    edge_attr = dist.to(device)
    return edge_index, edge_attr


def compute_knn_statistics(
    embeddings,
    labels_tensor,
    top_100_keys,
    k=10,
    device=torch.device("cuda")
):
    print("Computing kNN graph")
    edge_index = knn_graph(embeddings, k, loop=False)
    src, dst = edge_index

    print("Computing distances")
    distances = ((embeddings[src] - embeddings[dst]) ** 2).sum(dim=-1).sqrt()

    # Map node index to list of (neighbor_index, distance)
    knn_map = defaultdict(list)
    for s, d, dist in zip(src.tolist(), dst.tolist(), distances.tolist()):
        knn_map[s].append((d, dist))

    stats = {}

    for key in top_100_keys:
        key_int = int(key)
        neighbors = knn_map[key_int]

        if len(neighbors) < k:
            print(f"Warning: less than {k} neighbors for sample {key_int}")

        distances_all = torch.tensor([dist for _, dist in neighbors])
        neighbor_indices = [idx for idx, _ in neighbors]
        neighbor_labels = labels_tensor[neighbor_indices]
        key_label = labels_tensor[key_int]

        # Filter neighbors with same label
        same_label_mask = (neighbor_labels == key_label).cpu()
        distances_same_label = distances_all[same_label_mask]

        # Compute statistics
        if distances_all.numel() > 0:
            min_dist = distances_all.min().item()
            max_dist = distances_all.max().item()
            mean_dist = distances_all.mean().item()
        else:
            min_dist = -1
            max_dist = -1
            mean_dist = -1

        if distances_same_label.numel() > 0:
            min_dist_same = distances_same_label.min().item()
            max_dist_same = distances_same_label.max().item()
            mean_dist_same = distances_same_label.mean().item()
            count_same_label = distances_same_label.numel()
        else:
            min_dist_same = -1
            max_dist_same = -1
            mean_dist_same = -1
            count_same_label = 0

        stats[key_int] = {
            "min_distance": min_dist,
            "max_distance": max_dist,
            "min_distance_same_label": min_dist_same,
            "max_distance_same_label": max_dist_same,
            "mean_distance": mean_dist,
            "mean_distance_same_label": mean_dist_same,
            "same_label_count": count_same_label
        }

    return stats


def compute_softmax_probabilities(model, train_loader, num_samples, device):
    """
    Compute softmax probabilities for all samples in the dataset.
    
    Returns:
        probs_at_true_label: tensor of shape (num_samples,) with probability at true label
        max_probs: tensor of shape (num_samples,) with max probability
        predicted_labels: tensor of shape (num_samples,) with predicted labels
    """
    print("Computing softmax probabilities")
    
    probs_at_true_label = torch.zeros(num_samples, device=device)
    max_probs = torch.zeros(num_samples, device=device)
    predicted_labels = torch.zeros(num_samples, dtype=torch.int64, device=device)
    
    model.eval()
    with torch.no_grad():
        for images, labels, sample_idxs in tqdm(train_loader, desc="Computing softmax probs"):
            images = images.to(device)
            labels = labels.to(device)
            sample_idxs = sample_idxs.to(device)
            
            # Get model logits (assuming the model has a classifier or outputs logits)
            # You may need to adjust this depending on your model structure
            # If your model only outputs embeddings, you'll need to add a classifier head
            logits = model.classifier(model(images)) if hasattr(model, 'classifier') else model(images)
            
            # Compute softmax probabilities
            probs = F.softmax(logits, dim=1)
            
            # Get probability at true label for each sample
            probs_at_true_label[sample_idxs] = probs[range(len(labels)), labels]
            
            # Get max probability and predicted label
            max_prob, pred_label = probs.max(dim=1)
            max_probs[sample_idxs] = max_prob
            predicted_labels[sample_idxs] = pred_label
    
    return probs_at_true_label, max_probs, predicted_labels


def main(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)
    cfg = cfg.CIFAR10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainset, train_loader, _, num_samples = prepare_data(cfg.dataset, 1024)

    #with open("/ceph/hdd/shared/schmidt_schwinn_data_pruning/unsupervised-data-pruning/scores/prune/CIFAR10_dynamic_uncertainty_0.json") as f:
    #    full_scores_dict = json.load(f)

    #with open("/ceph/hdd/shared/schmidt_schwinn_data_pruning/unsupervised-data-pruning/scores/extrapolation/extrapolated/gnn_DU_CIFAR10_resnet50-self-trained_k_10_seed_20000_euclidean.json") as f:
    #    extrapolated_scores_dict = json.load(f)

    with open("/ceph/hdd/shared/schmidt_schwinn_data_pruning/unsupervised-data-pruning/scores/prune/PLACES_365_tdds_0.json") as f:
        full_scores_dict = json.load(f)

    with open("/ceph/hdd/shared/schmidt_schwinn_data_pruning/unsupervised-data-pruning/scores/extrapolation/extrapolated/gnn_tdds_PLACES_365_resnet50-self-trained_k_10_seed_450000_euclidean.json") as f:
        extrapolated_scores_dict = json.load(f)

    labels_tensor = torch.zeros(num_samples, dtype=torch.int64, device=device)

    embedding_model = load_model_by_name("resnet18-self-trained", 10, 32,
        "/ceph/hdd/shared/schmidt_schwinn_data_pruning/unsupervised-data-pruning/models/CIFAR10/full_data/dual_model.pth", device,
    )
    embedding_model.eval()

    sample_input, _, _ = trainset[0]  # first sample, ignore label & idx
    sample_input = sample_input.unsqueeze(0).to(device)
    with torch.no_grad():
        sample_output = embedding_model(sample_input)
    embedding_dim = sample_output.shape[1]

    embeddings = torch.zeros(num_samples, embedding_dim, device=device)

    # Compute embeddings and labels
    for images, labels, sample_idxs in tqdm(
        train_loader, mininterval=20, maxinterval=40, desc="Computing embeddings"
    ):
        images = images.to(device)
        sample_idxs = sample_idxs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            batch_embeddings = embedding_model(images)

        embeddings[sample_idxs] = batch_embeddings
        labels_tensor[sample_idxs] = labels

    # Compute softmax probabilities
    probs_at_true_label, max_probs, predicted_labels = compute_softmax_probabilities(
        embedding_model, train_loader, num_samples, device
    )

    full_scores_dict={int(k): v for k, v in full_scores_dict.items()}
    extrapolated_scores_dict={int(k): v for k, v in extrapolated_scores_dict.items()}
    full_scores_dict_sorted = dict(sorted(full_scores_dict.items(), key=lambda item: item[1]))
    extrapolated_scores_dict_sorted = dict(
        sorted(extrapolated_scores_dict.items(), key=lambda item: item[1])
    )
    full_scores_ranks = {
        int(k): i for i, k in enumerate(full_scores_dict_sorted.keys())
    }
    extrapolated_scores_ranks = {
        int(k): i for i, k in enumerate(extrapolated_scores_dict_sorted.keys())
    }

    score_diff = {
        int(k): abs(full_scores_dict[k] - extrapolated_scores_dict[k])      for k in full_scores_dict.keys()
    }

    rank_diff = {
        int(k): abs(full_scores_ranks[k] - extrapolated_scores_ranks[k])
        for k in full_scores_dict.keys()
    }
    rank_diff_sorted = dict(sorted(rank_diff.items(), key=lambda item: item[1], reverse=True))
    score_diff_sorted = dict(sorted(score_diff.items(), key=lambda item: item[1], reverse=True))
    score_diff_sorted = dict(sorted(score_diff.items(), key=lambda item: item[1], reverse=True))
    top_100_keys = list(score_diff_sorted.keys())[:1000]
    top_100_keys = set(top_100_keys)

    top_100_stats = compute_knn_statistics(
        embeddings,
        labels_tensor,
        top_100_keys,
        k=20,
        device=device,
    )
    
    print(f"Type of first key in rank_diff: {type(list(rank_diff.keys())[0])}")
    print(f"Type of first key in top_100_stats: {type(list(top_100_stats.keys())[0])}")
    print(f"Number of items in top_100_stats: {len(top_100_stats)}")
    
    print(f"Sample keys in rank_diff: {list(rank_diff.keys())[:5]}")
    print(f"Sample keys in top_100_stats: {list(top_100_stats.keys())[:5]}")

    data = []

    for idx in top_100_stats:
        idx_int = int(idx)
        try:
            row = {
                "true_index": idx_int,
                "rank_difference": rank_diff[idx],
                "true_label": labels_tensor[idx_int].item(),
                "predicted_label": predicted_labels[idx_int].item(),
                "prob_at_true_label": probs_at_true_label[idx_int].item(),
                "max_softmax_prob": max_probs[idx_int].item(),
                "is_correct": (labels_tensor[idx_int] == predicted_labels[idx_int]).item(),
                "min_distance": top_100_stats[idx]["min_distance"],
                "max_distance": top_100_stats[idx]["max_distance"],
                "min_distance_same_label": top_100_stats[idx]["min_distance_same_label"],
                "max_distance_same_label": top_100_stats[idx]["max_distance_same_label"],
                "mean_distance": top_100_stats[idx]["mean_distance"],
                "mean_distance_same_label": top_100_stats[idx]["mean_distance_same_label"],
                "same_label_count": top_100_stats[idx]["same_label_count"],
                "score_diff": score_diff[idx],
                "full_score": full_scores_dict[idx],
                "extrapolated_score": extrapolated_scores_dict[idx]
            }
            data.append(row)

        except Exception as e:
            logger.exception(f"Error with idx: {idx}: {repr(e)}")

    print(f"Number of rows appended: {len(data)}")
    df = pd.DataFrame(data)
    
    # Sort by rank difference for better readability
    df = df.sort_values("rank_difference", ascending=False).reset_index(drop=True)
    
    # Save to CSV
    output_path = "/ceph/hdd/shared/schmidt_schwinn_data_pruning/unsupervised-data-pruning/analysis/top_100_extrapolation_analysis_places.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Analysis saved to {output_path}")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Average prob at true label: {df['prob_at_true_label'].mean():.4f}")
    print(f"Average max softmax prob: {df['max_softmax_prob'].mean():.4f}")
    print(f"Accuracy: {df['is_correct'].mean():.4f}")
    print(f"Average same label count: {df['same_label_count'].mean():.2f}")
    
    return df


if __name__ == "__main__":
    default_config_path = os.path.join(
        "/ceph/hdd/shared/schmidt_schwinn_data_pruning/unsupervised-data-pruning/src/extrapolate/configs/gnn_config.yaml"
    )
    config_path = parse_config(
        default_config=default_config_path, description="Run GNN Extrapolation"
    )
    df = main(cfg_path=config_path)