import datetime
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from loguru import logger
from omegaconf import OmegaConf
from scipy.stats import spearmanr
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCNConv, knn_graph
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.dataset import prepare_data
from utils.helpers import parse_config, seed_everything
from utils.models import load_model_by_name, ResNetEmbedding
from utils.prune_utils import prune

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

    if read_knn and os.path.exists(knn_file):
        edge_index = torch.load(knn_file, map_location=device)
        logger.info(f"Loaded edge_index from {knn_file}")
    else:
        logger.info(f"Computing kNN graph (k={k})")
        edge_index = knn_graph(embeddings, k, loop=False)
        logger.info("Finished computing kNN graph")

        if save_knn:
            torch.save(edge_index, knn_file)
            logger.info(f"Saved edge_index to {knn_file}")

    src, dst = edge_index

    if read_edge_attr and os.path.exists(edge_attr_file):
        edge_attr = torch.load(edge_attr_file, map_location=device)
        logger.info(f"Loaded edge_attr from {edge_attr_file}")
    else:
        logger.info(f"Computing edge attributes using {distance} distance")
        
        chunk_size = 10000
        dist_list = []
        for i in range(0, len(src), chunk_size):
            chunk_src = src[i : i + chunk_size]
            chunk_dst = dst[i : i + chunk_size]
            chunk_dist = (
                (embeddings[chunk_src] - embeddings[chunk_dst])
                .pow(2)
                .sum(dim=-1)
                .sqrt()
            )
            dist_list.append(chunk_dist)
        
        dist = torch.cat(dist_list, dim=0)

        edge_attr = torch.exp(-dist)
        if save_edge_attr:
            torch.save(edge_attr, edge_attr_file)
            logger.info(f"Saved edge_attr to {edge_attr_file}")

    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    edge_attr = torch.sqrt(edge_attr)

    logger.info("Finished computing edge attributes")
    return edge_index, edge_attr


def compute_avg_knn_distance(embeddings, k=5, batch_size=1000, device="cuda"):
    """
    Computes the average distance of the k-nearest neighbors for each point (global search).
    """
    logger.info(f"Computing average distance to {k} nearest neighbors (Global)...")
    num_samples = embeddings.size(0)
    avg_dists = torch.zeros(num_samples, device=device)
    
    for i in range(0, num_samples, batch_size):
        end = min(i + batch_size, num_samples)
        batch_emb = embeddings[i:end]
        dists = torch.cdist(batch_emb, embeddings, p=2)
        vals, _ = dists.topk(k + 1, dim=1, largest=False)
        neighbor_dists = vals[:, 1:] # Exclude self
        avg_dists[i:end] = neighbor_dists.mean(dim=1)
        
    return avg_dists


def compute_same_class_stats(embeddings, labels, k=5, device="cuda"):
    """
    1. Counts number of samples per class for each sample.
    2. Finds 5 nearest neighbors *within the same class* and computes avg distance.
    """
    logger.info(f"Computing same-class neighbor statistics (k={k})...")
    num_samples = embeddings.size(0)
    
    # Prepare output tensors
    avg_dists_same_class = torch.zeros(num_samples, device=device)
    class_counts_per_sample = torch.zeros(num_samples, dtype=torch.long, device=device)
    
    # Get unique classes
    unique_classes = labels.unique()
    
    # Count totals per class first
    total_counts = torch.bincount(labels)
    class_counts_per_sample = total_counts[labels]

    # Iterate over classes to compute KNN within that class
    for c in unique_classes:
        # Boolean mask for current class
        mask = (labels == c)
        
        # Get indices and embeddings for this class
        class_indices = torch.where(mask)[0]
        class_embeddings = embeddings[mask]
        
        num_in_class = class_embeddings.size(0)
        
        # If class has fewer samples than k+1, adjust k effectively for this class
        # (We need self + k neighbors)
        curr_k = min(k, num_in_class - 1)
        
        if curr_k <= 0:
            # If singular sample in class, distance is 0 or undefined. 
            # We leave it as 0.0 initialized above.
            continue

        # Compute pairwise distances only within this class
        # Shape: [num_in_class, num_in_class]
        # This is much faster than global cdist with masking
        dists = torch.cdist(class_embeddings, class_embeddings, p=2)
        
        # Top k+1 (including self)
        vals, _ = dists.topk(curr_k + 1, dim=1, largest=False)
        
        # Exclude self (first column) and take mean
        # shape: [num_in_class]
        neighbor_dists = vals[:, 1:].mean(dim=1)
        
        # Map back to global storage
        avg_dists_same_class[class_indices] = neighbor_dists

    logger.info("Finished computing same-class neighbor statistics.")
    return class_counts_per_sample, avg_dists_same_class


def prepare_data_graph(
    embeddings,
    labels_dict,
    num_classes,
    seed_samples,
    training_dict,
    k,
    use_labels=True,
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

    if use_labels:
        x = torch.cat((torch.eye(num_classes)[labels].to(device), embeddings), dim=1)
    else:
        x = embeddings

    val_idxs = random.sample(seed_samples, int(val_frac * len(seed_samples)))
    train_idxs = [i for i in seed_samples if i not in val_idxs]
    test_idx = [i for i in samples_list if i not in seed_samples]

    train_mask = torch.zeros(y.size(0), dtype=torch.bool)
    train_mask[train_idxs] = True
    val_mask = torch.zeros(y.size(0), dtype=torch.bool)
    val_mask[val_idxs] = True
    test_mask = torch.zeros(y.size(0), dtype=torch.bool)
    test_mask[test_idx] = True

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
def evaluate(model, test_loader, num_nodes, device, orig_train, orig_val, orig_test, train_mask, val_mask, test_mask):
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
        corr_train, spearman_train, mse_train,
        corr_val, spearman_val, mse_val,
        corr_test, spearman_test, mse_test,
    )


def main(cfg_path: str):
    seed_everything(42)

    cfg = OmegaConf.load(cfg_path)
    cfg = cfg.CIFAR10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainset, train_loader, _, num_samples = prepare_data(cfg.dataset, 1024)
    logger.info(f"Loaded dataset: {cfg.dataset.name}, Device: {device}")

    # Load Scores
    with open(cfg.scores.original_scores_file) as f:
        full_scores_dict = json.load(f)
    with open(cfg.scores.subset_scores_file) as f:
        subset_scores_dict = json.load(f)

    # Align scores to arrays
    full_scores_np = np.zeros(num_samples)
    for k, v in full_scores_dict.items():
        if int(k) < num_samples:
            full_scores_np[int(k)] = v

    subset_scores_np = np.array([subset_scores_dict[str(i)] for i in range(len(full_scores_dict.keys())) if str(i) in subset_scores_dict])
    
    if cfg.scores.normalize_scores:
        min_val, max_val = np.min(full_scores_np), np.max(full_scores_np)
        full_scores_dict = {k: (v - min_val) / (max_val - min_val) for k, v in full_scores_dict.items()}
        subset_scores_dict = {k: (v - min_val) / (max_val - min_val) for k, v in subset_scores_dict.items()}
        full_scores_np = (full_scores_np - min_val) / (max_val - min_val)

    labels_tensor = torch.zeros(num_samples, dtype=torch.int64, device=device)
    seed_samples = [int(key) for key in subset_scores_dict.keys()]
    unseed_samples = [i for i in range(num_samples) if i not in seed_samples]

    training_dict = subset_scores_dict.copy()
    for samples in unseed_samples:
        training_dict[str(samples)] = 0.0

    classifier_stats_file = cfg.checkpoints.embeddings_file.replace(".pth", "_stats.pth")
    conf_true_class = torch.zeros(num_samples, device=device)
    conf_max = torch.zeros(num_samples, device=device)
    is_correct = torch.zeros(num_samples, dtype=torch.bool, device=device)
    embeddings = None

    for model_name in tqdm(cfg.models.names):
        read_emb = cfg.checkpoints.read_embeddings and os.path.exists(cfg.checkpoints.embeddings_file)
        read_stats = cfg.checkpoints.read_embeddings and os.path.exists(classifier_stats_file)

        if read_emb and read_stats:
            embeddings = torch.load(cfg.checkpoints.embeddings_file, map_location=device)
            stats = torch.load(classifier_stats_file, map_location=device)
            conf_true_class = stats['conf_true_class']
            conf_max = stats['conf_max']
            is_correct = stats['is_correct']
            labels_tensor = stats['labels']
            logger.info(f"Loaded embeddings and stats from disk.")
        else:
            logger.info("Computing embeddings and classifier stats...")
            model = load_model_by_name(
                model_name, cfg.dataset.num_classes, cfg.dataset.image_size, 
                cfg.models.resnet50.path, device
            )
            embedding_model = ResNetEmbedding(model).to(device)
            model.eval()
            embedding_model.eval()

            with torch.no_grad():
                dummy_out = embedding_model(trainset[0][0].unsqueeze(0).to(device))
                emb_dim = dummy_out.shape[1]

            embeddings = torch.zeros(num_samples, emb_dim, device=device)

            for images, labels, sample_idxs in tqdm(train_loader, mininterval=10):
                images = images.to(device)
                sample_idxs = sample_idxs.to(device)
                labels = labels.to(device)
                
                with torch.no_grad():
                    batch_embeddings = embedding_model(images)
                    logits = model(images)
                    probs = torch.softmax(logits, dim=1)
                    
                    batch_conf_true = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
                    batch_conf_max, batch_preds = torch.max(probs, dim=1)
                    batch_correct = (batch_preds == labels)

                embeddings[sample_idxs] = batch_embeddings
                conf_true_class[sample_idxs] = batch_conf_true
                conf_max[sample_idxs] = batch_conf_max
                is_correct[sample_idxs] = batch_correct
                labels_tensor[sample_idxs] = labels

            if cfg.checkpoints.save_embeddings:
                torch.save(embeddings, cfg.checkpoints.embeddings_file)
                stats_to_save = {
                    'conf_true_class': conf_true_class,
                    'conf_max': conf_max,
                    'is_correct': is_correct,
                    'labels': labels_tensor
                }
                torch.save(stats_to_save, classifier_stats_file)
                logger.info(f"Saved embeddings and stats.")

        # --- Compute Geometric Properties ---
        # 1. Global 5-NN Distance
        avg_5nn_dists = compute_avg_knn_distance(embeddings, k=5, device=device)
        
        # 2. Same-Class Stats
        count_same_class, avg_5nn_dists_same = compute_same_class_stats(embeddings, labels_tensor, k=5, device=device)

        best_all_preds_tensor = None 

        for k in tqdm(cfg.hyperparams.k_values):
            data = prepare_data_graph(
                embeddings,
                labels_tensor,
                cfg.dataset.num_classes,
                seed_samples,
                training_dict,
                k,
                use_labels=cfg.hyperparams.use_labels,
                read_knn=cfg.checkpoints.read_knn,
                save_knn=cfg.checkpoints.save_knn,
                knn_file=cfg.checkpoints.knn_path + "knn_k_" + str(k) + ".pth",
                read_edge_attr=cfg.checkpoints.read_edge_attr,
                save_edge_attr=cfg.checkpoints.save_edge_attr,
                edge_attr_file=cfg.checkpoints.edge_attr_path + "edge_attr_k_" + str(k) + ".pth",
                distance=cfg.hyperparams.distance,
                device=device,
            ).to(device)

            train_loader_gnn = NeighborLoader(data, num_neighbors=[10, 10], batch_size=cfg.hyperparams.batch_size, shuffle=True, input_nodes=data.train_mask)
            all_loader = NeighborLoader(data, num_neighbors=[-1], batch_size=1024, shuffle=False, input_nodes=None)

            model = GNN(input_dim=data.num_node_features, output_dim=1).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.hyperparams.lr)

            orig_train = np.array([subset_scores_dict[str(i)] for i in range(num_samples) if data.train_mask[i]])
            orig_val = np.array([subset_scores_dict[str(i)] for i in range(num_samples) if data.val_mask[i]])
            orig_test = np.array([full_scores_dict[str(i)] for i in range(num_samples) if data.test_mask[i]])

            best_val_corr = -1

            for epoch in range(cfg.hyperparams.epochs):
                model.train()
                for batch_data in train_loader_gnn:
                    batch_data = batch_data.to(device)
                    optimizer.zero_grad()
                    out = model(batch_data.x, batch_data.edge_index, batch_data.edge_attr).squeeze()
                    out_root = out[: batch_data.batch_size]
                    y_root = batch_data.y[: batch_data.batch_size]
                    loss = F.mse_loss(out_root, y_root)
                    loss.backward()
                    optimizer.step()

                (all_preds, _, _, _, corr_val, _, _, corr_test, _, _) = evaluate(
                    model, all_loader, data.num_nodes, device,
                    orig_train, orig_val, orig_test,
                    data.train_mask, data.val_mask, data.test_mask
                )

                if corr_val > best_val_corr:
                    best_val_corr = corr_val
                    best_all_preds_tensor = all_preds.detach().cpu()
                    
            logger.info(f"Best validation correlation: {best_val_corr}")

    # --- DataFrame Creation ---
    logger.info("Creating analysis DataFrame...")

    if best_all_preds_tensor is None:
        best_all_preds_tensor = all_preds.detach().cpu()
        
    extrapolated_scores = best_all_preds_tensor.numpy()
    true_ranks = pd.Series(full_scores_np).rank(ascending=False, method='min')
    extrapolated_ranks = pd.Series(extrapolated_scores).rank(ascending=False, method='min')
    score_diff = np.abs(full_scores_np - extrapolated_scores)

    df = pd.DataFrame({
        'sample_id': np.arange(num_samples),
        'true_score': full_scores_np,
        'extrapolated_score': extrapolated_scores,
        'true_rank': true_ranks,
        'extrapolated_rank': extrapolated_ranks,
        'accuracy': is_correct.cpu().numpy().astype(int),
        'confidence_true_class': conf_true_class.cpu().numpy(),
        'confidence_max': conf_max.cpu().numpy(),
        'score_difference': score_diff,
        'avg_dist_5nn': avg_5nn_dists.cpu().numpy(),
        'num_same_class': count_same_class.cpu().numpy(),
        'avg_dist_5nn_same_class': avg_5nn_dists_same.cpu().numpy()
    })

    output_csv = "analysis_results.csv"
    df.to_csv(output_csv, index=False)
    logger.info(f"Analysis saved to {output_csv}")

    # --- Plotting ---
    logger.info("Generating plots...")
    
    reports_dir = "analysis_reports"
    os.makedirs(reports_dir, exist_ok=True)

    # Set plot style
    sns.set(style="whitegrid")

    # 1. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    # Select numerical columns of interest
    cols_to_corr = [
        'true_score', 'extrapolated_score', 'score_difference',
        'confidence_true_class', 'confidence_max',
        'avg_dist_5nn', 'avg_dist_5nn_same_class'
    ]
    corr_matrix = df[cols_to_corr].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix of Extrapolation Properties")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "correlation_heatmap.png"))
    plt.close()

    # 2. Scatter Plot: Confidence vs Score Difference
    plt.figure(figsize=(8, 6))
    sns.regplot(x='confidence_true_class', y='score_difference', data=df, 
                scatter_kws={'alpha':0.3, 's': 10}, line_kws={'color': 'red'})
    corr_val = df['confidence_true_class'].corr(df['score_difference'])
    plt.title(f"Confidence (True Class) vs Score Diff (Corr: {corr_val:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "scatter_conf_vs_diff.png"))
    plt.close()

    # 3. Scatter Plot: Avg Neighbor Distance vs Score Difference
    plt.figure(figsize=(8, 6))
    sns.regplot(x='avg_dist_5nn', y='score_difference', data=df, 
                scatter_kws={'alpha':0.3, 's': 10}, line_kws={'color': 'red'})
    corr_val = df['avg_dist_5nn'].corr(df['score_difference'])
    plt.title(f"Avg Dist (Global 5NN) vs Score Diff (Corr: {corr_val:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "scatter_dist_vs_diff.png"))
    plt.close()

    # 4. Scatter Plot: Avg Same-Class Neighbor Distance vs Score Difference
    plt.figure(figsize=(8, 6))
    sns.regplot(x='avg_dist_5nn_same_class', y='score_difference', data=df, 
                scatter_kws={'alpha':0.3, 's': 10}, line_kws={'color': 'red'})
    corr_val = df['avg_dist_5nn_same_class'].corr(df['score_difference'])
    plt.title(f"Avg Dist (Same Class 5NN) vs Score Diff (Corr: {corr_val:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "scatter_same_class_dist_vs_diff.png"))
    plt.close()

    logger.info(f"Plots saved to directory: {reports_dir}")


if __name__ == "__main__":
    default_config_path = os.path.join(
        os.path.dirname(__file__), "configs", "gnn_config.yaml"
    )
    config_path = parse_config(
        default_config=default_config_path, description="Run GNN Extrapolation"
    )
    main(cfg_path=config_path)