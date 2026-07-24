import datetime
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

### NEW ###
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch_scatter import scatter_mean
### END NEW ###

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

    src, dst = edge_index

    if read_edge_attr:
        edge_attr = torch.load(edge_attr_file, map_location=device)
        logger.info(f"Loaded edge_attr from {edge_attr_file}")
    else:
        logger.info(f"Computing edge attributes using {distance} distance")
        chunk_size = 10000
        dist_chunks = []
        for i in range(0, len(src), chunk_size):
            chunk_src = src[i : i + chunk_size]
            chunk_dst = dst[i : i + chunk_size]
            chunk_dist = (
                (embeddings[chunk_src] - embeddings[chunk_dst])
                .pow(2)
                .sum(dim=-1)
                .sqrt()
            )
            dist_chunks.append(chunk_dist)
        dist = torch.cat(dist_chunks, dim=0)

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
        embeddings, k, distance, read_knn, save_knn, knn_file,
        read_edge_attr, save_edge_attr, edge_attr_file, device=device,
    )
    if use_labels:
        x = torch.cat((torch.eye(num_classes)[labels].to(device), embeddings), dim=1)
    else:
        x = embeddings
    val_idxs = random.sample(seed_samples, int(val_frac * len(seed_samples)))
    train_idxs = [i for i in seed_samples if i not in val_idxs]
    test_idx = [i for i in samples_list if i not in seed_samples]
    train_mask = torch.zeros(y.size(0), dtype=torch.bool); train_mask[train_idxs] = True
    val_mask = torch.zeros(y.size(0), dtype=torch.bool); val_mask[val_idxs] = True
    test_mask = torch.zeros(y.size(0), dtype=torch.bool); test_mask[test_idx] = True
    logger.info("Finished preparing data")
    return Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
    )


@torch.no_grad()
def evaluate(model, test_loader, num_nodes, device, orig_train, orig_val,
             orig_test, train_mask, val_mask, test_mask):
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
        all_preds, corr_train, spearman_train, mse_train, corr_val,
        spearman_val, mse_val, corr_test, spearman_test, mse_test,
    )


### NEW ###
@torch.no_grad()
def perform_analysis_and_generate_report(
    full_scores_dict,
    best_extrapolated_scores,
    classifier_confidence,
    embeddings,
    k_value,
    device,
    output_dir="analysis_reports"
):
    """
    Computes detailed metrics, creates a DataFrame, and generates plots for analysis.
    """
    logger.info(f"--- Starting detailed analysis for k={k_value} ---")
    os.makedirs(output_dir, exist_ok=True)
    num_samples = len(full_scores_dict)

    # 1. Calculate average distance to 5 nearest neighbors
    logger.info("Calculating average distance to 5-NN...")
    knn_5_index = knn_graph(embeddings, k=5, loop=False)
    src, dst = knn_5_index[0].to(device), knn_5_index[1].to(device)
    
    # Calculate Euclidean distances for the 5-NN graph edges
    distances = (embeddings[src] - embeddings[dst]).pow(2).sum(dim=-1).sqrt()
    
    # Use scatter_mean to average distances for each source node
    avg_dist_5nn = scatter_mean(distances, src, dim=0, dim_size=num_samples)

    # 2. Prepare data for DataFrame
    logger.info("Assembling DataFrame...")
    true_scores = np.array([full_scores_dict[str(i)] for i in range(num_samples)])
    
    analysis_data = {
        "true_score": true_scores,
        "extrapolated_score": best_extrapolated_scores.cpu().numpy(),
        "classifier_confidence": classifier_confidence.cpu().numpy(),
        "score_difference": true_scores - best_extrapolated_scores.cpu().numpy(),
        "avg_distance_5nn": avg_dist_5nn.cpu().numpy(),
    }
    df = pd.DataFrame(analysis_data)
    
    # Add rank columns
    df["rank_true_score"] = df["true_score"].rank(method="average", ascending=False)
    df["rank_extrapolated_score"] = df["extrapolated_score"].rank(method="average", ascending=False)

    # 3. Save DataFrame to CSV
    csv_filename = os.path.join(output_dir, f"analysis_k_{k_value}.csv")
    df.to_csv(csv_filename, index_label="sample_id")
    logger.info(f"Saved analysis data to {csv_filename}")

    # 4. Print statistics and correlations
    print("\n--- Analysis Report ---")
    print("\nDataFrame Head:")
    print(df.head())
    print("\nDescriptive Statistics:")
    print(df.describe())
    print("\nCorrelation Matrix:")
    corr_matrix = df.corr()
    print(corr_matrix)

    # 5. Generate and save plots
    logger.info("Generating plots...")
    
    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Correlation Matrix (k={k_value})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"correlation_heatmap_k_{k_value}.png"))
    plt.close()
    
    # Scatter plot: Score Difference vs. Avg Neighbor Distance
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="avg_distance_5nn", y="score_difference", alpha=0.5)
    plt.title(f"Score Difference vs. Avg 5-NN Distance (k={k_value})")
    plt.xlabel("Average Distance to 5 Nearest Neighbors")
    plt.ylabel("True Score - Extrapolated Score")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"scatter_diff_vs_dist_k_{k_value}.png"))
    plt.close()

    # Scatter plot: Score Difference vs. Classifier Confidence
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="classifier_confidence", y="score_difference", alpha=0.5)
    plt.title(f"Score Difference vs. Classifier Confidence (k={k_value})")
    plt.xlabel("Classifier Confidence (Prob of True Label)")
    plt.ylabel("True Score - Extrapolated Score")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"scatter_diff_vs_confidence_k_{k_value}.png"))
    plt.close()

    logger.info(f"--- Finished analysis for k={k_value}. Reports saved to '{output_dir}'. ---")
### END NEW ###


def main(cfg_path: str):
    seed_everything(42)

    cfg = OmegaConf.load(cfg_path)
    cfg = cfg.CIFAR10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainset, train_loader, _, num_samples = prepare_data(cfg.dataset, 1024)
    logger.info(f"Loaded dataset: {cfg.dataset.name}, Device: {device}")

    with open(cfg.scores.original_scores_file) as f:
        full_scores_dict = json.load(f)
    with open(cfg.scores.subset_scores_file) as f:
        subset_scores_dict = json.load(f)

    logger.info(f"Number of samples in subset: {len(subset_scores_dict)}")
    subset_scores_np = np.array([subset_scores_dict[str(i)] for i in range(len(full_scores_dict.keys())) if str(i) in subset_scores_dict])
    full_scores_np = np.array([full_scores_dict[str(i)] for i in range(len(full_scores_dict.keys())) if str(i) in subset_scores_dict])
    corr = np.corrcoef(full_scores_np, subset_scores_np)[0, 1]
    spearman = spearmanr(full_scores_np, subset_scores_np).correlation
    mse = np.mean((full_scores_np - subset_scores_np) ** 2)
    logger.info(f"Max achievable correlation: {corr} Spearman: {spearman} MSE: {mse}")

    if cfg.scores.normalize_scores:
        logger.info("Normalizing scores with min-max normalization")
        min_sub, max_sub = np.min(subset_scores_np), np.max(subset_scores_np)
        min_full, max_full = np.min(full_scores_np), np.max(full_scores_np)
        subset_scores_dict = {key: (value - min_sub) / (max_sub - min_sub) for key, value in subset_scores_dict.items()}
        full_scores_dict = {key: (value - min_full) / (max_full - min_full) for key, value in full_scores_dict.items()}
        
    labels_tensor = torch.zeros(num_samples, dtype=torch.int64, device=device)
    seed_samples = [int(key) for key in subset_scores_dict.keys()]
    unseed_samples = [i for i in range(num_samples) if i not in seed_samples]
    training_dict = subset_scores_dict.copy()
    for samples in unseed_samples:
        training_dict[str(samples)] = 0.0

    for model_name in tqdm(cfg.models.names):
        # ### MODIFIED ###: Initialize pred_mean outside the if/else block
        pred_mean = torch.zeros(num_samples, device=device)

        if cfg.checkpoints.read_embeddings:
            embeddings = torch.load(cfg.checkpoints.embeddings_file, map_location=device)
            logger.info(f"Loaded embeddings from {cfg.checkpoints.embeddings_file}")
            
            # ### MODIFIED ###: Calculate classifier confidence even if embeddings are pre-loaded
            logger.info("Calculating classifier confidence...")
            model = load_model_by_name(
                model_name, cfg.dataset.num_classes, cfg.dataset.image_size,
                cfg.models.resnet50.path, device,
            )
            model.eval()
            for images, labels, sample_idxs in tqdm(train_loader, mininterval=20, maxinterval=40):
                images, labels, sample_idxs = images.to(device), labels.to(device), sample_idxs.to(device)
                with torch.no_grad():
                    logits = model(images)
                    probs = torch.softmax(logits, dim=1)
                    batch_pred_mean = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
                pred_mean[sample_idxs] = batch_pred_mean
                labels_tensor[sample_idxs] = labels
            del model # free up memory

        else:
            model = load_model_by_name(
                model_name, cfg.dataset.num_classes, cfg.dataset.image_size,
                cfg.models.resnet50.path, device
            )
            embedding_model = ResNetEmbedding(model).to(device)
            model.eval()
            embedding_model.eval()
            sample_input, _, _ = trainset[0]
            sample_input = sample_input.unsqueeze(0).to(device)
            with torch.no_grad():
                sample_output = embedding_model(sample_input)
            embedding_dim = sample_output.shape[1]
            logger.info(f"Embedding dimension: {embedding_dim}")
            embeddings = torch.zeros(num_samples, embedding_dim, device=device)
            
            for images, labels, sample_idxs in tqdm(train_loader, mininterval=20, maxinterval=40):
                images, labels, sample_idxs = images.to(device), labels.to(device), sample_idxs.to(device)
                with torch.no_grad():
                    batch_embeddings = embedding_model(images)
                    logits = model(images)
                    probs = torch.softmax(logits, dim=1)
                    batch_pred_mean = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
                embeddings[sample_idxs] = batch_embeddings
                pred_mean[sample_idxs] = batch_pred_mean
                labels_tensor[sample_idxs] = labels
            if cfg.checkpoints.save_embeddings:
                torch.save(embeddings, cfg.checkpoints.embeddings_file)
                logger.info(f"Saved embeddings to {cfg.checkpoints.embeddings_file}")

        for k in tqdm(cfg.hyperparams.k_values):
            data = prepare_data_graph(
                embeddings, labels_tensor, cfg.dataset.num_classes, seed_samples,
                training_dict, k, use_labels=cfg.hyperparams.use_labels,
                read_knn=cfg.checkpoints.read_knn, save_knn=cfg.checkpoints.save_knn,
                knn_file=cfg.checkpoints.knn_path + "knn_k_" + str(k) + ".pth",
                read_edge_attr=cfg.checkpoints.read_edge_attr,
                save_edge_attr=cfg.checkpoints.save_edge_attr,
                edge_attr_file=cfg.checkpoints.edge_attr_path + "edge_attr_k_" + str(k) + ".pth",
                distance=cfg.hyperparams.distance, device=device
            ).to(device)
            logger.info(f"Finished preparing data for k={k}")

            train_loader_gnn = NeighborLoader(
                data, num_neighbors=[10, 10], batch_size=cfg.hyperparams.batch_size,
                shuffle=True, input_nodes=data.train_mask
            )
            all_loader = NeighborLoader(
                data, num_neighbors=[-1], batch_size=1024, shuffle=False, input_nodes=None
            )
            model = GNN(input_dim=data.num_node_features, output_dim=1).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.hyperparams.lr)
            orig_train = np.array([subset_scores_dict[str(i)] for i in range(num_samples) if data.train_mask[i]])
            orig_val = np.array([subset_scores_dict[str(i)] for i in range(num_samples) if data.val_mask[i]])
            orig_test = np.array([full_scores_dict[str(i)] for i in range(num_samples) if data.test_mask[i]])

            (all_preds, *_) = evaluate(
                model, all_loader, data.num_nodes, device, orig_train,
                orig_val, orig_test, data.train_mask, data.val_mask, data.test_mask
            )
            logger.info(f"[Before training k={k}] Test - Corr: {spearmanr(orig_test, all_preds[data.test_mask].cpu().numpy()).correlation:.4f}")

            best_val_corr, best_test_corr, best_test_spearman = -1, -1, -1
            
            # ### MODIFIED ###: Store the best predictions tensor
            best_all_preds = None

            for epoch in range(cfg.hyperparams.epochs):
                model.train()
                for batch_idx, batch_data in enumerate(train_loader_gnn):
                    batch_data = batch_data.to(device)
                    optimizer.zero_grad()
                    out = model(batch_data.x, batch_data.edge_index, batch_data.edge_attr).squeeze()
                    out_root, y_root = out[: batch_data.batch_size], batch_data.y[: batch_data.batch_size]
                    loss = F.mse_loss(out_root, y_root)
                    loss.backward()
                    optimizer.step()

                (all_preds, _, _, _, corr_val, _, _, corr_test, spearman_test, _) = evaluate(
                    model, all_loader, data.num_nodes, device, orig_train, orig_val,
                    orig_test, data.train_mask, data.val_mask, data.test_mask
                )
                logger.info(f"Epoch={epoch} k={k} Val Corr: {corr_val:.4f} Test Corr: {corr_test:.4f} Spearman: {spearman_test:.4f}")

                if corr_val > best_val_corr:
                    logger.info(f"New best val corr: {corr_val:.4f}")
                    best_val_corr = corr_val
                    best_test_corr = corr_test
                    best_test_spearman = spearman_test
                    # ### MODIFIED ###
                    best_all_preds = all_preds.clone() 

            logger.info(f"Best Test Corr for k={k}: {best_test_corr:.4f} Spearman: {best_test_spearman:.4f}")

            # ### NEW ###: Call the analysis function after training for this k is done
            if best_all_preds is not None:
                perform_analysis_and_generate_report(
                    full_scores_dict=full_scores_dict,
                    best_extrapolated_scores=best_all_preds,
                    classifier_confidence=pred_mean,
                    embeddings=embeddings,
                    k_value=k,
                    device=device
                )
            else:
                logger.warning(f"Skipping analysis for k={k} as no best model was found.")


if __name__ == "__main__":
    default_config_path = os.path.join(os.path.dirname(__file__), "configs", "analyse_config.yaml")
    config_path = parse_config(default_config=default_config_path, description="Run GNN Extrapolation")
    main(cfg_path=config_path)