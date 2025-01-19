import json
import os
import random
import sys

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import OmegaConf
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch_geometric.nn import knn_graph
from tqdm import tqdm

sys.path.append("/nfs/homedirs/dhp/unsupervised-data-pruning/src")

import ast
import logging
import time

import numpy as np
import torch
from pprgo.pprgo import ppr, utils
from pprgo.pprgo.dataset import PPRDataset
from pprgo.pprgo.pprgo_regression import PPRGo
from pprgo.pprgo.predict_regression import predict
from pprgo.pprgo.train_regression import train

from prune.utils.argparse import parse_config
from prune.utils.dataset import get_dataset
from prune.utils.models import load_model_by_name

# Set up logging
logger = logging.getLogger()
logger.handlers = []
ch = logging.StreamHandler()
formatter = logging.Formatter(
    fmt="%(asctime)s (%(levelname)s): %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel("INFO")


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


def prepare_data_sparse(
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
    val_fraction=0.1,
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

    attr_matrix = np.concatenate((one_hot_labels, embeddings), axis=1)

    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()

    w = edge_attr.cpu().numpy()
    N = len(samples_list)
    adj_matrix = sp.csr_matrix((w, (src, dst)), shape=(N, N))

    num_seed = len(seed_samples_pos)
    val_size = int(val_fraction * num_seed)
    random.shuffle(seed_samples_pos)
    val_idx = np.array(seed_samples_pos[:val_size], dtype=int)
    train_idx = np.array(seed_samples_pos[val_size:], dtype=int)

    seed_set = set(seed_samples_pos)
    all_indices = set(range(N))
    test_idx = np.array(list(all_indices - seed_set), dtype=int)
    test_idx.sort()

    logger.info("Finished preparing data")
    return adj_matrix, attr_matrix, y.cpu().numpy(), train_idx, val_idx, test_idx


def main():
    cfg = OmegaConf.load(
        "/nfs/homedirs/dhp/unsupervised-data-pruning/src/extrapolate/configs/pprgo_config.yaml"
    )

    # with open('/nfs/homedirs/dhp/unsupervised-data-pruning/src/extrapolate/pprgo/config_demo.yaml', 'r') as c:
    #     config = yaml.safe_load(c)

    # for key, val in config.items():
    #     if type(val) is str:
    #         try:
    #             config[key] = ast.literal_eval(val)
    #         except (ValueError, SyntaxError):
    #             pass

    alpha = cfg.pprgo.alpha  # PPR teleport probability
    eps = cfg.pprgo.eps  # Stopping threshold for ACL's ApproximatePR
    topk = cfg.pprgo.topk  # Number of PPR neighbors for each node
    ppr_normalization = (
        cfg.pprgo.ppr_normalization
    )  # Adjacency matrix normalization for weighting neighbors

    hidden_size = cfg.pprgo.hidden_size  # Size of the MLP's hidden layer
    nlayers = cfg.pprgo.nlayers  # Number of MLP layers
    weight_decay = cfg.pprgo.weight_decay  # Weight decay used for training the MLP
    dropout = cfg.pprgo.dropout  # Dropout used for training

    lr = cfg.pprgo.lr  # Learning rate
    max_epochs = (
        cfg.pprgo.max_epochs
    )  # Maximum number of epochs (exact number if no early stopping)
    batch_size = cfg.pprgo.batch_size  # Batch size for training
    batch_mult_val = cfg.pprgo.batch_mult_val  # Multiplier for validation batch size

    eval_step = (
        cfg.pprgo.eval_step
    )  # Accuracy is evaluated after every this number of steps
    run_val = cfg.pprgo.run_val  # Evaluate accuracy on validation set during training

    early_stop = cfg.pprgo.early_stop  # Use early stopping
    patience = cfg.pprgo.patience  # Patience for early stopping

    nprop_inference = (
        cfg.pprgo.nprop_inference
    )  # Number of propagation steps during inference
    inf_fraction = (
        cfg.pprgo.inf_fraction
    )  # Fraction of nodes for which local predictions are computed during inference

    cfg = OmegaConf.load(
        "/nfs/homedirs/dhp/unsupervised-data-pruning/src/extrapolate/configs/gnn_config.yaml"
    )
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

    for model_name in tqdm(cfg.models.names):
        embedding_model = load_model_by_name(
            model_name,
            cfg.dataset.num_classes,
            cfg.dataset.image_size,
            cfg.models.resnet50.path,
            device,
        )
        embedding_model.eval()

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

    seed_samples = [int(key) for key in subset_scores_dict.keys()]
    k = 10
    adj_matrix, attr_matrix, y, train_idx, val_idx, test_idx = prepare_data_sparse(
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
        val_fraction=0.1,
        distance=cfg.hyperparams.distance,
        device=device,
    )
    start = time.time()
    topk_train = ppr.topk_ppr_matrix(
        adj_matrix, alpha, eps, train_idx, topk, normalization=ppr_normalization
    )
    train_set = PPRDataset(
        attr_matrix_all=attr_matrix,
        ppr_matrix=topk_train,
        indices=train_idx,
        labels_all=y,
    )
    if run_val:
        topk_val = ppr.topk_ppr_matrix(
            adj_matrix, alpha, eps, val_idx, topk, normalization=ppr_normalization
        )
        val_set = PPRDataset(
            attr_matrix_all=attr_matrix,
            ppr_matrix=topk_val,
            indices=val_idx,
            labels_all=y,
        )
    else:
        val_set = None
    time_preprocessing = time.time() - start
    logger.info(f"Runtime: {time_preprocessing:.2f}s")
    try:
        d = attr_matrix.n_columns
    except AttributeError:
        d = attr_matrix.shape[1]
    time_loading = time.time() - start
    logger.info(f"Runtime: {time_loading:.2f}s")
    start = time.time()
    model = PPRGo(d, hidden_size, nlayers, dropout)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    _, _ = train(
        model=model,
        train_set=train_set,
        val_set=val_set,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        batch_size=batch_size,
        batch_mult_val=batch_mult_val,
        eval_step=eval_step,
        early_stop=early_stop,
        patience=patience,
    )
    time_training = time.time() - start
    logger.info("Training done.")
    logger.info(f"Runtime: {time_training:.2f}s")
    start = time.time()
    predictions, _, _ = predict(
        model=model,
        adj_matrix=adj_matrix,
        attr_matrix=attr_matrix,
        alpha=alpha,
        nprop=nprop_inference,
        inf_fraction=inf_fraction,
        ppr_normalization=ppr_normalization,
    )
    time_inference = time.time() - start
    logger.info(f"Runtime: {time_inference:.2f}s")
    mse_train = mean_squared_error(y[train_idx], predictions[train_idx])
    mse_val = mean_squared_error(y[val_idx], predictions[val_idx])
    mse_test = mean_squared_error(y[test_idx], predictions[test_idx])
    mae_train = mean_absolute_error(y[train_idx], predictions[train_idx])
    mae_val = mean_absolute_error(y[val_idx], predictions[val_idx])
    mae_test = mean_absolute_error(y[test_idx], predictions[test_idx])

    logger.info(
        f"Train MSE: {mse_train:.4f}, Val MSE: {mse_val:.4f}, Test MSE: {mse_test:.4f}"
    )
    logger.info(
        f"Train MAE: {mae_train:.4f}, Val MAE: {mae_val:.4f}, Test MAE: {mae_test:.4f}"
    )

    gpu_memory = torch.cuda.max_memory_allocated()
    memory = utils.get_max_memory_bytes()
    time_total = time_preprocessing + time_training + time_inference

    logger.info(f"GPU memory used: {gpu_memory / 1024 ** 3} GB")
    logger.info(f"Total memory used: {memory / 1024 ** 3} GB")
    logger.info(f"Total runtime: {time_total:.2f}s")


if __name__ == "__main__":
    main()
