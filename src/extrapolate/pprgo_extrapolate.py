import json
import os
import random
import sys
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch_geometric.nn import knn_graph
from tqdm import tqdm

sys.path.append("/nfs/homedirs/dhp/unsupervised-data-pruning/src")

import time

import numpy as np
import torch
from loguru import logger
from pprgo import ppr, utils
from pprgo.dataset import PPRDataset
from pprgo.pprgo_regression import PPRGo
from pprgo.predict_regression import predict
from pprgo.train_regression import train

from prune.utils.argparse import parse_config
from prune.utils.dataset import get_dataset
from prune.utils.models import load_model_by_name

logger.remove()
logger.add(sys.stdout, format="{time:MM-DD HH:mm} - {message}")


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
    run_val=False,
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

    if run_val:
        num_seed = len(seed_samples_pos)
        val_size = int(val_fraction * num_seed)
        val_idx = np.random.choice(num_seed, val_size, replace=False)
        # sort indices in val_idx
        val_idx.sort()

        train_idx = np.array(
            [i for i in seed_samples_pos if i not in val_idx], dtype=int
        )
        train_idx.sort()

    else:
        train_idx = np.array(seed_samples_pos, dtype=int)
        train_idx.sort()
        val_idx = np.array([], dtype=int)

    seed_set = set(seed_samples_pos)
    all_indices = set(samples_list)
    test_idx = np.array(list(all_indices - seed_set), dtype=int)
    test_idx.sort()

    logger.info("Finished preparing data")
    return adj_matrix, attr_matrix, y.cpu().numpy(), train_idx, val_idx, test_idx


def main(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)

    alpha = cfg.pprgo.alpha  # PPR teleport probability
    eps = cfg.pprgo.eps  # Stopping threshold for ACL's ApproximatePR
    topk = cfg.pprgo.topk  # Number of PPR neighbors for each node
    ppr_normalization = cfg.pprgo.ppr_normalization
    hidden_size = cfg.pprgo.hidden_size  # Size of the MLP's hidden layer
    nlayers = cfg.pprgo.nlayers  # Number of MLP layers
    weight_decay = cfg.pprgo.weight_decay  # Weight decay used for training the MLP
    dropout = cfg.pprgo.dropout  # Dropout used for training
    lr = cfg.pprgo.lr  # Learning rate
    max_epochs = cfg.pprgo.max_epochs  # Maximum number of epochs
    batch_size = cfg.pprgo.batch_size  # Batch size for training
    batch_mult_val = cfg.pprgo.batch_mult_val  # Multiplier for validation batch size
    eval_step = cfg.pprgo.eval_step
    run_val = cfg.pprgo.run_val  # Evaluate accuracy on validation set during training
    early_stop = cfg.pprgo.early_stop  # Use early stopping
    patience = cfg.pprgo.patience  # Patience for early stopping
    nprop_inference = cfg.pprgo.nprop_inference  # propagation steps during inference
    inf_fraction = cfg.pprgo.inf_fraction
    # Fraction of nodes for which local predictions are computed during inference

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
        run_val=run_val,
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
    try:
        d = attr_matrix.n_columns
    except AttributeError:
        d = attr_matrix.shape[1]

    start = time.time()
    model = PPRGo(d, hidden_size, nlayers, dropout)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    ###################################################################################
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        sampler=torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(train_set),
            batch_size=batch_size,
            drop_last=False,
        ),
        batch_size=None,
        num_workers=0,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    step = 0
    best_loss = np.inf

    loss_hist = {"train": [], "val": []}

    accumulated_loss = 0.0
    nsamples = 0
    best_state = None
    best_epoch = 0

    predictions, _, _ = predict(
        model=model,
        adj_matrix=adj_matrix,
        attr_matrix=attr_matrix,
        alpha=alpha,
        nprop=nprop_inference,
        inf_fraction=inf_fraction,
        ppr_normalization=ppr_normalization,
    )

    true_train = y[train_idx]
    pred_train = predictions[train_idx]
    corr_train = np.corrcoef(true_train, pred_train)[0, 1]
    spearman_train = spearmanr(true_train, pred_train).correlation
    logger.info(
        f"[Before Training] Train-Corr={corr_train}, Train-Spearman={spearman_train}"
    )

    if run_val:
        true_val = y[val_idx]
        pred_val = predictions[val_idx]
        corr_val = np.corrcoef(true_val, pred_val)[0, 1]
        spearman_val = spearmanr(true_val, pred_val).correlation
        logger.info(
            f"[Before Training] Val-Corr={corr_val}, Val-Spearman={spearman_val}"
        )

    true_test = y[test_idx]
    pred_test = predictions[test_idx]
    corr_test = np.corrcoef(true_test, pred_test)[0, 1]
    spearman_test = spearmanr(true_test, pred_test).correlation
    logger.info(
        f"[Before Training] Test-Corr={corr_test}, Test-Spearman={spearman_test}"
    )

    for epoch in range(max_epochs):
        for xbs, yb in train_loader:
            xbs, yb = [xb.to(device) for xb in xbs], yb.to(device)

            model.train()
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                pred = model(*xbs)
                pred = pred.squeeze(-1)
                loss = F.mse_loss(pred, yb.float())
                loss.backward()
                optimizer.step()

            loss_value = loss.item()

            batch_size_current = yb.shape[0]
            accumulated_loss += loss_value * batch_size_current
            nsamples += batch_size_current

            step += 1
            if step % eval_step == 0:
                predictions, _, _ = predict(
                    model=model,
                    adj_matrix=adj_matrix,
                    attr_matrix=attr_matrix,
                    alpha=alpha,
                    nprop=nprop_inference,
                    inf_fraction=inf_fraction,
                    ppr_normalization=ppr_normalization,
                )
                mse_train = mean_squared_error(y[train_idx], predictions[train_idx])
                mse_test = mean_squared_error(y[test_idx], predictions[test_idx])
                mae_train = mean_absolute_error(y[train_idx], predictions[train_idx])
                mae_test = mean_absolute_error(y[test_idx], predictions[test_idx])

                if run_val:
                    mse_val = mean_squared_error(y[val_idx], predictions[val_idx])
                    mae_val = mean_absolute_error(y[val_idx], predictions[val_idx])

                    logger.info(
                        f"Train MSE: {mse_train:.4f}, Val MSE: {mse_val:.4f}, Test MSE: {mse_test:.4f}"
                    )
                    logger.info(
                        f"Train MAE: {mae_train:.4f}, Val MAE: {mae_val:.4f}, Test MAE: {mae_test:.4f}"
                    )

                else:
                    logger.info(f"Train MSE: {mse_train:.4f}, Test MSE: {mse_test:.4f}")
                    logger.info(f"Train MAE: {mae_train:.4f}, Test MAE: {mae_test:.4f}")

                true_train = y[train_idx]
                pred_train = predictions[train_idx]
                corr_train = np.corrcoef(true_train, pred_train)[0, 1]
                spearman_train = spearmanr(true_train, pred_train).correlation
                logger.info(f"Train-Corr={corr_train}, Train-Spearman={spearman_train}")

                if run_val:
                    true_val = y[val_idx]
                    pred_val = predictions[val_idx]
                    corr_val = np.corrcoef(true_val, pred_val)[0, 1]
                    spearman_val = spearmanr(true_val, pred_val).correlation
                    logger.info(f"Val-Corr={corr_val}, Val-Spearman={spearman_val}")

                true_test = y[test_idx]
                pred_test = predictions[test_idx]
                corr_test = np.corrcoef(true_test, pred_test)[0, 1]
                spearman_test = spearmanr(true_test, pred_test).correlation
                logger.info(f"Test-Corr={corr_test}, Test-Spearman={spearman_test}")

                train_loss = accumulated_loss / nsamples
                loss_hist["train"].append(train_loss)

                if val_set is not None:
                    sample_size = min(len(val_set), batch_mult_val * batch_size)
                    rnd_idx = np.random.choice(
                        len(val_set), size=sample_size, replace=False
                    )

                    xbs_val, yb_val = val_set[rnd_idx]
                    xbs_val, yb_val = [xb.to(device) for xb in xbs_val], yb_val.to(
                        device
                    )

                    # Evaluate in eval mode (no gradient)
                    model.eval()
                    with torch.no_grad():
                        val_pred = model(*xbs_val)
                        val_pred = val_pred.squeeze(-1)

                        val_loss = F.mse_loss(val_pred, yb_val.float()).item()
                    loss_hist["val"].append(val_loss)

                    logging.info(
                        f"Subsidiary: Epoch {epoch}, step {step}: "
                        f"train_loss={train_loss:.5f}, val_loss={val_loss:.5f}\n"
                    )

                    # Check if this is the best so far
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_epoch = epoch
                        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

                    # Early stopping
                    elif early_stop and epoch >= best_epoch + patience:
                        logging.info(
                            f"Early stopping at epoch {epoch}, best epoch was {best_epoch}"
                        )
                        model.load_state_dict(best_state)
                        return epoch + 1, loss_hist

                else:
                    logging.info(
                        f"Subsidiary: Epoch {epoch}, step {step}: train_loss={train_loss:.5f}\n"
                    )

        accumulated_loss = 0.0
        nsamples = 0

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
    mse_test = mean_squared_error(y[test_idx], predictions[test_idx])
    mae_train = mean_absolute_error(y[train_idx], predictions[train_idx])
    mae_test = mean_absolute_error(y[test_idx], predictions[test_idx])

    if run_val:
        mse_val = mean_squared_error(y[val_idx], predictions[val_idx])
        mae_val = mean_absolute_error(y[val_idx], predictions[val_idx])

        logger.info(
            f"Train MSE: {mse_train:.4f}, Val MSE: {mse_val:.4f}, Test MSE: {mse_test:.4f}"
        )
        logger.info(
            f"Train MAE: {mae_train:.4f}, Val MAE: {mae_val:.4f}, Test MAE: {mae_test:.4f}"
        )

    else:
        logger.info(f"Train MSE: {mse_train:.4f}, Test MSE: {mse_test:.4f}")
        logger.info(f"Train MAE: {mae_train:.4f}, Test MAE: {mae_test:.4f}")

    true_train = y[train_idx]
    pred_train = predictions[train_idx]
    corr_train = np.corrcoef(true_train, pred_train)[0, 1]
    spearman_train = spearmanr(true_train, pred_train).correlation
    logger.info(
        f"[After Training] Train-Corr={corr_train}, Train-Spearman={spearman_train}"
    )

    if run_val:
        true_val = y[val_idx]
        pred_val = predictions[val_idx]
        corr_val = np.corrcoef(true_val, pred_val)[0, 1]
        spearman_val = spearmanr(true_val, pred_val).correlation
        logger.info(
            f"[After Training] Val-Corr={corr_val}, Val-Spearman={spearman_val}"
        )

    true_test = y[test_idx]
    pred_test = predictions[test_idx]
    corr_test = np.corrcoef(true_test, pred_test)[0, 1]
    spearman_test = spearmanr(true_test, pred_test).correlation
    logger.info(
        f"[After Training] Test-Corr={corr_test}, Test-Spearman={spearman_test}"
    )

    logger.info("Train preds: ", pred_train[:20])
    logger.info("Train true: ", true_train[:20])

    logger.info("Test preds: ", pred_test[:20])
    logger.info("Test true: ", true_test[:20])

    gpu_memory = torch.cuda.max_memory_allocated()
    memory = utils.get_max_memory_bytes()
    time_total = time_preprocessing + time_training + time_inference

    logger.info(f"GPU memory used: {gpu_memory / 1024 ** 3} GB")
    logger.info(f"Total memory used: {memory / 1024 ** 3} GB")
    logger.info(f"Total runtime: {time_total:.2f}s")


if __name__ == "__main__":
    default_config_path = os.path.join(
        os.path.dirname(__file__), "configs", "pprgo_config.yaml"
    )
    config_path = parse_config(
        default_config=default_config_path, description="Run PPRGo Extrapolation"
    )
    main(cfg_path=config_path)
