import sys
import time

import numpy as np
import torch
from loguru import logger

from .pytorch_utils import matrix_to_torch

logger.remove()
logger.add(sys.stdout, format="{time:MM-DD HH:mm} - {message}")


def get_local_scores(model, attr_matrix, batch_size=10000):
    device = next(model.parameters()).device
    nnodes = attr_matrix.shape[0]
    all_scores = []

    with torch.set_grad_enabled(False):
        for start_idx in range(0, nnodes, batch_size):
            batch_attr = matrix_to_torch(
                attr_matrix[start_idx : start_idx + batch_size]
            ).to(device)
            scores = model(batch_attr)
            all_scores.append(scores.cpu().numpy())

    all_scores = np.row_stack(all_scores)
    return all_scores


def predict(
    model,
    adj_matrix,
    attr_matrix,
    alpha,
    nprop=2,
    inf_fraction=1.0,
    ppr_normalization="sym",
    batch_size_scores=10000,
):
    model.eval()

    start = time.time()
    if inf_fraction < 1.0:
        logger.info(f"inf_fraction < 1.0 is chosen: {inf_fraction}")
        idx_sub = np.random.choice(
            adj_matrix.shape[0], int(inf_fraction * adj_matrix.shape[0]), replace=False
        )
        idx_sub.sort()
        attr_sub = attr_matrix[idx_sub]

        scores_sub = get_local_scores(model.mlp, attr_sub, batch_size_scores)

        local_scores = np.zeros(
            [adj_matrix.shape[0], scores_sub.shape[1]], dtype=np.float32
        )
        local_scores[idx_sub] = scores_sub
    else:
        logger.info(f"inf_fraction >= 1.0 is chosen: {inf_fraction}")
        local_scores = get_local_scores(model.mlp, attr_matrix, batch_size_scores)

    time_local = time.time() - start

    start = time.time()
    scores = local_scores.copy()
    logger.info(f"Local scores type: {type(local_scores)}")
    logger.info(f"Local scores shape: {local_scores.shape}")
    logger.info(f"First 10 local scores: {local_scores[:10]}")

    if ppr_normalization == "sym":
        deg = adj_matrix.sum(1).A1
        deg_sqrt_inv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
        for _ in range(nprop):
            scores = (1 - alpha) * deg_sqrt_inv[:, None] * (
                adj_matrix @ (deg_sqrt_inv[:, None] * scores)
            ) + alpha * local_scores

    elif ppr_normalization == "col":
        deg_col = adj_matrix.sum(0).A1
        deg_col_inv = 1.0 / np.maximum(deg_col, 1e-12)
        for _ in range(nprop):
            scores = (1 - alpha) * (
                adj_matrix @ (deg_col_inv[:, None] * scores)
            ) + alpha * local_scores

    elif ppr_normalization == "row":
        deg_row = adj_matrix.sum(1).A1
        deg_row_inv_alpha = (1 - alpha) / np.maximum(deg_row, 1e-12)
        for _ in range(nprop):
            scores = (
                deg_row_inv_alpha[:, None] * (adj_matrix @ scores)
                + alpha * local_scores
            )

    else:
        raise ValueError(f"Unknown PPR normalization: {ppr_normalization}")

    predictions = scores.squeeze(1)
    time_propagation = time.time() - start

    return predictions, time_local, time_propagation
