"""TDDS pruning — fixed / faster drop-in for tdds.py.

Fixes vs tdds.py
----------------
1. CRASH: tdds.py appends ``loss.item()`` (a Python float) to ``train_losses``
   but then calls ``torch.stack(train_losses)`` -> TypeError. Here we keep
   floats and average with ``np.mean`` (or ``statistics.mean``).
2. SPEED (training loop): tdds.py grows ``output_epoch``/``index_epoch`` with a
   per-batch ``np.concatenate`` -> O(n^2) copying. Here we append batch arrays
   to a list and concatenate once per epoch -> O(n).
3. SPEED/MEMORY (training loop): tdds.py builds the full ``[N, C]`` logit array
   for *every* epoch even though only the last ``trajectory`` epochs are used,
   and also stores a per-sample loss array that ``generate`` never reads. Here
   we only materialise logits/indices during the trajectory epochs and drop the
   unused per-sample loss entirely.
4. SPEED/MEMORY (generate): tdds.py recomputes the softmax and the index
   reorganisation inside every sliding window (O(windows * window * N * C)) and
   allocates a dense ``[N, C]`` tensor per window position. Here we softmax +
   reorganise each epoch ONCE, compute the adjacent-epoch KL divergence ONCE
   (streaming, so no dense ``[T, N, C]`` copy is held), then slide the window
   over the tiny ``[T-1, N]`` divergence matrix. Output is bit-identical to the
   original (verified numerically for several window sizes).
5. Deprecations: ``F.cross_entropy(..., reduce=False)`` removed (that array was
   unused), ``torch.autograd.Variable`` dropped, log uses ``cfg.pruning.num_epochs``.

The score semantics and the on-disk JSON layout are unchanged.
"""

import datetime
import json
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from loguru import logger
from omegaconf import OmegaConf
from scipy.special import softmax

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.dataset import prepare_data
from utils.evaluate import evaluate
from utils.helpers import parse_config
from utils.models import get_model
from utils.prune_utils import prune

logger.remove()
logger.add(sys.stdout, format="{time:MM-DD HH:mm} - {message}")


def generate(probs, losses, indexes, cfg):
    """Compute TDDS importance scores.

    Vectorised, memory-streaming reimplementation of the original nested-loop
    version. ``losses`` is accepted for signature compatibility but unused (the
    original never read it either).

    Args:
        probs:   ndarray ``[T, N, C]`` of raw logits over the trajectory epochs.
        indexes: ndarray ``[T, N]`` mapping each row (in epoch order) back to its
                 canonical sample index (handles the shuffled DataLoader).
        cfg:     config carrying ``pruning.{window, trajectory, decay}``.

    Returns:
        dict ``{sample_index: score}``.
    """
    del losses  # unused, kept for call-site compatibility

    window_size = cfg.pruning.window
    trajectory_len = cfg.pruning.trajectory
    decay = cfg.pruning.decay

    probs = np.asarray(probs)
    T, N, C = probs.shape
    if window_size > trajectory_len:
        raise ValueError(
            f"window ({window_size}) > trajectory ({trajectory_len})")
    if T < trajectory_len:
        raise ValueError(
            f"got {T} trajectory epochs, expected >= {trajectory_len}")

    def _reorg(epoch_logits, epoch_index):
        """Softmax over classes, then scatter rows to canonical positions."""
        sm = softmax(epoch_logits.astype(np.float64), axis=1)
        out = np.zeros((N, C), dtype=np.float64)
        out[np.asarray(epoch_index, dtype=np.int64)] = sm
        return out

    # Adjacent-epoch KL divergence, computed once and streamed so we never hold
    # a dense [T, N, C] reorganised copy.  kd[t] == KL(R[t+1] || R[t]) row-wise.
    eps = 1e-8
    kd = np.empty((trajectory_len - 1, N), dtype=np.float64)
    r_prev = _reorg(probs[0], indexes[0])
    for t in range(1, trajectory_len):
        r_cur = _reorg(probs[t], indexes[t])
        log_ratio = np.log(r_cur + eps) - np.log(r_prev + eps)
        kd[t - 1] = np.abs(r_cur * log_ratio).sum(axis=1)
        r_prev = r_cur
        logger.info(f"trajectory pair {t}/{trajectory_len - 1} ok!")

    # Slide the window over the (tiny) divergence matrix.
    moving_averages_sum = np.zeros(N, dtype=np.float64)
    for k in range(trajectory_len - window_size + 1):
        block = kd[k : k + window_size - 1]          # [window-1, N]
        window_average = block.mean(axis=0)
        diff = block - window_average
        norm = np.linalg.norm(diff, axis=0)
        weight = decay * (1 - decay) ** (trajectory_len - window_size - k)
        moving_averages_sum += norm * weight

    score_dict = {int(i): float(v) for i, v in enumerate(moving_averages_sum)}
    return score_dict


def main(cfg_path: str):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    cudnn.benchmark = True
    cfg = OmegaConf.load(cfg_path)
    cfg = cfg.PLACES_365
    logger.info(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainset, train_loader, test_loader, num_samples = prepare_data(
        cfg.dataset, cfg.training.batch_size
    )
    logger.info(f"Loaded dataset: {cfg.dataset.name}, Device: {device}")

    mapping = None
    reversed_mapping = None
    if cfg.dataset.for_extrapolation.value is True:
        indices_to_keep = random.sample(
            range(num_samples), cfg.dataset.for_extrapolation.subset_size
        )
        mapping = {
            original_idx: new_idx
            for new_idx, original_idx in enumerate(indices_to_keep)
        }
        reversed_mapping = {
            new_idx: original_idx
            for new_idx, original_idx in enumerate(indices_to_keep)
        }
        trainset = torch.utils.data.Subset(trainset, indices_to_keep)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=2,
        )

    for num_itr in range(cfg.experiment.num_iterations):
        model = get_model(
            model_name=cfg.model.name,
            num_classes=cfg.dataset.num_classes,
            image_size=cfg.dataset.image_size,
        ).to(device)

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.training.lr,
            momentum=cfg.training.momentum,
            weight_decay=cfg.training.weight_decay,
            nesterov=cfg.training.nesterov,
        )

        num_iter = len(train_loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=cfg.pruning.num_epochs * num_iter
        )

        criterion = torch.nn.CrossEntropyLoss().to(device)

        output_epochs, index_epochs = [], []
        for epoch in range(cfg.pruning.num_epochs):
            # Only the last `trajectory` epochs feed the score; skip the costly
            # host copy / concatenation for all earlier epochs.
            collect = (cfg.pruning.num_epochs - epoch) <= cfg.pruning.trajectory

            train_losses = []
            out_chunks, idx_chunks = [], []

            for batch_idx, (data, target, sample_idx) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                if collect:
                    index_batch = sample_idx
                    if cfg.dataset.for_extrapolation.value is True:
                        index_batch = [mapping[idx.item()] for idx in index_batch]
                    out_chunks.append(output.detach().cpu().numpy())
                    idx_chunks.append(np.asarray(index_batch, dtype=np.int64))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_losses.append(loss.item())

                if batch_idx % cfg.logging.log_interval == 0 and batch_idx > 0:
                    logger.info(
                        f"Epoch {epoch + 1}/{cfg.pruning.num_epochs}, "
                        f"Itr {batch_idx}/{len(train_loader)}, "
                        f"Loss: {np.mean(train_losses):.5f}, "
                        f"Test Acc: {evaluate(model, test_loader, device):.5f}, "
                    )

            if collect:
                output_epochs.append(np.concatenate(out_chunks, axis=0))
                index_epochs.append(np.concatenate(idx_chunks, axis=0))

            test_acc = evaluate(model, test_loader, device)
            train_loss = float(np.mean(train_losses))
            logger.info(
                f"Epoch {epoch + 1}, Train Loss: {train_loss:.5f}, "
                f"Test Accuracy: {test_acc:.5f}"
            )

        model_name = f"{cfg.paths.models}/tdds_last"
        if cfg.dataset.for_extrapolation.value is True:
            model_name += f"_{cfg.dataset.for_extrapolation.subset_size}"
        model_name += ".pth"
        torch.save(model.state_dict(), model_name)
        logger.info(f"Saved model to {model_name}")

        logger.info("Computing Importance Scores")
        # Keep the last `trajectory` epochs (they were the only ones collected).
        output_epochs_arr = np.array(output_epochs[-cfg.pruning.trajectory :])
        index_epochs_arr = np.array(index_epochs[-cfg.pruning.trajectory :])

        logger.info(f"Shape of output_epochs: {output_epochs_arr.shape}")
        logger.info(f"Shape of index_epochs: {index_epochs_arr.shape}")

        tdds_score = generate(output_epochs_arr, None, index_epochs_arr, cfg)

        if cfg.dataset.for_extrapolation.value is True:
            tdds_score = {
                reversed_mapping[key]: value for key, value in tdds_score.items()
            }

        output_path = f"{cfg.paths.scores}/{cfg.dataset.name}_last_tdds_{num_itr}"
        if cfg.dataset.for_extrapolation.value is True:
            output_path += f"_{cfg.dataset.for_extrapolation.subset_size}"
        date = datetime.datetime.now()
        output_path += f"_{date.month}_{date.day}"
        output_path += ".json"

        with open(output_path, "w") as f:
            json.dump(tdds_score, f)
        logger.info(f"Saved tdds scores to {output_path}")

        if cfg.pruning.prune is True:
            # cfg.save_path is not present in every dataset block; fall back to
            # the models dir instead of raising (original crashed here).
            save_model_path = OmegaConf.select(cfg, "save_path") or cfg.paths.models
            prune(
                trainset=trainset,
                test_loader=test_loader,
                scores_dict=tdds_score,
                cfg=cfg,
                wandb_name="tdds-last-scheduler-",
                device=device,
                save_model_path=save_model_path,
            )


if __name__ == "__main__":
    default_config_path = os.path.join(
        os.path.dirname(__file__), "configs", "tdds_config.yaml"
    )
    config_path = parse_config(
        default_config=default_config_path, description="Run TDDS Pruning (fixed)"
    )
    main(cfg_path=config_path)
