import datetime
import json
import os
import random
import sys
import time

import torch
from loguru import logger
from omegaconf import OmegaConf
from torch.optim import Adam

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from collections import defaultdict

from utils.dataset import prepare_data
from utils.evaluate import evaluate
from utils.helpers import parse_config, seed_everything
from utils.models import get_model
from utils.prune_utils import prune

logger.remove()
logger.add(sys.stdout, format="{time:MM-DD HH:mm} - {message}")

def get_dual_scores(cfg, device, trainset, train_loader, test_loader):
    # Initialize model and optimizer
    model = get_model(
        model_name=cfg.model.name,
        num_classes=cfg.dataset.num_classes,
        image_size=cfg.dataset.image_size,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=0.0008)

    window_size = cfg.uncertainty.window

    # Full softmax history needed for DUAL scores
    preds_history = {sample_idx: [] for _, _, sample_idx in trainset}

    # Only true-class probability history -> list of floats
    true_prob_history = {sample_idx: [] for _, _, sample_idx in trainset}

    torch.cuda.empty_cache()
    start_time = time.time()

    # ==== Training + storing prediction trajectories ====
    for epoch in range(cfg.training.num_epochs):
        train_losses = []
        for batch_idx, (data, target, sample_idx) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)

            optimizer.zero_grad()
            train_losses.append(loss)
            loss.backward()
            optimizer.step()

            # Logging
            if batch_idx % cfg.logging.log_interval == 0 and batch_idx > 0:
                logger.info(
                    f"Epoch {epoch+1}/{cfg.training.num_epochs}, "
                    f"Itr {batch_idx}/{len(train_loader)}, "
                    f"Loss: {torch.stack(train_losses).mean().item():.5f}, "
                    f"Test Acc: {evaluate(model, test_loader, device):.5f}, "
                    f"Time: {time.time() - start_time:.5f}"
                )

            # === Full softmax output (needed for DUAL scores) ===
            softmax_output = torch.nn.functional.softmax(output, dim=1)
            softmax_output_cpu = softmax_output.detach().cpu()

            for i, idx in enumerate(sample_idx):
                # Store full vector for DUAL
                preds_history[idx.item()].append(softmax_output_cpu[i])

                # Store *only* true class prob
                true_class = target[i].item()
                prob_true = softmax_output_cpu[i, true_class].item()
                true_prob_history[idx.item()].append(prob_true)

        logger.info(
            f"Epoch {epoch+1}, Train Loss: {torch.stack(train_losses).mean().item():.5f}, "
            f"Test Acc: {evaluate(model, test_loader, device):.5f}"
        )

    # Save model
    model_name = f"{cfg.paths.models}/dual_model"
    if cfg.dataset.for_extrapolation.value:
        model_name += f"_{cfg.dataset.for_extrapolation.subset_size}"
    model_name += ".pth"
    torch.save(model.state_dict(), model_name)
    logger.info(f"Saved model to {model_name}")

    dual_scores = {}

    for sample_idx, history in preds_history.items():
        preds = torch.stack(history)  # [epochs, C]

        windows = []
        for i in range(preds.size(0) - window_size + 1):
            window = preds[i : i + window_size]
            win_std = window.std(dim=0) * 10
            win_mean = window.mean(dim=0)
            windows.append(win_std * (1 - win_mean))

        if len(windows) == 0:
            dual_scores[int(sample_idx)] = 0.0
            continue

        score = torch.stack(windows).mean(dim=0).mean().item()
        dual_scores[int(sample_idx)] = score

    num_epochs_mean = min(30, cfg.training.num_epochs)

    pred_mean = {}
    for sample_idx, probs in true_prob_history.items():
        arr = torch.tensor(probs)[:num_epochs_mean]
        pred_mean[sample_idx] = arr.mean().item()

    # Compute mu_d
    # top-10 samples by dual score
    top10 = sorted(dual_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    top10_ids = [sid for sid, _ in top10]

    # mean true-class probability across these samples (averaged over all epochs)
    values = []
    for sid in top10_ids:
        arr = torch.tensor(true_prob_history[sid])
        values.append(arr.mean().item())

    mu_d = float(sum(values) / len(values))

    return dual_scores, pred_mean, mu_d


def main(cfg_path: str):
    seed_everything(42)

    cfg = OmegaConf.load(cfg_path)
    cfg = cfg.CIFAR10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainset, train_loader, test_loader, num_samples = prepare_data(
        cfg.dataset, cfg.training.batch_size
    )
    logger.info(f"Loaded dataset: {cfg.dataset.name}, Device: {device}")

    if cfg.dataset.for_extrapolation.value is True:
        # Group indices by class
        class_indices = defaultdict(list)
        for idx in range(len(trainset)):
            # Each element is assumed to be (data, target, sample_idx)
            _, target, _ = trainset[idx]
            class_indices[target].append(idx)

        total_subset_size = cfg.dataset.for_extrapolation.subset_size
        # Compute stratified sample sizes for each class based on proportions
        indices_to_keep = []
        for cls, indices in class_indices.items():
            proportion = len(indices) / num_samples
            # Determine the number of samples to pick for the class
            n_samples = max(1, int(round(proportion * total_subset_size)))
            n_samples = min(n_samples, len(indices))  # do not exceed available indices
            selected = random.sample(indices, n_samples)
            indices_to_keep.extend(selected)

        # Adjust if total samples differ from expected subset_size
        if len(indices_to_keep) > total_subset_size:
            indices_to_keep = random.sample(indices_to_keep, total_subset_size)
        elif len(indices_to_keep) < total_subset_size:
            # Fill the remaining slots randomly from unselected indices
            all_indices = set(range(num_samples))
            remaining = list(all_indices - set(indices_to_keep))
            if len(remaining) >= (total_subset_size - len(indices_to_keep)):
                indices_to_keep.extend(
                    random.sample(remaining, total_subset_size - len(indices_to_keep))
                )
            else:
                indices_to_keep.extend(remaining)

        trainset = torch.utils.data.Subset(trainset, indices_to_keep)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=2,
        )

    for num_itr in range(cfg.experiment.num_iterations):
        dual_scores, pred_mean, mu_d = get_dual_scores(cfg, device, trainset, train_loader, test_loader)

        output_path = f"{cfg.paths.scores}/{cfg.dataset.name}_dual_{num_itr}"

        if cfg.dataset.for_extrapolation.value is True:
            output_path += f"_{cfg.dataset.for_extrapolation.subset_size}"

        date = datetime.datetime.now()
        output_path += f"_{date.month}_{date.day}"
        output_path += ".json"

        with open(output_path, "w") as f:
            json.dump(dual_scores, f)

        logger.info(f"Saved DUAL scores to {output_path}")
        # Pruning and evaluation

        if cfg.pruning.prune is True:
            prune(
                trainset=trainset,
                test_loader=test_loader,
                scores_dict=dual_scores,
                cfg=cfg,
                wandb_name="dual",
                rebalance_labels=False,
                device=device,
                sampling_method="beta",
                pred_mean=list(pred_mean.values()),
                mu_d=mu_d,
            )


if __name__ == "__main__":
    default_config_path = os.path.join(
        os.path.dirname(__file__), "configs", "dual_config.yaml"
    )
    config_path = parse_config(
        default_config=default_config_path,
        description="Run DUAL score Pruning",
    )
    main(cfg_path=config_path)
