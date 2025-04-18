import datetime
import json
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from loguru import logger
from numpy import linalg as LA
from omegaconf import OmegaConf
from scipy.special import softmax
from torch.optim import Adam

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.argparse import parse_config
from utils.dataset import prepare_data
from utils.evaluate import evaluate
from utils.models import get_model
from utils.prune_utils import prune

logger.remove()
logger.add(sys.stdout, format="{time:MM-DD HH:mm} - {message}")


def generate(probs, indexes, cfg):
    # Initialize variables
    k = 0
    window_size = cfg.pruning.window
    moving_averages = []
    trajectory_len = cfg.pruning.trajectory
    decay = cfg.pruning.decay

    while k < trajectory_len - window_size + 1:
        probs_window = probs[k : k + window_size, :]
        indexes_window = indexes[k : k + window_size, :]
        probs_window_softmax = softmax(probs_window, axis=2)

        probs_window_rere = []
        # Reorganize probabilities according to indexes
        for i in range(window_size):
            probs_window_re = torch.zeros_like(
                torch.tensor(probs_window_softmax[0, :, :])
            )
            probs_window_re = probs_window_re.index_add(
                0,
                torch.tensor(indexes_window[i], dtype=int),
                torch.tensor(probs_window_softmax[i, :]),
            )
            probs_window_rere.append(probs_window_re)

        probs_window_kd = []
        # Calculate KL divergence in one window
        for j in range(window_size - 1):
            log = torch.log(probs_window_rere[j + 1] + 1e-8) - torch.log(
                probs_window_rere[j] + 1e-8
            )
            kd = torch.abs(torch.multiply(probs_window_rere[j + 1], log)).sum(axis=1)
            probs_window_kd.append(kd)
        probs_window_kd = np.array(probs_window_kd)

        window_average = probs_window_kd.sum(0) / (window_size - 1)

        window_diffdiff = []
        for ii in range(window_size - 1):
            window_diffdiff.append((np.array(probs_window_kd[ii]) - window_average))
        window_diffdiff_norm = LA.norm(np.array(window_diffdiff), axis=0)
        moving_averages.append(
            window_diffdiff_norm
            * decay
            * (1 - decay) ** (trajectory_len - window_size - k)
        )
        k += 1
        logger.info(str(k) + " window ok!")

    moving_averages_sum = np.squeeze(sum(np.array(moving_averages), 0))
    data_mask = moving_averages_sum.argsort()
    moving_averages_sum_sort = np.sort(moving_averages_sum)

    # create a dict with data_mask as key and moving_averages_sum_sort as value
    score_dict = {
        int(key): float(value)
        for key, value in zip(data_mask, moving_averages_sum_sort)
    }
    return score_dict


def main(cfg_path: str):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    cudnn.benchmark = True
    cfg = OmegaConf.load(cfg_path)
    cfg = cfg.PLACES_365

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainset, train_loader, test_loader, num_samples = prepare_data(
        cfg.dataset, cfg.training.batch_size
    )
    logger.info(f"Loaded dataset: {cfg.dataset.name}, Device: {device}")

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
        # Initialize model and optimizer
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

        torch.cuda.empty_cache()
        output_epochs, index_epochs = [], []
        for epoch in range(cfg.training.num_epochs):
            train_losses = []

            for batch_idx, (data, target, sample_idx) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                # loss = criterion(output, target_var)

                loss = torch.nn.functional.cross_entropy(output, target)
                index_batch = sample_idx

                # use the mapped indices if the dataset is for extrapolation
                if cfg.dataset.for_extrapolation.value is True:
                    index_batch = [mapping[idx.item()] for idx in index_batch]

                if batch_idx == 0:
                    output_epoch = np.array(output.detach().cpu())
                    index_epoch = np.array(index_batch)
                else:
                    output_epoch = np.concatenate(
                        (output_epoch, np.array(output.detach().cpu())), axis=0
                    )
                    index_epoch = np.concatenate(
                        (index_epoch, np.array(index_batch)), axis=0
                    )

                optimizer.zero_grad()
                train_losses.append(loss)
                loss.backward()
                optimizer.step()

                if batch_idx % cfg.logging.log_interval == 0 and batch_idx > 0:
                    logger.info(
                        f"Epoch {epoch + 1}/{cfg.training.num_epochs}, "
                        f"Itr {batch_idx}/{len(train_loader)}, "
                        f"Loss: {torch.stack(train_losses).mean().item():.5f}, "
                        f"Test Acc: {evaluate(model, test_loader, device):.5f}, "
                    )

            output_epochs.append(output_epoch)
            index_epochs.append(index_epoch)

            test_acc = evaluate(model, test_loader, device)
            train_loss = torch.stack(train_losses).mean().item()
            logger.info(
                f"Epoch {epoch + 1}, Train Loss: {train_loss:.5f}, Test Accuracy: {test_acc:.5f}"
            )

        model_name = f"{cfg.paths.models}/tdds"
        if cfg.dataset.for_extrapolation.value is True:
            model_name += f"_{cfg.dataset.for_extrapolation.subset_size}"

        model_name += f".pth"

        torch.save(model.state_dict(), model_name)

        logger.info(f"Saved model to {model_name}")

        logger.info("Computing Importance Scores")
        # output_epochs = np.array(output_epochs[: cfg.pruning.trajectory])
        # instead take the last trajectory epochs
        output_epochs = np.array(output_epochs[-cfg.pruning.trajectory :])

        # index_epochs = np.array(index_epochs[: cfg.pruning.trajectory])
        index_epochs = np.array(index_epochs[-cfg.pruning.trajectory :])

        logger.info(f"Shape of output_epochs: {output_epochs.shape}")
        logger.info(f"Shape of index_epochs: {index_epochs.shape}")

        tdds_score = generate(output_epochs, index_epochs, cfg)

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
            prune(
                trainset=trainset,
                test_loader=test_loader,
                scores_dict=tdds_score,
                cfg=cfg,
                wandb_name="tdds-",
                device=device,
            )


if __name__ == "__main__":
    default_config_path = os.path.join(
        os.path.dirname(__file__), "configs", "tdds_config.yaml"
    )
    config_path = parse_config(
        default_config=default_config_path, description="Run TDDS Pruning"
    )
    main(cfg_path=config_path)
