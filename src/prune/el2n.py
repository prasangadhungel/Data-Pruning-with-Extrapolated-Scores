import datetime
import json
import logging
import os
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.optim import Adam
from loguru import logger
import random

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.dataset import prepare_data
from utils.evaluate import evaluate
from utils.helpers import parse_config, seed_everything
from utils.models import get_model
from utils.prune_utils import get_error, prune
from collections import defaultdict

logger.remove()
logger.add(sys.stdout, format="{time:MM-DD HH:mm} - {message}")

def main(cfg_path: str):
    seed_everything(42)
    torch.backends.cudnn.benchmark = True

    cfg = OmegaConf.load(cfg_path)
    cfg = cfg.PLACES_365

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainset, train_loader, test_loader, num_samples = prepare_data(
        cfg.dataset, cfg.training.batch_size
    )
    logger.info(f"Loaded Dataset: {cfg.dataset.name}, device: {device}")
    print(range(num_samples), cfg.dataset.for_extrapolation.subset_size)
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
    print(len(train_loader), len(trainset))
    for num_itr in range(cfg.experiment.num_iterations):
        el2n_scores = defaultdict(list)
        torch.cuda.empty_cache()
        start_time = time.time()

        for model_idx in range(cfg.uncertainty.num_ensembles):
            model = get_model(
                model_name=cfg.model.name, num_classes=cfg.dataset.num_classes,image_size=cfg.dataset.image_size,
            ).to(device)
            optimizer = Adam(model.parameters(), lr=cfg.training.lr)

            for epoch in range(cfg.uncertainty.prune_epochs):
                train_losses = []
                for batch_idx, (data, target, sample_idx) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = torch.nn.functional.cross_entropy(output, target)

                    optimizer.zero_grad()
                    train_losses.append(loss)
                    loss.backward()
                    optimizer.step()

                test_acc = evaluate(model, test_loader, device)
                train_loss = torch.stack(train_losses).mean().item()
                logger.info(
                    f"Model - {model_idx+1}, Epoch {epoch + 1}, Train Loss: {train_loss}, Test Accuracy: {test_acc}"
                )

            for data, target, sample_idx in train_loader:
                data, target = data.to(device), target.to(device)
                scores = get_error(
                    model, data, target, num_classes=cfg.dataset.num_classes
                )
                for i, sample in enumerate(sample_idx):
                    sample = sample.item()
                    el2n_scores[sample].append(scores[i])

        # Take average of scores
        el2n_values = {
            sample: np.mean(scores).item() for sample, scores in el2n_scores.items()
        }
        print(len(el2n_values.keys()))
        #if cfg.dataset.for_extrapolation.value is True:
        #    el2n_values2 = {
        #        reversed_mapping[key]: value for key, value in el2n_values.items()
        #    }
        model_name = f"{cfg.paths.models}/el2n"
        if cfg.dataset.for_extrapolation.value is True:
            model_name += f"_{cfg.dataset.for_extrapolation.subset_size}"

        model_name += f".pth"

        torch.save(model.state_dict(), model_name)

        end_time = time.time()
        training_time = end_time - start_time
        logger.info(f"Training time: {training_time:.2f} seconds")

        date = datetime.datetime.now()
        output_path = f"{cfg.paths.scores}/{cfg.dataset.name}_el2n_score_{num_itr}_{date.month}_{date.day}.json"
        if cfg.dataset.for_extrapolation.value is True:
            output_path = f"{cfg.paths.scores}/{cfg.dataset.name}_el2n_score_{num_itr}_{date.month}_{date.day}_{cfg.dataset.for_extrapolation.subset_size}.json"
        with open(output_path, "w") as f:
            json.dump(el2n_values, f)

        logger.info(f"Saved EL2N scores to {output_path}")

        if cfg.pruning.prune is True:
            prune(
                trainset=trainset,
                test_loader=test_loader,
                scores_dict=el2n_values,
                cfg=cfg,
                wandb_name="el2n",
                device=device,
            )


if __name__ == "__main__":
    default_config_path = os.path.join(
        os.path.dirname(__file__), "configs", "el2n_config.yaml"
    )
    config_path = parse_config(
        default_config=default_config_path, description="Run EL2N Pruning"
    )
    main(cfg_path=config_path)
