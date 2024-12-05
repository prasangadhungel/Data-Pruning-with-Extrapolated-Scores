import argparse
import json
import logging
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.optim import Adam

from utils.dataset import prepare_data
from utils.evaluate import evaluate
from utils.models import get_model
from utils.prune_utils import get_error, prune

logger = logging.getLogger(__name__)


def main(cfg_path: str, cfg_name: str):
    # Load the configuration using OmegaConf
    cfg = OmegaConf.load(f"{cfg_path}/{cfg_name}.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainset, train_loader, test_loader = prepare_data(
        cfg.dataset, cfg.training.batch_size
    )
    num_train_examples = len(trainset)
    logger.info(f"Loaded dataset: {cfg.dataset.name}, Device: {device}")

    for num_itr in range(cfg.experiment.num_iterations):
        el2n_scores = {i: [] for i in range(num_train_examples)}
        torch.cuda.empty_cache()
        start_time = time.time()

        for model_idx in range(cfg.uncertainty.num_ensembles):
            model = get_model(
                model_name=cfg.model.name, num_classes=cfg.dataset.num_classes
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

        end_time = time.time()
        training_time = end_time - start_time
        logger.info(f"Training time: {training_time:.2f} seconds")

        output_path = f"{cfg.paths.scores}/{cfg.dataset.name}_el2n_score_{num_itr}.json"
        with open(output_path, "w") as f:
            json.dump(el2n_values, f)

        prune(
            trainset=trainset,
            test_loader=test_loader,
            scores_dict=el2n_values,
            cfg=cfg,
            wandb_name="el2n",
            device=device,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EL2N Pruning")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs",
        help="Path to the configuration files (default: configs)",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="el2n_config",
        help="Name of the configuration file (without .yaml extension) (default: el2n_config)",
    )
    args = parser.parse_args()

    main(cfg_path=args.config_path, cfg_name=args.config_name)
