import json
import os
import random
import time

import torch
from loguru import logger
from omegaconf import OmegaConf
from torch.optim import Adam
from utils.argparse import parse_config
from utils.dataset import prepare_data
from utils.evaluate import evaluate
from utils.models import get_model
from utils.prune_utils import calculate_uncertainty, prune

logger.add(
    "/nfs/homedirs/dhp/unsupervised-data-pruning/logs/slurm/logfile.log",
    format="{time:MM-DD HH:mm} - {message}",
    rotation="10 MB",
)


def get_dynamic_uncertainty_scores(cfg, device, trainset, train_loader, test_loader):
    # Initialize model and optimizer
    model = get_model(
        model_name=cfg.model.name,
        num_classes=cfg.dataset.num_classes,
        image_size=cfg.dataset.image_size,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=cfg.training.lr)

    # Initialize variables
    uncertainty_window = cfg.uncertainty.window
    uncertainty_history = {sample_idx: [] for _, _, sample_idx in trainset}

    torch.cuda.empty_cache()
    start_time = time.time()
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

            # Log progress
            if batch_idx % cfg.logging.log_interval == 0 and batch_idx > 0:
                logger.info(
                    f"Epoch {epoch + 1}/{cfg.training.num_epochs}, "
                    f"Itr {batch_idx}/{len(train_loader)}, "
                    f"Loss: {torch.stack(train_losses).mean().item():.5f}, "
                    f"Test Acc: {evaluate(model, test_loader, device):.5f}, "
                    f"Time: {time.time() - start_time:.5f}"
                )

            # Update uncertainty history
            for i, sample in enumerate(sample_idx):
                sample = sample.item()
                softmax_output = torch.nn.functional.softmax(output[i], dim=0)
                prediction = softmax_output[target[i]]
                prediction = prediction.detach().cpu().numpy().item()
                uncertainty_history[sample].append(prediction)

        # Epoch logging
        test_acc = evaluate(model, test_loader, device)
        train_loss = torch.stack(train_losses).mean().item()
        logger.info(
            f"Epoch {epoch + 1}, Train Loss: {train_loss:.5f}, Test Acc: {test_acc:.5f}"
        )

    # save model
    torch.save(model.state_dict(), f"{cfg.paths.models}/dynamic_uncertainty.pth")
    # Save dynamic uncertainty scores

    dynamic_uncertainty = {}
    for sample_idx, history in uncertainty_history.items():
        std_devs = [
            calculate_uncertainty(history[i + 2 : i + 2 + uncertainty_window])
            for i in range(len(history) - uncertainty_window - 1)
        ]
        dynamic_uncertainty[sample_idx] = sum(std_devs) / len(std_devs)

    return dynamic_uncertainty


def main(cfg_path: str):
    random.seed(42)
    torch.manual_seed(42)

    cfg = OmegaConf.load(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainset, train_loader, test_loader, num_samples = prepare_data(
        cfg.dataset, cfg.training.batch_size
    )
    logger.info(f"Loaded dataset: {cfg.dataset.name}, Device: {device}")

    if cfg.dataset.for_extrapolation.value is True:
        indices_to_keep = random.sample(
            range(num_samples), cfg.dataset.for_extrapolation.subset_size
        )

        trainset = torch.utils.data.Subset(trainset, indices_to_keep)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=2,
        )

    for num_itr in range(cfg.experiment.num_iterations):
        dynamic_uncertainty = get_dynamic_uncertainty_scores(
            cfg, device, trainset, train_loader, test_loader
        )

        output_path = (
            f"{cfg.paths.scores}/{cfg.dataset.name}_dynamic_uncertainty_{num_itr}.json"
        )
        with open(output_path, "w") as f:
            json.dump(dynamic_uncertainty, f)

        # Pruning and evaluation

        if cfg.pruning.prune is True:
            prune(
                trainset=trainset,
                test_loader=test_loader,
                scores_dict=dynamic_uncertainty,
                cfg=cfg,
                wandb_name="dynamic-uncertainty",
                device=device,
            )


if __name__ == "__main__":
    default_config_path = os.path.join(
        os.path.dirname(__file__), "configs", "du_config.yaml"
    )
    config_path = parse_config(
        default_config=default_config_path,
        description="Run Dynamic Uncertainty Pruning",
    )
    main(cfg_path=config_path)
