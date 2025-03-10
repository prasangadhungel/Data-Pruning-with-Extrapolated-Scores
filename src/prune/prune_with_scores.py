import json
import os
import random
import sys

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.argparse import parse_config
from utils.dataset import prepare_data
from utils.prune_utils import prune

logger.remove()
logger.add(sys.stdout, format="{time:MM-DD HH:mm} - {message}")


def main(cfg_path: str):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    cfg = OmegaConf.load(cfg_path)
    cfg = cfg.PLACES_365
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainset, _, test_loader, _ = prepare_data(cfg.dataset, cfg.training.batch_size)
    logger.info(f"Loaded dataset: {cfg.dataset.name}, Device: {device}")

    # read json from cfg.json_path
    with open(cfg.json_path, "r") as f:
        data = json.load(f)

    importance_score = {int(k): v for k, v in data.items()}

    prune(
        trainset=trainset,
        test_loader=test_loader,
        scores_dict=importance_score,
        cfg=cfg,
        wandb_name=cfg.wandb_name,
        device=device,
    )


if __name__ == "__main__":
    default_config_path = os.path.join(
        os.path.dirname(__file__), "configs", "json_config.yaml"
    )
    config_path = parse_config(
        default_config=default_config_path,
        description="Run Pruning with provided scores",
    )
    main(cfg_path=config_path)
