import json
import os
import sys

import torch
from loguru import logger
from omegaconf import OmegaConf

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.dataset import prepare_data
from utils.helpers import parse_config, seed_everything
from utils.prune_utils import prune

logger.remove()
logger.add(sys.stdout, format="{time:MM-DD HH:mm} - {message}")
os.environ.setdefault("WANDB_MODE", "online")
os.environ.setdefault("WANDB_DISABLED", "false")
os.environ.setdefault("WANDB_API_KEY", "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")

def main(cfg_path: str):
    seed_everything(42)

    cfg = OmegaConf.load(cfg_path)
    cfg = cfg.PLACES_365
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainset, _, test_loader, _ = prepare_data(cfg.dataset, cfg.training.batch_size)
    logger.info(f"Loaded dataset: {cfg.dataset.name}, Device: {device}")

    # read json from cfg.json_path
    logger.info(f"Reading scores from {cfg.json_path}")
    with open(cfg.json_path, "r") as f:
        data = json.load(f)

    
    importance_score = {int(k): v for k, v in data.items()}

    # Check if subset_indices_path is provided in config
    if cfg.is_subset:
        
        logger.info(f"Scores: {len(importance_score)} out of {len(trainset)}")
        
        # Create modified scores: set to 0 for samples NOT in subset, keep loaded values for subset samples
        patched_importance_score = {}
        for idx in range(len(trainset)):
            # Use loaded score if available, otherwise 0
            patched_importance_score[idx] = importance_score.get(idx, 0.0)

        importance_score = patched_importance_score

    prune(
        trainset=trainset,
        test_loader=test_loader,
        scores_dict=importance_score,
        cfg=cfg,
        wandb_name=cfg.wandb_name,
        rebalance_labels=cfg.pruning.rebalance_labels,
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
