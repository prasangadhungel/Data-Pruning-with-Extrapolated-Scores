import json
import logging
import os
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.optim import Adam
from utils.argparse import parse_config
from utils.dataset import prepare_data
from utils.evaluate import evaluate
from utils.models import get_model
from utils.prune_utils import (get_correct, init_forget_stats, prune,
                               update_forget_stats)

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%m-%d %H:%M")


def main(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainset, train_loader, test_loader, num_train_examples = prepare_data(
        cfg.dataset, cfg.training.batch_size
    )

    logger.info(f"Loaded dataset: {cfg.dataset.name}, Device: {device}")

    for num_itr in range(cfg.experiment.num_iterations):
        # Initialize model and optimizer
        model = get_model(
            model_name=cfg.model.name, num_classes=cfg.dataset.num_classes
        ).to(device)
        optimizer = Adam(model.parameters(), lr=cfg.training.lr)

        forget_stats = init_forget_stats(num_train_examples)
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

                batch_accs = np.array(get_correct(output, target).cpu()).astype(
                    np.int32
                )
                forget_stats = update_forget_stats(forget_stats, sample_idx, batch_accs)

                if batch_idx % cfg.logging.log_interval == 0 and batch_idx > 0:
                    logger.info(
                        f"Epoch {epoch + 1}/{cfg.training.num_epochs}, "
                        f"Itr {batch_idx}/{len(train_loader)}, "
                        f"Loss: {torch.stack(train_losses).mean().item():.5f}, "
                        f"Test Acc: {evaluate(model, test_loader, device):.5f}, "
                        f"Time: {time.time() - start_time:.5f}"
                    )

            test_acc = evaluate(model, test_loader, device)
            train_loss = torch.stack(train_losses).mean().item()
            logger.info(
                f"Epoch {epoch + 1}, Train Loss: {train_loss:.5f}, Test Accuracy: {test_acc:.5f}"
            )

        end_time = time.time()
        training_time = end_time - start_time
        logger.info(f"Training time: {training_time:.2f} seconds")

        # Save forget stats as dict of idx: stat
        forget_stats_dict = {
            i: forget_stats.num_forgets[i] for i in range(num_train_examples)
        }

        output_path = (
            f"{cfg.paths.scores}/{cfg.dataset.name}_forget_score_{num_itr}.json"
        )
        with open(output_path, "w") as f:
            json.dump(forget_stats_dict, f)

        prune(
            trainset=trainset,
            test_loader=test_loader,
            scores_dict=forget_stats_dict,
            cfg=cfg,
            wandb_name="forgetting-prune-t1",
            device=device,
        )


if __name__ == "__main__":
    default_config_path = os.path.join(
        os.path.dirname(__file__), "configs", "forget_config.yaml"
    )
    config_path = parse_config(
        default_config=default_config_path, description="Run Forget Pruning"
    )
    main(cfg_path=config_path)
