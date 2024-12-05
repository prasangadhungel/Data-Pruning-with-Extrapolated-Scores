import json
import logging
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.optim import Adam

from utils.dataset import prepare_data
from utils.evaluate import evaluate
from utils.models import get_model
from utils.prune_utils import (get_correct, init_forget_stats, prune,
                               update_forget_stats)

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="forget_config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainset, train_loader, test_loader = prepare_data(
        cfg.dataset, cfg.training.batch_size
    )
    num_train_examples = len(trainset)
    logger.info(f"loaded dataset: {cfg.dataset.name}, device: {device}")

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

                if batch_idx % cfg.logging.log_interval == 0:
                    print(
                        f"Epoch {epoch + 1}/{cfg.training.num_epochs}, "
                        f"Iteration {batch_idx}/{len(train_loader)}, "
                        f"Loss: {torch.stack(train_losses).mean().item()}, "
                        f"Test Accuracy: {evaluate(model, test_loader, device)}, "
                        f"Time: {time.time() - start_time}"
                    )

        end_time = time.time()
        training_time = end_time - start_time
        print(f"Training time: {training_time:.2f} seconds")

        # save forget stats as dict of idx: stat
        forget_stats_dict = {
            i: forget_stats.num_forgets[i] for i in range(num_train_examples)
        }

        output_path = (
            f"{cfg.paths.scores}/{cfg.dataset.name}_forget_score_{num_itr}.json"
        )
        with open(
            output_path,
            "w",
        ) as f:
            json.dump(forget_stats_dict, f)

        prune(
            trainset=trainset,
            test_loader=test_loader,
            scores_dict=forget_stats_dict,
            cfg=cfg,
            wandb_name="forgetting-prune",
            device=device,
        )


if __name__ == "__main__":
    main()
