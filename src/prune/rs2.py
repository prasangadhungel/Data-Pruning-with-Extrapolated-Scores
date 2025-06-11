import math
import os
import random
import sys
import time

import torch
import torch.optim as optim
from loguru import logger
from omegaconf import OmegaConf

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import wandb
from utils.dataset import prepare_data
from utils.evaluate import evaluate, get_top_k_accuracy
from utils.helpers import parse_config, seed_everything
from utils.models import get_model

logger.remove()
logger.add(sys.stdout, format="{time:MM-DD HH:mm} - {message}")


def generate_pruned_indices_over_epochs(num_samples, num_samples_to_keep, num_epochs):
    all_indices = list(range(num_samples))
    indices_over_epochs = []

    unused_indices = set(all_indices)

    while len(indices_over_epochs) < num_epochs:
        # If not enough unused to fill the next epoch, take all remaining, shuffle, fill up with fresh shuffle.
        if len(unused_indices) < num_samples_to_keep:
            # Take what's left
            curr = list(unused_indices)
            random.shuffle(curr)
            # Fill up to keep_size from a reshuffle of all indices (except the ones just used)
            fresh = list(set(all_indices) - unused_indices)
            random.shuffle(fresh)
            curr += fresh[: num_samples_to_keep - len(curr)]
            indices_this_epoch = curr
            # Mark indices used
            used_now = set(curr)
            # Find what wasn't used this cycle, reset for next pass
            unused_indices = set(all_indices) - used_now
        else:
            # Pick a random subset of unused
            curr = random.sample(list(unused_indices), num_samples_to_keep)
            indices_this_epoch = curr
            unused_indices -= set(curr)

            # If we've used all, reset unused for next cycle
            if len(unused_indices) == 0:
                unused_indices = set(all_indices)

        indices_over_epochs.append(indices_this_epoch)
    return indices_over_epochs


def main(cfg_path: str):
    seed_everything(42)

    cfg = OmegaConf.load(cfg_path)
    cfg = cfg.PLACES_365
    logger.info("RS2 Pruning")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainset, train_loader, test_loader, num_samples = prepare_data(
        cfg.dataset, cfg.training.batch_size
    )
    logger.info(
        f"Loaded dataset: {cfg.dataset.name}, Device: {device}, Num_samples: {num_samples}"
    )

    num_epochs = cfg.training.num_epochs

    for num_itr in range(cfg.experiment.num_iterations):
        for prune_percentage in cfg.pruning.percentages:
            str_prune_percentage = str(int(prune_percentage * 100))
            frac_to_keep = 1 - prune_percentage
            num_samples_to_keep = int(frac_to_keep * num_samples)

            if cfg.rs2.replacement:
                logger.info("Using RS2 with replacement")
                wandb_name = f"RS2-replacement-{str_prune_percentage}"
            else:
                logger.info("Using RS2 without replacement")
                wandb_name = f"RS2-no-replacement-{str_prune_percentage}"

                indices_over_epochs = generate_pruned_indices_over_epochs(
                    num_samples, num_samples_to_keep, num_epochs
                )

            wandb.init(
                project=cfg.dataset.name,
                name=wandb_name,
            )
            wandb.config.update(OmegaConf.to_container(cfg, resolve=True))

            # Initialize the model
            model = get_model(
                model_name=cfg.model.name,
                num_classes=cfg.dataset.num_classes,
                image_size=cfg.dataset.image_size,
            ).to(device)

            optimizer = optim.Adam(
                model.parameters(),
                lr=cfg.training.lr,
                weight_decay=cfg.training.weight_decay,
            )

            num_itr_epoch = math.ceil(num_samples_to_keep / cfg.training.batch_size)

            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=cfg.training.lr,
                epochs=num_epochs,
                steps_per_epoch=num_itr_epoch,
            )

            torch.cuda.empty_cache()
            start_time = time.time()
            logger.info("Starting training")
            for epoch in range(num_epochs):
                model.train()
                train_losses = []

                if cfg.rs2.replacement:
                    indices_to_keep = random.sample(
                        range(num_samples), num_samples_to_keep
                    )

                else:
                    indices_to_keep = indices_over_epochs[epoch]

                pruned_trainset = torch.utils.data.Subset(trainset, indices_to_keep)
                train_loader = torch.utils.data.DataLoader(
                    pruned_trainset,
                    batch_size=cfg.training.batch_size,
                    shuffle=True,
                    num_workers=2,
                )

                for batch_idx, (data, target, sample_idx) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = torch.nn.functional.cross_entropy(output, target)

                    optimizer.zero_grad()
                    train_losses.append(loss)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    if batch_idx % cfg.logging.log_interval == 0 and batch_idx > 0:
                        logger.info(
                            f"Epoch {epoch + 1}/{num_epochs}, "
                            f"Iteration {batch_idx}/{len(train_loader)}, "
                            f"Loss: {torch.stack(train_losses).mean().item()}, "
                            f"Test Acc: {evaluate(model, test_loader, device)}, "
                            f"Time: {time.time() - start_time}"
                        )

                # Evaluate the model
                test_acc = evaluate(model, test_loader, device)
                train_loss = torch.stack(train_losses).mean().item()
                wandb.log({"Loss": train_loss}, step=epoch)
                wandb.log({"Accuracy": test_acc}, step=epoch)

                logger.info(
                    f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test Acc: {test_acc:.4f}"
                )

            # Final metrics
            end_time = time.time()
            training_time = end_time - start_time

            accuracy, top5_accuracy = get_top_k_accuracy(
                model, test_loader, device, k=5
            )

            wandb.log(
                {
                    "Final-Accuracy": accuracy,
                    "Top-5 Accuracy": top5_accuracy,
                    "Training Time": training_time,
                }
            )
            logger.info(f"Final Acc: {accuracy:.4f}, Top-5 Acc: {top5_accuracy:.4f}")

            wandb.finish()


if __name__ == "__main__":
    default_config_path = os.path.join(
        os.path.dirname(__file__), "configs", "rs2_config.yaml"
    )
    config_path = parse_config(
        default_config=default_config_path, description="RS2 Pruning"
    )
    main(cfg_path=config_path)
