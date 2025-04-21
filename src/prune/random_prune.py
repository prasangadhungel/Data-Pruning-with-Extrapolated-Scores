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
from utils.helpers import parse_config, seed_everything
from utils.dataset import prepare_data
from utils.evaluate import evaluate, get_top_k_accuracy
from utils.models import get_model

logger.remove()
logger.add(sys.stdout, format="{time:MM-DD HH:mm} - {message}")


def main(cfg_path: str):
    seed_everything(42)

    cfg = OmegaConf.load(cfg_path)
    cfg = cfg.IMAGENET
    logger.info("Random Pruning")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainset, train_loader, test_loader, num_samples = prepare_data(
        cfg.dataset, cfg.training.batch_size
    )
    logger.info(
        f"Loaded dataset: {cfg.dataset.name}, Device: {device}, Num_samples: {num_samples}"
    )

    for num_itr in range(cfg.experiment.num_iterations):
        for prune_percentage in cfg.pruning.percentages:
            str_prune_percentage = str(int(prune_percentage * 100))
            if prune_percentage == 0:
                logger.info("Unpruned Training")
                wandb_name = "unpruned-"
            if num_itr > 0:
                train_loader = torch.utils.data.DataLoader(
                    trainset,
                    batch_size=cfg.training.batch_size,
                    shuffle=True,
                    num_workers=2,
                )
            else:
                wandb_name = f"random-prune-{str_prune_percentage}"
                frac_to_keep = 1 - prune_percentage
                num_samples_to_keep = int(frac_to_keep * num_samples)
                indices_to_keep = random.sample(range(num_samples), num_samples_to_keep)
                pruned_trainset = torch.utils.data.Subset(trainset, indices_to_keep)
                train_loader = torch.utils.data.DataLoader(
                    pruned_trainset,
                    batch_size=cfg.training.batch_size,
                    shuffle=True,
                    num_workers=2,
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
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=cfg.training.lr,
                epochs=cfg.training.num_epochs,
                steps_per_epoch=len(train_loader),
            )

            torch.cuda.empty_cache()
            start_time = time.time()
            logger.info("Starting training")
            for epoch in range(cfg.training.num_epochs):
                model.train()
                train_losses = []

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
                            f"Epoch {epoch + 1}/{cfg.training.num_epochs}, "
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
        os.path.dirname(__file__), "configs", "random_prune_config.yaml"
    )
    config_path = parse_config(
        default_config=default_config_path, description="Run Random Pruning"
    )
    main(cfg_path=config_path)
