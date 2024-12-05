import logging
import random
import time

import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf

import wandb
from utils.dataset import prepare_data
from utils.evaluate import evaluate, get_top_k_accuracy
from utils.models import get_model

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="random_prune_config")
def main(cfg: DictConfig):
    logger.info("Random Pruning")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainset, train_loader, test_loader = prepare_data(
        cfg.dataset, cfg.training.batch_size
    )
    num_samples = len(trainset)
    logger.info(f"loaded dataset: {cfg.dataset.name}, device: {device}")

    for num_itr in range(cfg.experiment.num_iterations):
        for prune_percentage in cfg.pruning.percentages:
            str_prune_percentage = str(int(prune_percentage * 100))
            if prune_percentage == 0:
                logger.info("Unpruned Training")
                wandb_name = "unpruned"
            else:
                wandb_name = f"random-prune-{str_prune_percentage}"
                frac_to_keep = 1 - prune_percentage
                num_samples_to_keep = int(frac_to_keep * num_samples)
                # Generate a random list of indices to keep
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
            model = get_model(cfg.model.name, num_classes=cfg.dataset.num_classes).to(
                device
            )

            # Define the loss function, optimizer, and scheduler
            criterion = nn.CrossEntropyLoss()
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
            logger.info(f"starting training")
            for epoch in range(cfg.training.num_epochs):
                model.train()
                train_losses = []

                for batch_idx, (data, target, sample_idx) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    train_losses.append(loss)

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                    if batch_idx % cfg.logging.log_interval == 0 and batch_idx > 0:
                        logger.info(
                            f"Epoch {epoch + 1}/{cfg.training.num_epochs}, "
                            f"Iteration {batch_idx}/{len(train_loader)}, "
                            f"Loss: {torch.stack(train_losses).mean().item()}, "
                            f"Test Accuracy: {evaluate(model, test_loader, device)}, "
                            f"Time: {time.time() - start_time}"
                        )

                # Evaluate the model
                test_acc = evaluate(model, test_loader, device)
                train_loss = torch.stack(train_losses).mean().item()
                wandb.log({"Loss": train_loss}, step=epoch)
                wandb.log({"Accuracy": test_acc}, step=epoch)

                logger.info(
                    f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test Accuracy: {test_acc:.4f}"
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
            logger.info(
                f"Final Accuracy: {accuracy:.4f}, Top-5 Accuracy: {top5_accuracy:.4f}"
            )

            wandb.finish()


if __name__ == "__main__":
    main()
