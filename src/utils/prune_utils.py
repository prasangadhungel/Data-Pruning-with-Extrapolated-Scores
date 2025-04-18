import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.optim as optim
from loguru import logger
from omegaconf import OmegaConf

import wandb
from utils.evaluate import evaluate, get_top_k_accuracy
from utils.models import get_model

logger.remove()
logger.add(sys.stdout, format="{time:MM-DD HH:mm} - {message}")


def calculate_uncertainty(history):
    """
    Calculates uncertainty based on the prediction history.

    Args:
        history: List of predictions for a sample over past epochs.

    Returns:
        Standard deviation of the predictions.
    """
    mean_prediction = sum(history) / len(history)
    variance = sum((p - mean_prediction) ** 2 for p in history) / len(history)
    return variance**0.5


def get_correct(logits, labels):
    return torch.argmax(logits, dim=-1) == labels


def init_forget_stats(num_train_examples):
    forget_stats = SimpleNamespace()
    forget_stats.prev_accs = np.zeros(num_train_examples, dtype=np.int32)
    forget_stats.num_forgets = np.zeros(num_train_examples, dtype=float)
    forget_stats.never_correct = np.arange(num_train_examples, dtype=np.int32)
    return forget_stats


def update_forget_stats(forget_stats, idxs, accs):
    forget_stats.num_forgets[idxs[forget_stats.prev_accs[idxs] > accs]] += 1
    forget_stats.prev_accs[idxs] = accs
    forget_stats.never_correct = np.setdiff1d(
        forget_stats.never_correct, idxs[accs.astype(bool)], True
    )
    return forget_stats


def get_embeddings(model, data_loader, device):
    model.eval()
    features = {}

    with torch.no_grad():
        for _, (data, _, sample_idx) in enumerate(data_loader):
            data = data.to(device)
            outputs = model(
                data
            )  # Replace with appropriate output extraction for your model

            for i, sample in enumerate(sample_idx):
                sample = sample.item()
                selected_features = outputs[i].detach().cpu().numpy()
                features[sample] = selected_features

    return features


def get_error(model, X, target, num_classes=10):
    target = torch.nn.functional.one_hot(target, num_classes=num_classes)
    preds = model(X)
    prob_preds = torch.nn.functional.softmax(preds, dim=-1)
    errors = prob_preds - target
    scores = torch.norm(errors, p=2, dim=-1)
    scores = scores.cpu().detach().numpy()
    return scores


def prune(trainset, test_loader, scores_dict, cfg, wandb_name, device):
    """
    Prune the dataset based on the uncertainty scores.
    """
    sorted_importance_scores = {
        k: v
        for k, v in sorted(
            scores_dict.items(), key=lambda item: item[1], reverse=True
        )
    }

    for prune_percentage in cfg.pruning.percentages:
        str_prune_percentage = str(int(prune_percentage * 100))
        wandb.init(
            project=cfg.dataset.name,
            name=wandb_name + str_prune_percentage,
        )
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))

        # sort the uncertainty scores in descending order and get the indices of most uncertain samples
        
        top_samples = list(sorted_importance_scores.keys())[
            : int((1 - prune_percentage) * len(sorted_importance_scores))
        ]

        # Get the indices of the top samples
        indices_to_keep = [int(sample) for sample in top_samples]

        pruned_trainset = torch.utils.data.Subset(trainset, indices_to_keep)

        trainloader = torch.utils.data.DataLoader(
            pruned_trainset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=2,
        )

        # Initialize the ConvNet model
        net = get_model(
            cfg.model.name,
            num_classes=cfg.dataset.num_classes,
            image_size=cfg.dataset.image_size,
        ).to(device)
        # Define the loss function and optimizer
        optimizer = optim.Adam(
            net.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.training.lr,
            epochs=cfg.training.num_epochs,
            steps_per_epoch=len(trainloader),
        )

        torch.cuda.empty_cache()
        start_time = time.time()
        for epoch in range(cfg.training.num_epochs):
            net.train()
            train_losses = []
            for i, data in enumerate(trainloader):
                inputs, labels, _ = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                optimizer.zero_grad()
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                scheduler.step()

            test_acc = evaluate(net, test_loader, device)
            train_loss = torch.stack(train_losses).mean().item()

            wandb.log({"Loss": train_loss}, step=epoch)
            wandb.log({"Accuracy": test_acc}, step=epoch)
            logger.info(
                f"Epoch {epoch + 1}, Train Loss: {train_loss:.5f}, Test Acc: {test_acc:.5f}"
            )

        end_time = time.time()
        training_time = end_time - start_time

        accuracy, top5_accuracy = get_top_k_accuracy(net, test_loader, device, k=5)

        wandb.log = {
            "Final-Accuracy": accuracy,
            "Top-5 Accuracy": top5_accuracy,
            "Training Time": training_time,
        }
        logger.info(
            f"Final Accuracy: {accuracy:.5f}, Top-5 Accuracy: {top5_accuracy:.5f}"
        )

        wandb.finish()
