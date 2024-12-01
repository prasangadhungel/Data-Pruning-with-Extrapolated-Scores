import json
import time

import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.optim import Adam

import wandb
from utils.dataset import prepare_data
from utils.evaluate import evaluate, get_top_k_accuracy
from utils.models import get_model
from utils.prune_utils import calculate_uncertainty


@hydra.main(config_path="configs", config_name="du_config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for num_itr in range(cfg.experiment.num_iterations):
        trainset, train_loader, test_loader = prepare_data(
            cfg.dataset, cfg.training.batch_size
        )

        # Initialize model and optimizer
        model = get_model(
            model_name=cfg.model.name, num_classes=cfg.dataset.num_classes
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
                if batch_idx % cfg.logging.log_interval == 0:
                    print(
                        f"Epoch {epoch + 1}/{cfg.training.num_epochs}, "
                        f"Iteration {batch_idx}/{len(train_loader)}, "
                        f"Loss: {torch.stack(train_losses).mean().item()}, "
                        f"Test Accuracy: {evaluate(model, test_loader, device)}, "
                        f"Time: {time.time() - start_time}"
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
            print(
                f"Epoch {epoch + 1}, Train Loss: {train_loss}, Test Accuracy: {test_acc}"
            )

        # Save dynamic uncertainty scores
        dynamic_uncertainty = {}
        for sample_idx, history in uncertainty_history.items():
            std_devs = [
                calculate_uncertainty(history[i + 2 : i + 2 + uncertainty_window])
                for i in range(len(history) - uncertainty_window - 1)
            ]
            dynamic_uncertainty[sample_idx] = sum(std_devs) / len(std_devs)

        output_path = (
            f"{cfg.paths.scores}/{cfg.dataset.name}_dynamic_uncertainty_{num_itr}.json"
        )
        with open(output_path, "w") as f:
            json.dump(dynamic_uncertainty, f)

        # Pruning and evaluation
        for prune_percentage in cfg.pruning.percentages:
            str_prune_percentage = str(int(prune_percentage * 100))
            wandb.init(
                project=f"{cfg.dataset.name}",
                name="dynamic-uncertainty-" + str_prune_percentage,
            )
            wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
            # Get top samples by uncertainty
            sorted_uncertainty_scores = {
                k: v
                for k, v in sorted(
                    dynamic_uncertainty.items(), key=lambda item: item[1], reverse=True
                )
            }
            top_samples = list(sorted_uncertainty_scores.keys())[
                : int((1 - prune_percentage) * len(sorted_uncertainty_scores))
            ]
            indices_to_keep = [int(sample) for sample in top_samples]

            # Prune dataset and create dataloader
            pruned_trainset = torch.utils.data.Subset(trainset, indices_to_keep)
            trainloader = torch.utils.data.DataLoader(
                pruned_trainset,
                batch_size=cfg.training.batch_size,
                shuffle=True,
                num_workers=2,
            )

            # Train pruned model
            net = get_model(cfg.model.name, num_classes=cfg.dataset.num_classes).to(
                device
            )
            criterion = nn.CrossEntropyLoss()
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

            for epoch in range(cfg.training.num_epochs):
                net.train()
                train_losses = []
                for i, data in enumerate(trainloader):
                    inputs, labels, _ = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    train_losses.append(loss)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                test_acc = evaluate(net, test_loader, device)
                train_loss = torch.stack(train_losses).mean().item()
                wandb.log({"Loss": train_loss, "Accuracy": test_acc}, step=epoch)
                print(
                    f"Epoch {epoch + 1}, Train Loss: {train_loss}, Test Accuracy: {test_acc}"
                )

            # Log final metrics
            accuracy, top5_accuracy = get_top_k_accuracy(net, test_loader, device, k=5)
            wandb.log(
                {
                    "Final-Accuracy": accuracy,
                    "Top-5 Accuracy": top5_accuracy,
                    "Training Time": time.time() - start_time,
                }
            )
            wandb.finish()


if __name__ == "__main__":
    main()
