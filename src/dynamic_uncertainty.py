import json
import time

import hydra
import torch
from omegaconf import DictConfig
from torch.optim import Adam

from utils.dataset import prepare_data
from utils.evaluate import evaluate
from utils.models import get_model
from utils.prune_utils import calculate_uncertainty, prune


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
        prune(
            trainset=trainset,
            test_loader=test_loader,
            scores_dict=dynamic_uncertainty,
            cfg=cfg,
            wandb_name="forgetting-prune",
            device=device,
        )


if __name__ == "__main__":
    main()
