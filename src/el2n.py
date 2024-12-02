import json
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.optim import Adam

from utils.dataset import prepare_data
from utils.evaluate import evaluate
from utils.models import get_model
from utils.prune_utils import get_error, prune


@hydra.main(config_path="configs", config_name="el2n_config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for num_itr in range(cfg.experiment.num_iterations):
        trainset, train_loader, test_loader = prepare_data(
            cfg.dataset, cfg.training.batch_size
        )

        num_train_examples = len(trainset)

        el2n_scores = {i: [] for i in range(num_train_examples)}
        torch.cuda.empty_cache()
        start_time = time.time()

        for model_idx in range(cfg.uncertainty.num_ensembles):
            model = get_model(
                model_name=cfg.model.name, num_classes=cfg.dataset.num_classes
            ).to(device)
            optimizer = Adam(model.parameters(), lr=cfg.training.lr)

            for epoch in range(cfg.uncertainty.prune_epochs):
                train_losses = []
                for batch_idx, (data, target, sample_idx) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = torch.nn.functional.cross_entropy(output, target)

                    optimizer.zero_grad()
                    train_losses.append(loss)
                    loss.backward()
                    optimizer.step()

                test_acc = evaluate(model, test_loader, device)
                train_loss = torch.stack(train_losses).mean().item()
                print(
                    f"Model - {model_idx+1}, Epoch {epoch + 1}, Train Loss: {train_loss}, Test Accuracy: {test_acc}"
                )

            for data, target, sample_idx in train_loader:
                data, target = data.to(device), target.to(device)
                scores = get_error(
                    model, data, target, num_classes=cfg.dataset.num_classes
                )
                for i, sample in enumerate(sample_idx):
                    sample = sample.item()
                    el2n_scores[sample].append(scores[i])

        # take average of scores
        el2n_values = {}
        for sample, scores in el2n_scores.items():
            el2n_values[sample] = np.mean(scores).item()

        end_time = time.time()
        training_time = end_time - start_time
        print(f"Training time: {training_time:.2f} seconds")

        with open(
            f"{cfg.paths.scores}/{cfg.dataset.name}_el2n_score_{num_itr}.json",
            "w",
        ) as f:
            json.dump(el2n_values, f)

        prune(
            trainset=trainset,
            test_loader=test_loader,
            scores_dict=el2n_values,
            cfg=cfg,
            wandb_name="forgetting-prune",
            device=device,
        )


if __name__ == "__main__":
    main()
