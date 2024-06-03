import json
import time

import torch
from torch.optim import Adam

from utils.dataset import get_dataloaders_from_dataset, get_dataset
from utils.evaluate import evaluate
from utils.models import get_model
from utils.prune_utils import calculate_uncertainty

num_classes = 100
dataset_str = "SYNTHETIC_CIFAR100_50"
model_name = "ResNet50"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

batch_size = 256
max_lr = 0.001
grad_clip = 0.01
weight_decay = 0.001

epochs = 50
uncertainty_window = 10

for num_itr in [
    "train","0","1","2","3","4","5","6","7","8","9","10",
    "11","12","13","14", "15","16","17","18","19"
]:
    print(f"Starting iteration {num_itr}")
    trainset, testset = get_dataset(dataset_str, partial=True, subset_idx=num_itr)
    train_loader, test_loader = get_dataloaders_from_dataset(
        trainset, testset, batch_size=batch_size
    )

    model = get_model(model_name=model_name, num_classes=num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=0.001)

    uncertainty_history = {sample_idx: [] for _, _, sample_idx in trainset}

    torch.cuda.empty_cache()
    start_time = time.time()
    for epoch in range(epochs):
        train_losses = []

        for batch_idx, (data, target, sample_idx) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)

            optimizer.zero_grad()
            train_losses.append(loss)
            loss.backward()
            optimizer.step()

            if batch_idx % 1000 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}, Iteration {batch_idx}/{len(train_loader)}, Loss: {torch.stack(train_losses).mean().item()}, Test Accuracy: {evaluate(model, test_loader, device)}, Time: {time.time() - start_time}"
                )

            for i, sample in enumerate(sample_idx):
                sample = sample.item()
                softmax_output = torch.nn.functional.softmax(output[i], dim=0)
                prediction = softmax_output[target[i]]
                prediction = prediction.detach().cpu().numpy().item()
                uncertainty_history[sample].append(prediction)

        test_acc = evaluate(model, test_loader, device)
        train_loss = torch.stack(train_losses).mean().item()
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss}, Test Accuracy: {test_acc}")

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time: {training_time:.2f} seconds")

    # Calculate final dynamic uncertainty for all data points
    dynamic_uncertainty = {}
    for sample_idx, history in uncertainty_history.items():
        # calculate standard deviation of window length uncertainty_window
        std_devs = []
        for i in range(len(history) - uncertainty_window - 1):
            std_devs.append(
                calculate_uncertainty(history[i + 2 : i + 2 + uncertainty_window])
            )

        # mean of std devs
        dynamic_uncertainty[sample_idx] = sum(std_devs) / len(std_devs)

    # Sort data by descending dynamic uncertainty
    with open(
        f"/nfs/homedirs/dhp/unsupervised-data-pruning/scores/{dataset_str}_dynamic_uncertainty_{num_itr}_2.json",
        "w",
    ) as f:
        json.dump(dynamic_uncertainty, f)
