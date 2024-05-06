import json
import time

import torch
import torchvision
from torch.optim import Adam
from torch.utils.data import DataLoader

from utils import IndexDataset, ResNet18, transform_test, transform_train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


train_data = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)

train_data = IndexDataset(train_data)

pruning_ratio = 0.2
epochs = 30
uncertainty_window = 10

model = ResNet18().to(device)
optimizer = Adam(model.parameters(), lr=0.001)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_test
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2
)

uncertainty_history = {i: [] for i in range(len(train_data))}

start_time = time.time()
for epoch in range(epochs):
    running_loss = 0.0
    for batch_idx, (data, target, sample_idx) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for i, sample in enumerate(sample_idx):
            # sample_idx = batch_idx * train_loader.batch_size + i
            sample = sample.item()
            softmax_output = torch.nn.functional.softmax(output[i], dim=0)
            prediction = softmax_output[target[i]]
            prediction = prediction.detach().cpu().numpy().item()
            uncertainty_history[sample].append(prediction)

        running_loss += loss.item()
        printed = False
        if batch_idx % 200 == 199:  # Print every 200 mini-batches
            printed = True
            print(
                "[%d, %5d] loss: %.3f" % (epoch + 1, batch_idx + 1, running_loss / 200)
            )
            running_loss = 0.0
            step_val = epoch * len(train_loader) + batch_idx + 1

        if len(train_loader) < 199 and not printed:
            if batch_idx % 100 == 99:
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, batch_idx + 1, running_loss / 100)
                )
                running_loss = 0.0
                step_val = epoch * len(train_loader) + batch_idx + 1

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print("[epoch:%d,  accuracy: %.3f" % (epoch + 1, accuracy))
    # # Calculate dynamic uncertainty after exceeding uncertainty window
    # if epoch >= uncertainty_window - 1:
    #     for sample_idx, history in uncertainty_history.items():
    #         if len(history) == uncertainty_window:
    #             uncertainty = calculate_uncertainty(history)
    #             uncertainty_history[sample_idx] = uncertainty

end_time = time.time()
training_time = end_time - start_time
print(f"Training time: {training_time:.2f} seconds")

print("Uncertainty history of first sample:")
print(uncertainty_history[0])

with open("uncertainty_history.json", "w") as f:
    json.dump(uncertainty_history, f)

# Calculate final dynamic uncertainty for all data points
dynamic_uncertainty = {}
for sample_idx, history in uncertainty_history.items():
    # calculate standard deviation of window length uncertainty_window
    std_devs = []
    for i in range(len(history) - uncertainty_window + 1):
        std_devs.append(calculate_uncertainty(history[i : i + uncertainty_window]))

    # mean of std devs
    dynamic_uncertainty[sample_idx] = sum(std_devs) / len(std_devs)

# Sort data by descending dynamic uncertainty
with open("dynamic_uncertainty.json", "w") as f:
    json.dump(dynamic_uncertainty, f)

sorted_data = sorted(train_data, key=lambda x: dynamic_uncertainty[x[2]], reverse=True)

# Prune data based on pruning ratio
num_samples = len(sorted_data)
num_to_keep = int((1 - pruning_ratio) * num_samples)
pruned_data = sorted_data[:num_to_keep]
