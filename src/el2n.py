import json
import time

import numpy as np
import torch
import torchvision
from torch.optim import Adam
from torch.utils.data import DataLoader

from utils import IndexDataset, ResNet18, transform_test, transform_train


def get_error(model, X, target):
    target = torch.nn.functional.one_hot(target, num_classes=10)
    preds = model(X)
    prob_preds = torch.nn.functional.softmax(preds, dim=-1)
    errors = prob_preds - target
    scores = torch.norm(errors, p=2, dim=-1)
    scores = scores.cpu().detach().numpy()
    return scores


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)

num_train_examples = len(train_data)

train_data = IndexDataset(train_data)

epochs = 15

num_ensembles = 10


train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_test
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2
)

el2n_scores = {i: [] for i in range(num_train_examples)}

start_time = time.time()

for model_idx in range(num_ensembles):
    model = ResNet18().to(device)
    optimizer = Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, target, sample_idx) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            printed = False
            if batch_idx % 200 == 199:  # Print every 200 mini-batches
                printed = True
                print(
                    "Model %d - [%d, %5d] loss: %.3f"
                    % (model_idx, epoch + 1, batch_idx + 1, running_loss / 200)
                )
                running_loss = 0.0
                step_val = epoch * len(train_loader) + batch_idx + 1

            if len(train_loader) < 199 and not printed:
                if batch_idx % 100 == 99:
                    print(
                        "Model %d - [%d, %5d] loss: %.3f"
                        % (model_idx + 1, epoch + 1, batch_idx + 1, running_loss / 100)
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
        print(
            "Model %d - [epoch:%d,  accuracy: %.3f" % (model_idx, epoch + 1, accuracy)
        )

    for data, target, sample_idx in train_loader:
        data, target = data.to(device), target.to(device)
        scores = get_error(model, data, target)
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

with open("el2n_score.json", "w") as f:
    json.dump(el2n_values, f)
