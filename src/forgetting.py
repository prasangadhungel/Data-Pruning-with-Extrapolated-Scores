import time
from types import SimpleNamespace
import json

import torch
import torchvision
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader

from utils import IndexDataset, ResNet18, transform_test, transform_train

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
  forget_stats.never_correct = np.setdiff1d(forget_stats.never_correct, idxs[accs.astype(bool)], True)
  return forget_stats


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)

num_train_examples = len(train_data)

train_data = IndexDataset(train_data)

epochs = 70

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

forget_stats = init_forget_stats(num_train_examples)

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

        batch_accs = np.array(get_correct(output, target).cpu()).astype(np.int32)
        forget_stats = update_forget_stats(forget_stats, sample_idx, batch_accs)

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

end_time = time.time()
training_time = end_time - start_time
print(f"Training time: {training_time:.2f} seconds")

# save forget stats as dict of idx: stat
forget_stats_dict = {i: forget_stats.num_forgets[i] for i in range(num_train_examples)}
with open("forget_stats.json", "w") as f:
    json.dump(forget_stats_dict, f)