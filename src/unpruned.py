import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

import wandb
from utils import ResNet18, transform_test, transform_train

wandb.init(project="cifar10_pruning", name="unpruned")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2
)

# Initialize the ConvNet model
net = ResNet18().to(device)
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Train the ConvNet
start_time = time.time()
for epoch in range(30):  # Adjust the number of epochs as needed
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:  # Print every 200 mini-batches
            print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 200))
            wandb.log({"Loss": running_loss / 200}, step=epoch * len(trainloader) + i)
            # Log the loss to wandb, so that we can visualize it
            running_loss = 0.0
            step_val = epoch * len(trainloader) + i + 1

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print("[epoch:%d,  accuracy: %.3f" % (epoch + 1, accuracy))
    wandb.log({"Accuracy": accuracy}, step=step_val)

    scheduler.step()


end_time = time.time()
training_time = end_time - start_time

# Test the ConvNet on the test set
correct = 0
total = 0
top5_correct = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        _, top5_predicted = torch.topk(outputs.data, 5, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        top5_correct += (top5_predicted == labels.view(-1, 1)).sum().item()

accuracy = 100 * correct / total
top5_accuracy = 100 * top5_correct / total

wandb.summary = {
    "Final-Accuracy": accuracy,
    "Top-5 Accuracy": top5_accuracy,
    "Training Time": training_time,
}

wandb.finish()
