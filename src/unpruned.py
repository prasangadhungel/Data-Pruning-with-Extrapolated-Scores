import time

import torch
import torch.nn as nn
import torch.optim as optim

import wandb
from utils.dataset import get_dataloaders
from utils.evaluate import evaluate, get_top_k_accuracy
from utils.models import get_model

num_classes = 100
dataset_str = "CIFAR10"
model_name = "ResNet18"
num_epochs = 50

batch_size = 256
max_lr = 0.001
grad_clip = 0.01
weight_decay = 0.001

wandb.init(project=f"{dataset_str}_pruning", name="unpruned")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

trainloader, testloader = get_dataloaders(dataset_str, batch_size=batch_size)

# Initialize the model
net = get_model(model_name, num_classes=num_classes)
net = net.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=max_lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr, epochs=num_epochs, steps_per_epoch=len(trainloader)
)
torch.cuda.empty_cache()

# Train
start_time = time.time()
for epoch in range(num_epochs):  # Adjust the number of epochs as needed
    net.train()

    train_losses = []
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        train_losses.append(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    test_acc = evaluate(net, testloader, device)
    train_loss = torch.stack(train_losses).mean().item()
    wandb.log({"Loss": train_loss}, step=epoch)
    wandb.log({"Accuracy": test_acc}, step=epoch)
    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Test Accuracy: {test_acc}"
    )

end_time = time.time()
training_time = end_time - start_time

accuracy, top5_accuracy = get_top_k_accuracy(net, testloader, device, k=5)

wandb.log = {
    "Final-Accuracy": accuracy,
    "Top-5 Accuracy": top5_accuracy,
    "Training Time": training_time,
}
print(f"Final Accuracy: {accuracy}, Top-5 Accuracy: {top5_accuracy}")

wandb.finish()
