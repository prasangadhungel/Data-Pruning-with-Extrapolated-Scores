import time

import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler

import wandb
from utils.dataset import get_dataloaders, get_dataloaders_from_dataset, get_dataset
from utils.evaluate import evaluate, get_top_k_accuracy
from utils.models import get_model

num_classes = 100
dataset_str = "SYNTHETIC_CIFAR100_1M"
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
    # ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], #done
    # ["0", "1", "2", "3", "4"],
    # ["0", "1"],
    # ["train"]
    # ["50k"],
    # ["100k"],
    # ["250k"],
    ["500k"],
]:
    if len(num_itr) == 1:
        project_name = "knn_extrapolated_" + num_itr[0]

    else:
        project_name = "standard_" + str(len(num_itr) * 50) + "k"

    wandb.init(project=dataset_str, name=project_name)

    print(f"Starting iteration {num_itr}")
    trainset, testset = get_dataset(dataset_str, partial=True, subset_idxs=num_itr)
    trainloader, _ = get_dataloaders_from_dataset(
        trainset, testset, batch_size=batch_size
    )

    _, testloader = get_dataloaders("CIFAR100", batch_size=batch_size)

    net = get_model(model_name, num_classes=num_classes)
    net = net.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=max_lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer, max_lr, epochs=epochs, steps_per_epoch=len(trainloader)
    )
    torch.cuda.empty_cache()

    # Train
    start_time = time.time()
    for epoch in range(epochs):  # Adjust the number of epochs as needed
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

        test_acc = evaluate(net, testloader, device)
        train_loss = torch.stack(train_losses).mean().item()
        wandb.log({"Loss": train_loss}, step=epoch)

        wandb.log({"Accuracy": test_acc}, step=epoch)
        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss}, Test Accuracy: {test_acc}"
        )

    # save the model in folder /nfs/homedirs/dhp/unsupervised-data-pruning/models/
    torch.save(
        net.state_dict(),
        f"/nfs/homedirs/dhp/unsupervised-data-pruning/models/{dataset_str}_{model_name}.pt",
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
