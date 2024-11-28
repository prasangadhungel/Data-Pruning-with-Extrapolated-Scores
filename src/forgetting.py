import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import Adam
from torch.utils.data import DataLoader

import wandb
from utils.dataset import IndexDataset, get_transforms
from utils.evaluate import evaluate, get_top_k_accuracy
from utils.models import get_model
from utils.prune_utils import get_correct, init_forget_stats, update_forget_stats

num_classes = 100
dataset_str = "CIFAR100"
model_name = "ResNet50"
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 256
max_lr = 0.001
grad_clip = 0.01
weight_decay = 0.001

for num_itr in range(5):
    if dataset_str == "CIFAR10":
        mean_cifar10 = (0.4914, 0.4822, 0.4465)
        std_cifar10 = (0.2470, 0.2435, 0.2616)
        transform_train, transform_test = get_transforms(mean_cifar10, std_cifar10)
        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )

    elif dataset_str == "CIFAR100":
        mean_cifar100 = (0.5071, 0.4865, 0.4409)
        std_cifar100 = (0.2673, 0.2564, 0.2761)
        transform_train, transform_test = get_transforms(mean_cifar100, std_cifar100)
        trainset = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR100(
            root="./data", train=False, download=True, transform=transform_test
        )

    num_train_examples = len(trainset)

    indexed_trainset = IndexDataset(trainset)

    epochs = 70

    model = get_model(model_name=model_name, num_classes=num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=0.001)

    train_loader = DataLoader(
        indexed_trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=2 * batch_size, shuffle=False, num_workers=2
    )

    forget_stats = init_forget_stats(num_train_examples)
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

            batch_accs = np.array(get_correct(output, target).cpu()).astype(np.int32)
            forget_stats = update_forget_stats(forget_stats, sample_idx, batch_accs)

        test_acc = evaluate(model, testloader, device)
        train_loss = torch.stack(train_losses).mean().item()
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss}, Test Accuracy: {test_acc}")

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time: {training_time:.2f} seconds")

    # save forget stats as dict of idx: stat
    forget_stats_dict = {
        i: forget_stats.num_forgets[i] for i in range(num_train_examples)
    }
    with open(
        f"/nfs/homedirs/dhp/unsupervised-data-pruning/scores/{dataset_str}_forget_score_{num_itr}.json",
        "w",
    ) as f:
        json.dump(forget_stats_dict, f)

    for prune_percentage in [0.1, 0.2, 0.25, 0.3, 0.5]:
        str_prune_percentage = str(int(prune_percentage * 100))
        wandb.init(
            project=f"{dataset_str}",
            name="forgetting-prune-" + str_prune_percentage,
        )

        # sort the uncertainty scores in descending order and get the indices of most uncertain samples
        sorted_forget_scores = {
            k: v
            for k, v in sorted(
                forget_stats_dict.items(), key=lambda item: item[1], reverse=True
            )
        }
        top_samples = list(sorted_forget_scores.keys())[
            : int((1 - prune_percentage) * len(sorted_forget_scores))
        ]

        # Get the indices of the top samples
        indices_to_keep = [int(sample) for sample in top_samples]

        pruned_trainset = torch.utils.data.Subset(trainset, indices_to_keep)

        trainloader = torch.utils.data.DataLoader(
            pruned_trainset, batch_size=batch_size, shuffle=True, num_workers=2
        )

        # Initialize the ConvNet model
        net = get_model(model_name=model_name, num_classes=num_classes).to(device)
        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        torch.cuda.empty_cache()
        start_time = time.time()
        for epoch in range(num_epochs):
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
                f"Epoch {epoch + 1}, Train Loss: {train_loss}, Test Accuracy: {test_acc}"
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
