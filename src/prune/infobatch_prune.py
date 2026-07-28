import math
from torch.utils.data import Dataset

import argparse
import datetime
import json
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from loguru import logger
from numpy import linalg as LA
from omegaconf import OmegaConf
from scipy.special import softmax
from torch.optim import Adam
import wandb
import torch.optim as optim
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.dataset import prepare_data
from utils.evaluate import evaluate, get_top_k_accuracy
from utils.helpers import parse_config
from utils.models import get_model
#from prune.infobatch_lib import InfoBatch


import math
import numpy as np
from torch.utils.data import Dataset

class InfoBatch(Dataset):
    def __init__(self, dataset, ratio = 0.5, num_epoch=None, delta = 0.875):
        self.dataset = dataset
        self.ratio = ratio
        self.num_epoch = num_epoch
        self.delta = delta
        self.scores = np.ones([len(self.dataset)])
        self.transform = dataset.transform
        self.weights = np.ones(len(self.dataset))
        self.save_num = 0

    def __setscore__(self, indices, values):
        self.scores[indices] = values

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, target, idx = self.dataset[index]
        weight = self.weights[index]
        return data, target, idx, weight

    def prune(self):
        # prune samples that are well learned, rebalence the weight by scaling up remaining
        # well learned samples' learning rate to keep estimation about the same
        # for the next version, also consider new class balance

        b = self.scores<self.scores.mean()
        well_learned_samples = np.where(b)[0]
        pruned_samples = []
        pruned_samples.extend(np.where(np.invert(b))[0])
        selected = np.random.choice(well_learned_samples, int(self.ratio*len(well_learned_samples)),replace=False)
        self.reset_weights()
        if len(selected)>0:
            self.weights[selected]=1/self.ratio
            pruned_samples.extend(selected)
        print('Cut {} samples for next iteration'.format(len(self.dataset)-len(pruned_samples)))
        self.save_num += len(self.dataset)-len(pruned_samples)
        np.random.shuffle(pruned_samples)
        return pruned_samples

    def pruning_sampler(self):
        return InfoBatchSampler(self, self.num_epoch, self.delta)

    def no_prune(self):
        samples = list(range(len(self.dataset)))
        np.random.shuffle(samples)
        return samples

    def mean_score(self):
        return self.scores.mean()

    def normal_sampler_no_prune(self):
        return InfoBatchSampler(self.no_prune)

    def get_weights(self,indexes):
        return self.weights[indexes]

    def total_save(self):
        return self.save_num

    def reset_weights(self):
        self.weights = np.ones(len(self.dataset))



class InfoBatchSampler():
    def __init__(self, infobatch_dataset, num_epoch = math.inf, delta = 1):
        self.infobatch_dataset = infobatch_dataset
        self.seq = None
        self.stop_prune = num_epoch * delta
        self.seed = 0
        self.reset()

    def reset(self):
        np.random.seed(self.seed)
        self.seed+=1
        if self.seed>self.stop_prune:
            if self.seed <= self.stop_prune+1:
                self.infobatch_dataset.reset_weights()
            self.seq = self.infobatch_dataset.no_prune()
        else:
            self.seq = self.infobatch_dataset.prune()
        self.ite = iter(self.seq)
        self.new_length = len(self.seq)

    def __next__(self):
        try:
            nxt = next(self.ite)
            return nxt
        except StopIteration:
            self.reset()
            raise StopIteration

    def __len__(self):
        return len(self.seq)

    def __iter__(self):
        self.ite = iter(self.seq)
        return self

def prune(
    trainset,
    test_loader,
    scores_dict,
    cfg,
    ratios,
    wandb_name,
    rebalance_labels=False,
    device="cuda",
    sampling_method="topk",
    pred_mean=None,
    mu_d=None,
):
    """
    Prune the dataset based on the uncertainty scores.
    """
    score_vector = np.array(list(scores_dict))
    # sorted_importance_scores = {
    #     k: v
    #     for k, v in sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)
    # }

    
    str_prune_percentage = str(int(ratios * 100))
    wandb.init(
        project=cfg.dataset.name,
        name=wandb_name + str_prune_percentage,
        mode="offline",
    )
    wandb.config.update(OmegaConf.to_container(cfg, resolve=True))

    if not rebalance_labels:
        if sampling_method == "topk":
            # np.argsort(score_vector)
            # top_samples = list(sorted_importance_scores.keys())[
            #     : int((1 - ratios) * len(sorted_importance_scores))
            # ]
            indices_to_keep = np.argsort(score_vector)[: int((1 - ratios) * len(list(scores_dict)))]

        elif sampling_method == "beta":
            chosen = beta_sampling(
                prune_percentage=prune_percentage,
                pred_mean=pred_mean,
                mu_d=mu_d,
                c_d=4.0,
                score_vector=score_vector
            )
            indices_to_keep = chosen.tolist()

        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")

    else:
        # sort based on the labels and then prune the dataset
        # considering stratified sampling based on the labels
        label_to_indices = defaultdict(list)
        for idx in range(len(trainset)):
            _, label, sample_idx = trainset[
                idx
            ]  # assuming trainset returns (img, label, sample_idx)
            label_to_indices[label].append(sample_idx)

        # Perform stratified pruning
        indices_to_keep = []

        for label, indices in label_to_indices.items():
            # Sort these indices by their importance score (descending)
            indices.sort(key=lambda x: sorted_importance_scores[x], reverse=True)

            # Keep top (1 - prune_percentage) of them
            retain_count = int((1 - prune_percentage) * len(indices))
            indices_to_keep.extend(indices[:retain_count])

    pruned_trainset = torch.utils.data.Subset(trainset, indices_to_keep)

    trainloader = torch.utils.data.DataLoader(
        pruned_trainset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=2,
    )

    # Initialize the ConvNet model
    net = get_model(
        cfg.model.name,
        num_classes=cfg.dataset.num_classes,
        image_size=cfg.dataset.image_size,
    ).to(device)
    # Define the loss function and optimizer
    optimizer = optim.Adam(
        net.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.training.lr,
        epochs=cfg.training.num_epochs,
        steps_per_epoch=len(trainloader),
    )

    torch.cuda.empty_cache()
    start_time = time.time()
    for epoch in range(cfg.training.num_epochs):
        net.train()
        train_losses = []
        for i, data in enumerate(trainloader):
            inputs, labels, _, _ = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            optimizer.zero_grad()
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            scheduler.step()

        test_acc = evaluate(net, test_loader, device)
        train_loss = torch.stack(train_losses).mean().item()

        wandb.log({"Loss": train_loss}, step=epoch)
        wandb.log({"Accuracy": test_acc}, step=epoch)
        logger.info(
            f"Epoch {epoch + 1}, Train Loss: {train_loss:.5f}, Test Acc: {test_acc:.5f}"
        )

    end_time = time.time()
    training_time = end_time - start_time

    accuracy, top5_accuracy = get_top_k_accuracy(net, test_loader, device, k=5)

    wandb.log = {
        "Final-Accuracy": accuracy,
        "Top-5 Accuracy": top5_accuracy,
        "Training Time": training_time,
    }
    logger.info(
        f"Final Accuracy: {accuracy:.5f}, Top-5 Accuracy: {top5_accuracy:.5f}"
    )

    wandb.finish()


def main(cfg_path: str, dataset: str = None):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    cudnn.benchmark = True
    cfg = OmegaConf.load(cfg_path)
    
    # Use the dataset specified via argparse or default to CIFAR10
    if dataset:
        cfg = cfg.get(dataset,cfg.CIFAR10)
    else:
        cfg = cfg.CIFAR10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # trainset, train_loader, test_loader, num_samples = prepare_data(
    #     cfg.dataset, cfg.training.batch_size
    # )

    ratios=cfg.pruning.percentages
    for num_itr in range(cfg.experiment.num_iterations):
        for ratio in ratios:
        #1.Substitute dataset with InfoBatch dataset, optionally set r; delta and num_epoch are needed to anneal with full data
            ratio_inv=1-ratio
            trainset, train_loader, test_loader, num_samples = prepare_data(
                cfg.dataset, cfg.training.batch_size
            )
            trainset = InfoBatch(trainset, ratio_inv if ratio_inv else None, cfg.pruning.num_epochs)
            
            train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=cfg.training.batch_size, sampler = trainset.pruning_sampler())

            logger.info(f"Loaded dataset: {cfg.dataset.name}, Device: {device}")


        
            logger.info(f"Ratio {ratio}, Iteration: {num_itr} of {cfg.experiment.num_iterations}")
            # Initialize model and optimizer
            model = get_model(
                model_name=cfg.model.name,
                num_classes=cfg.dataset.num_classes,
                image_size=cfg.dataset.image_size,
            ).to(device)

            if cfg.training.optimizer == "sgd":
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=cfg.training.lr,
                    momentum=cfg.training.momentum,
                    weight_decay=cfg.training.weight_decay,
                    nesterov=cfg.training.nesterov,
                )
            elif cfg.training.optimizer == "lars":
                from utils.lars import Lars
                optimizer = Lars(model.parameters(), lr=cfg.training.lr, momentum=cfg.training.momentum, weight_decay=cfg.training.weight_decay)
            elif cfg.training.optimizer == "adam":
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=cfg.training.lr,
                    weight_decay=cfg.training.weight_decay,
                )
            else:
                raise ValueError(f"Unknown optimizer: {cfg.training.optimizer}")

            num_iter = len(train_loader)
            if cfg.training.lr_scheduler == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer, T_max=cfg.pruning.num_epochs * num_iter
                )
            elif cfg.training.lr_scheduler == "onecycle":
                # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, cfg.training.lr, steps_per_epoch=len(train_loader),
                #                                                  epochs=cfg.pruning.num_epochs, div_factor=cfg.training.div_factor,
                #                                                  final_div_factor=cfg.training.final_div, pct_start=cfg.training.pct_start)
                scheduler = optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=cfg.training.lr,
                    epochs=cfg.training.num_epochs,
                    steps_per_epoch=len(train_loader),
                )
            else:
                raise ValueError(f"Unknown optimizer: {cfg.training.optimizer}")
            criterion = torch.nn.CrossEntropyLoss(reduction='none')
            model.cuda()
            criterion.cuda()

            # output_epochs, loss_epochs, index_epochs = [], [], []
            for epoch in range(cfg.training.num_epochs):
                train_losses = []

                for batch_idx, (data, target, sample_idx,rescale_weight) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    # 
                    rescale_weight = rescale_weight.to(device)
                    input_var = torch.autograd.Variable(data)
                    target_var = torch.autograd.Variable(target)
                    output = model(input_var)
                    loss = criterion(output, target_var)
                    trainset.__setscore__(sample_idx.detach().cpu().numpy(),loss.detach().cpu().numpy())
                    loss = loss*rescale_weight
                    loss = torch.mean(loss)
                    # print(loss)
                    # loss = torch.nn.functional.cross_entropy(output, target)

                    loss_batch = (
                        torch.nn.functional.cross_entropy(output, target_var, reduce=False)
                        .detach()
                        .cpu()
                    )

                    index_batch = sample_idx

                    # use the mapped indices if the dataset is for extrapolation

                    if batch_idx == 0:
                        loss_epoch = np.array(loss_batch)
                        output_epoch = np.array(output.detach().cpu())
                        index_epoch = np.array(index_batch)
                    else:
                        loss_epoch = np.concatenate(
                            (loss_epoch, np.array(loss_batch)), axis=0
                        )
                        output_epoch = np.concatenate(
                            (output_epoch, np.array(output.detach().cpu())), axis=0
                        )
                        index_epoch = np.concatenate(
                            (index_epoch, np.array(index_batch)), axis=0
                        )

                    optimizer.zero_grad()
                    train_losses.append(loss)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    if batch_idx % cfg.logging.log_interval == 0 and batch_idx > 0:
                        logger.info(
                            f"Epoch {epoch + 1}/{cfg.training.num_epochs}, "
                            f"Itr {batch_idx}/{len(train_loader)}, "
                            f"Loss: {torch.stack(train_losses).mean().item():.5f}, "
                            f"Test Acc: {evaluate(model, test_loader, device):.5f}, "
                        )

                # if cfg.pruning.num_epochs - epoch <= cfg.pruning.trajectory:
                #     output_epochs.append(output_epoch)
                #     loss_epochs.append(loss_epoch)
                #     index_epochs.append(index_epoch)

                test_acc = evaluate(model, test_loader, device)
                train_loss = torch.stack(train_losses).mean().item()
                logger.info(
                    f"Epoch {epoch + 1}, Train Loss: {train_loss:.5f}, Test Accuracy: {test_acc:.5f}"
                )

            model_name = f"{cfg.paths.models}/infobatch"
            if cfg.dataset.for_extrapolation.value is True:
                model_name += f"_{cfg.dataset.for_extrapolation.subset_size}"

            model_name += f".pth"

            torch.save(model.state_dict(), model_name)

            logger.info(f"Saved model to {model_name}")

            logger.info("Computing Importance Scores")
            # # output_epochs = np.array(output_epochs[: cfg.pruning.trajectory])
            # # instead take the last trajectory epochs
            # output_epochs = np.array(output_epochs[-cfg.pruning.trajectory :])

            # # loss_epochs = np.array(loss_epochs[: cfg.pruning.trajectory])
            # loss_epochs = np.array(loss_epochs[-cfg.pruning.trajectory :])

            # # index_epochs = np.array(index_epochs[: cfg.pruning.trajectory])
            # index_epochs = np.array(index_epochs[-cfg.pruning.trajectory :])

            output_path = f"{cfg.paths.scores}/{cfg.dataset.name}_infobatch_{ratio}_{num_itr}"
            if cfg.dataset.for_extrapolation.value is True:
                output_path += f"_{cfg.dataset.for_extrapolation.subset_size}"

            date = datetime.datetime.now()
            output_path += f"_{date.month}_{date.day}"
            output_path += ".json"

            logger.info(f"Saved infobatch scores to {output_path}")

            if cfg.pruning.prune is True:
                prune(
                    trainset=trainset,
                    test_loader=test_loader,
                    scores_dict=trainset.scores,
                    cfg=cfg,
                    ratios=ratio,
                    wandb_name="infobatch-",
                    device=device,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run infobatch Pruning")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "configs", "infobatch_config.yaml"),
        help="Path to config file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["CIFAR10", "CIFAR100", "SYNTHETIC_CIFAR100_1M", "SYNTHETIC_CIFAR100_1M_LT", "PLACES_365", "IMAGENET"],
        help="Dataset to use for training"
    )
    
    args = parser.parse_args()
    
    main(cfg_path=args.config, dataset=args.dataset)