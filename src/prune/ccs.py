import datetime
import json
import logging
import os
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.optim import Adam
import random
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import argparse
from utils.dataset import prepare_data
from utils.evaluate import evaluate
from utils.helpers import parse_config, seed_everything
from utils.models import get_model
#from utils.prune_utils import get_error
from utils.ccs_coreset import prune
import wandb
import time

logger = logging.getLogger(__name__)


class TrainingDynamicsLogger(object):
    """
    Helper class for saving training dynamics for each iteration.
    Maintain a list containing output probability for each sample.
    """
    def __init__(self, filename=None):
        self.training_dynamics = []

    def log_tuple(self, tuple):
        self.training_dynamics.append(tuple)

    def save_training_dynamics(self, filepath, data_name=None):
        pickled_data = {
            'data-name': data_name,
            'training_dynamics': self.training_dynamics
        }

        with open(filepath, 'wb') as handle:
            pickle.dump(pickled_data, handle)


"""Calculate loss and entropy"""
def post_training_metrics(model, dataloader, data_importance, device):
    model.eval()
    data_importance['entropy'] = torch.zeros(len(dataloader.dataset))
    data_importance['loss'] = torch.zeros(len(dataloader.dataset))

    for batch_idx, (idx, (inputs, targets)) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        logits = model(inputs)
        prob = nn.Softmax(dim=1)(logits)

        entropy = -1 * prob * torch.log(prob + 1e-10)
        entropy = torch.sum(entropy, dim=1).detach().cpu()

        loss = nn.CrossEntropyLoss(reduction='none')(logits, targets).detach().cpu()

        data_importance['entropy'][idx] = entropy
        data_importance['loss'][idx] = loss


"""Calculate td metrics"""
def training_dynamics_metrics(td_log, dataset, data_importance):
    targets = []
    data_size = len(dataset)

    for i in range(data_size):
        _, (_, y) = dataset[i]
        targets.append(y)
    targets = torch.tensor(targets)
    data_importance['targets'] = targets.type(torch.int32)

    data_importance['correctness'] = torch.zeros(data_size).type(torch.int32)
    data_importance['forgetting'] = torch.zeros(data_size).type(torch.int32)
    data_importance['last_correctness'] = torch.zeros(data_size).type(torch.int32)
    data_importance['accumulated_margin'] = torch.zeros(data_size).type(torch.float32)

    def record_training_dynamics(td_log):
        output = torch.exp(td_log['output'].type(torch.float))
        predicted = output.argmax(dim=1)
        index = td_log['idx'].type(torch.long)

        label = targets[index]

        correctness = (predicted == label).type(torch.int)
        data_importance['forgetting'][index] += torch.logical_and(data_importance['last_correctness'][index] == 1, correctness == 0)
        data_importance['last_correctness'][index] = correctness
        data_importance['correctness'][index] += data_importance['last_correctness'][index]

        batch_idx = range(output.shape[0])
        target_prob = output[batch_idx, label]
        output[batch_idx, label] = 0
        other_highest_prob = torch.max(output, dim=1)[0]
        margin = target_prob - other_highest_prob
        data_importance['accumulated_margin'][index] += margin

    for i, item in enumerate(td_log):
        if i % 10000 == 0:
            print(i)
        record_training_dynamics(item)


def EL2N(td_log, dataset, data_importance, max_epoch=10):
    targets = []
    data_size = len(dataset)

    for i in range(data_size):
        _, (_, y) = dataset[i]
        targets.append(y)
    targets = torch.tensor(targets)
    data_importance['targets'] = targets.type(torch.int32)
    data_importance['el2n'] = torch.zeros(data_size).type(torch.float32)
    l2_loss = torch.nn.MSELoss(reduction='none')

    def record_training_dynamics(td_log):
        output = torch.exp(td_log['output'].type(torch.float))
        predicted = output.argmax(dim=1)
        index = td_log['idx'].type(torch.long)

        label = targets[index]

        label_onehot = torch.nn.functional.one_hot(label, num_classes=num_classes)
        el2n_score = torch.sqrt(l2_loss(label_onehot,output).sum(dim=1))

        data_importance['el2n'][index] += el2n_score

    for i, item in enumerate(td_log):
        if i % 10000 == 0:
            print(i)
        if item['epoch'] == max_epoch:
            return
        record_training_dynamics(item)



def main(cfg_path: str, dataset: str = None):
    seed_everything(42)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    cfg = OmegaConf.load(cfg_path)

    if dataset:
        cfg = cfg.get(dataset,cfg.CIFAR10)
    else:
        cfg = cfg.CIFAR10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainset, train_loader, test_loader, num_train_examples = prepare_data(
        cfg.dataset, cfg.training.batch_size
    )
    logger.info(f"Loaded Dataset: {cfg.dataset.name}, device: {device}")

    for num_itr in range(cfg.experiment.num_iterations):
        #el2n_scores = {i: [] for i in range(num_train_examples)}
        #torch.cuda.empty_cache()
        start_time = time.time()
        td_logger = TrainingDynamicsLogger()
        
        # model = get_model(
        #     model_name=cfg.model.name, num_classes=cfg.dataset.num_classes
        # ).to(device)
        # optimizer = Adam(model.parameters(), lr=cfg.training.lr)
        model_idx=0

        model = get_model(
            model_name=cfg.model.name,
            num_classes=cfg.dataset.num_classes,
            image_size=cfg.dataset.image_size,
        ).to(device)

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.training.lr,
            momentum=cfg.training.momentum,
            weight_decay=cfg.training.weight_decay,
            nesterov=cfg.training.nesterov,
        )

        num_iter = len(train_loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=cfg.pruning.num_epochs * num_iter
        )

        criterion = torch.nn.CrossEntropyLoss()
        model.cuda()
        criterion.cuda()

        for epoch in range(cfg.pruning.num_epochs):
            train_losses = []
            for batch_idx, (data, target, sample_idx) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = torch.nn.functional.cross_entropy(output, target)

                optimizer.zero_grad()
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                
                log_tuple = {
                'epoch': epoch,
                'iteration': batch_idx,
                'idx': sample_idx.type(torch.long).clone(),
                'output': torch.nn.functional.log_softmax(output, dim=1).detach().cpu().type(torch.half)
                }
                td_logger.log_tuple(log_tuple)

            test_acc = evaluate(model, test_loader, device)
            train_loss = torch.stack(train_losses).mean().item()
            logger.info(
                f"Model - {model_idx+1}, Epoch {epoch + 1}, Train Loss: {train_loss}, Test Accuracy: {test_acc}"
            )

        data_importance = {}
        post_training_metrics(model, train_loader, data_importance, device)
        training_dynamics_metrics(training_dynamics, trainset, data_importance)
        EL2N(training_dynamics, trainset, data_importance, max_epoch=10)

        # for data, target, sample_idx in train_loader:
        #     data, target = data.to(device), target.to(device)
        #     scores = get_error(
        #         model, data, target, num_classes=cfg.dataset.num_classes
        #     )
        #     for i, sample in enumerate(sample_idx):
        #         sample = sample.item()
        #         el2n_scores[sample].append((scores[i],target.cpu().numpy()))

        # Take average of scores
        # el2n_values = {
        #     sample: np.mean(scores).item() for sample, (scores,target) in el2n_scores.items(),
        #     "targets": target for sample, (scores,target) in el2n_scores.items()
        # }


        end_time = time.time()
        training_time = end_time - start_time
        logger.info(f"Training time: {training_time:.2f} seconds")

        date = datetime.datetime.now()
        output_path = f"{cfg.paths.scores}/{cfg.dataset.name}_el2n_score_{num_itr}_{date.month}_{date.day}.json"
        with open(output_path, "w") as f:
            json.dump(el2n_values, f)


        logger.info(f"Saved EL2N scores to {output_path}")

        prune_cs(
            trainset=trainset,
            test_loader=test_loader,
            scores_dict=data_importance,
            cfg=cfg,
            wandb_name="el2n",
            device=device,
            sampling_method="ccs"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ccs Pruning")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "configs", "ccs_config.yaml"),
        help="Path to config file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["CIFAR10", "CIFAR100", "SYNTHETIC_CIFAR100_1M"],
        help="Dataset to use for training"
    )
    
    args = parser.parse_args()
    
    main(cfg_path=args.config, dataset=args.dataset)
