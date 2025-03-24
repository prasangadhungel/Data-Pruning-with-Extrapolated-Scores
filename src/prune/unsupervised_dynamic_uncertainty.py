import argparse
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import json
import datetime
import sys
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as dsets
import torchvision.transforms as transforms

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from torchvision.datasets import Places365

from loguru import logger

logger.remove()
logger.add(sys.stdout, format="{time:MM-DD HH:mm} - {message}")

datasets_to_c = {
    "CIFAR10": 10,
    "CIFAR100": 100,
    "SYNTHETIC_CIFAR100_1M": 100,
    "PLACES_365": 365
}

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def _parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default= "PLACES_365", help="Dataset to precompute embeddings")
    parser.add_argument('--phis', type=str, default="dinov2", help="Representation spaces to precompute", 
                        choices=['dinov2'])
    parser.add_argument('--batch_size_repr', type=int, default=256)
    parser.add_argument('--root_dir', type=str, default="data", help='Root dir to store everything')
    parser.add_argument('--device', type=str, default="cuda", help="cuda or cpu")
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gamma', type=float, default=10., help='Hyperparameter for entropy regularization in Eq. (12)')
    parser.add_argument('--T', type=int, default=6000, help='Number of outer iterations to train task encoder')
    parser.add_argument('--inner_lr', type=float, default=0.001, help='Learning rate for inner loop')
    parser.add_argument('--outer_lr', type=float, default=0.001, help='Learning rate for task encoder')
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--warm_start', action='store_true',
                        help="warm start = initialize inner learner from previous iteration, cold start = initialize randomly, cold-start is used by default") 
    parser.add_argument('--M', type=int, default=10, help='Number of inner steps at each outer iteration')
    
    return parser.parse_args(args)

def _convert_image_to_rgb(image):
    if torch.is_tensor(image):
        return image
    else:
        return image.convert("RGB")

def _safe_to_tensor(x):
    if torch.is_tensor(x):
        return x
    else:
        return transforms.ToTensor()(x)

def get_default_transforms():
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        _convert_image_to_rgb,
        _safe_to_tensor,
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])


class IndexDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return (item[0], item[1], idx)

class CustomDatasetWithIndices(Dataset):
    def __init__(self, images, labels, indices, transform=None):
        self.images = images
        self.labels = labels
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        index = self.indices[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, index

def get_datasets(dataset, transform, root_dir='./data'):
    data_path = os.path.join(root_dir, "datasets")

    if dataset == 'CIFAR10':
        train_dataset = dsets.CIFAR10(root=data_path, train=True, transform=transform, download=True)
        val_dataset = dsets.CIFAR10(root=data_path, train=False, transform=transform, download=True)
    elif dataset == 'CIFAR100':
        train_dataset = dsets.CIFAR100(root=data_path, train=True, transform=transform, download=True)
        val_dataset = dsets.CIFAR100(root=data_path, train=False, transform=transform, download=True)
    elif dataset == 'SYNTHETIC_CIFAR100_1M':
        data = np.load(
            "/nfs/homedirs/dhp/unsupervised-data-pruning/data/cifar100_1m.npz"
        )

        num_samples = len(data["label"])
        train_images = data["image"]
        train_labels = data["label"]

        indices = np.arange(num_samples)
        transform = get_default_transforms()
        train_dataset = CustomDatasetWithIndices(
            train_images, train_labels, indices, transform=transform
        )
        val_dataset = dsets.CIFAR100(root=data_path, train=False, transform=transform, download=True)
        val_dataset = IndexDataset(val_dataset)
        return train_dataset, val_dataset
    
    elif dataset == 'PLACES_365':
        root_dir = "/ceph/ssd/shared/datasets/places-365"
        transform = get_default_transforms()

        train_dataset = Places365(
            root=root_dir,
            split="train-standard",
            download=False,
            small=True,
            transform=transform,
        )
        val_dataset = Places365(
            root=root_dir,
            split="val",
            download=False,
            small=True,
            transform=transform
        )

    train_dataset = IndexDataset(train_dataset)
    val_dataset = IndexDataset(val_dataset)
    
    return train_dataset, val_dataset

def get_dataloaders(dataset, transform, batch_size, root_dir='data'):
    if transform is None:
        transform = get_default_transforms()
    train_dataset, val_dataset = get_datasets(dataset, transform, root_dir)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=10)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=10)
    return trainloader, valloader

def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_features(dataloader, model, device):
    features_dict = {}
    with torch.no_grad():
        for x, y, idx in tqdm(dataloader):
            features = model(x.to(device)).detach().cpu().numpy()
            for i, index in enumerate(idx.numpy()):
                features_dict[index] = features[i]
    
    sorted_indices = sorted(features_dict.keys())
    sorted_features = np.array([features_dict[i] for i in sorted_indices])
    return sorted_features

def run_representation(args=None):
    device = torch.device(args.device)

    torch.hub.set_dir("/nfs/homedirs/dhp/turtle/data/checkpoints/dinov2")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').to(device)
    model.eval()
    logger.info(f"Model parameters: {np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    preprocess = None

    trainloader, valloader = get_dataloaders(args.dataset, preprocess, args.batch_size_repr, args.root_dir)
    feats_train = get_features(trainloader, model, device)
    feats_val = get_features(valloader, model, device)

    return feats_train, feats_val


def get_labels(dataset):
    labels_dict = {}
    for idx in range(len(dataset)):
        _, label, index = dataset[idx]  # Assuming dataset returns (image, label, index)
        labels_dict[index] = label
    
    sorted_indices = sorted(labels_dict.keys())
    sorted_labels = np.array([labels_dict[i] for i in sorted_indices])
    return sorted_labels

def run_labels(args=None):
    train_dataset, val_dataset = get_datasets(args.dataset, None, args.root_dir)
    labels_train = get_labels(train_dataset)
    labels_val = get_labels(val_dataset)
    
    logger.info(f"Num train: {len(labels_train)}")
    logger.info(f"Num val: {len(labels_val)}")
    logger.info(f"Num classes: {len(np.unique(labels_train))}")

    return labels_train, labels_val

def get_cluster_acc(y_pred, y_true, return_matching=False):
    """
    Calculate clustering accuracy and clustering mean per class accuracy.
    Requires scipy installed
    # Arguments
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        y_true: true labels, numpy.array with shape `(n_samples,)`
    # Return
        Accuracy in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    match = np.array(list(map(lambda i: col_ind[i], y_pred)))

    mean_per_class = [0 for i in range(D)]
    for c in range(D):
        mask = y_true == c
        mean_per_class[c] = np.mean((match[mask] == y_true[mask]))
    mean_per_class_acc = np.mean(mean_per_class)

    if return_matching:
        return w[row_ind, col_ind].sum() / y_pred.size, mean_per_class_acc, match
    else:
        return w[row_ind, col_ind].sum() / y_pred.size, mean_per_class_acc


def compute_dynamic_uncertainty(uncertainty_history):
    uncertainty_scores = {}
    window_size = 10
    for sample_idx, history in uncertainty_history.items():
        std_devs = [np.std(history[i:i+window_size]) for i in range(len(history) - window_size)]
        uncertainty_scores[sample_idx] = sum(std_devs) / len(std_devs) if std_devs else 0
    return uncertainty_scores

def run_turtle(Zs_train, Zs_val, y_gt_val, args=None):
    logger.info("Running TURTLE with dynamic uncertainty")
    preprocess = None

    trainloader, valloader = get_dataloaders(args.dataset, preprocess, 10000, "data")
    epochs = 50

    logger.info("Loading dataset and representations")
    # Load pre-computed representations 
    logger.info(f'Load dataset {args.dataset}')
    logger.info(f'Representations of {args.phis}: ' + ' '.join(str(Z_train.shape) for Z_train in Zs_train))

    n_tr, C = Zs_train[0].shape[0], datasets_to_c[args.dataset]
    feature_dims = [Z_train.shape[1] for Z_train in Zs_train]
    batch_size = min(args.batch_size, n_tr)
    logger.info("Number of training samples:", n_tr)
    logger.info("Number of classes:", C)
    logger.info("Feature dimensions:", feature_dims)
    logger.info("Batch size:", batch_size)

    # Define task encoder
    task_encoder = [nn.utils.weight_norm(nn.Linear(d, C)).to(args.device) for d in feature_dims] 

    logger.info("Task encoder length:", len(task_encoder))
    logger.info("Task encoder first element:", task_encoder[0])

    def task_encoding(Zs):
        assert len(Zs) == len(task_encoder)
        # Generate labeling by the average of $\sigmoid(\theta \phi(x))$, Eq. (9) in the paper
        label_per_space = [F.softmax(task_phi(z), dim=1) for task_phi, z in zip(task_encoder, Zs)] # shape of (K, N, C)
        labels = torch.mean(torch.stack(label_per_space), dim=0) # shape of (N, C)
        return labels, label_per_space
    
    # we use Adam optimizer for faster convergence, other optimziers such as SGD could also work
    optimizer = torch.optim.Adam(sum([list(task_phi.parameters()) for task_phi in task_encoder], []), lr=args.outer_lr, betas=(0.9, 0.999))

    # Define linear classifiers for the inner loop
    def init_inner():
        W_in = [nn.Linear(d, C).to(args.device) for d in feature_dims] 
        inner_opt = torch.optim.Adam(sum([list(W.parameters()) for W in W_in], []), lr=args.inner_lr, betas=(0.9, 0.999))
    
        return W_in, inner_opt
    
    W_in, inner_opt = init_inner()
    
    uncertainty_history = {i: [] for i in range(n_tr)}
    
    for epoch in range(epochs):
        for _, _, idx in trainloader:
            indices = idx.numpy()
            Zs_tr = [torch.from_numpy(Z_train[indices]).to(args.device) for Z_train in Zs_train]
            labels, label_per_space = task_encoding(Zs_tr)
            
            if not args.warm_start:
                W_in, inner_opt = init_inner()
            
            for _ in range(args.M):
                inner_opt.zero_grad()
                loss = sum([F.cross_entropy(w_in(z_tr), labels.detach()) for w_in, z_tr in zip(W_in, Zs_tr)])
                loss.backward()
                inner_opt.step()
            
            optimizer.zero_grad()
            pred_error = sum([F.cross_entropy(w_in(z_tr).detach(), labels) for w_in, z_tr in zip(W_in, Zs_tr)])
            entr_reg = sum([torch.special.entr(l.mean(0)).sum() for l in label_per_space])
            (pred_error - args.gamma * entr_reg).backward()
            optimizer.step()
            
            for i, sample_idx in enumerate(indices):
                softmax_output = labels[i].detach().cpu().numpy()
                uncertainty_history[sample_idx].append(softmax_output)
        
        labels_val, _ = task_encoding([torch.from_numpy(Z_val).to(args.device) for Z_val in Zs_val])
        preds_val = labels_val.argmax(dim=1).detach().cpu().numpy()
        cluster_acc, _ = get_cluster_acc(preds_val, y_gt_val)
        
        logger.info(f'Epoch {epoch}/{epochs} Training loss {float(pred_error):.3f}, entropy {float(entr_reg):.3f}, cluster acc {cluster_acc:.4f}')
    
    logger.info(f'Training finished!')
    
    labels, _ = task_encoding([torch.from_numpy(Z_train).to(args.device) for Z_train in Zs_train])
    y_pred = labels.argmax(dim=-1).detach().cpu().numpy()
    
    # now from the uncertainty_history, we compute the softmax prediction
    # at the pseudo-labels
    softmax_pseudo_labels = {i: [] for i in range(n_tr)}
    for sample_idx, history in uncertainty_history.items():
        true_label = y_pred[sample_idx]
        for softmax_output in history:
            prediction = softmax_output[true_label]
            softmax_pseudo_labels[sample_idx].append(prediction)
    
    uncertainty_scores = compute_dynamic_uncertainty(softmax_pseudo_labels)

    output_path = f"/nfs/homedirs/dhp/unsupervised-data-pruning/scores/prune/{args.dataset}_unsupervised_dynamic_uncertainty"
    date = datetime.datetime.now()
    output_path += f"_{date.month}_{date.day}"
    output_path += ".json"

    with open(output_path, 'w') as f:
        json.dump(uncertainty_scores, f)

    logger.info("Saved dynamic uncertainty scores.")


def main(args=None):
    args = _parse_args(args)
    seed_everything(args.seed)
    Zs_train, Zs_val = run_representation(args)
    _, y_gt_val = run_labels(args)
    run_turtle([Zs_train], [Zs_val], y_gt_val, args)

if __name__ == "__main__":
    main()
    