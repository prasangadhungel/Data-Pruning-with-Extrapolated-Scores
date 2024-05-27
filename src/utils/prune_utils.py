from types import SimpleNamespace

import numpy as np
import torch


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
    forget_stats.never_correct = np.setdiff1d(
        forget_stats.never_correct, idxs[accs.astype(bool)], True
    )
    return forget_stats


def get_embeddings(model, data_loader, device):
    model.eval()
    features = {}

    with torch.no_grad():
        for _, (data, _, sample_idx) in enumerate(data_loader):
            data = data.to(device)
            outputs = model(
                data
            )  # Replace with appropriate output extraction for your model

            for i, sample in enumerate(sample_idx):
                sample = sample.item()
                selected_features = outputs[i].detach().cpu().numpy()
                features[sample] = selected_features

    return features


def get_error(model, X, target, num_classes=10):
    target = torch.nn.functional.one_hot(target, num_classes=num_classes)
    preds = model(X)
    prob_preds = torch.nn.functional.softmax(preds, dim=-1)
    errors = prob_preds - target
    scores = torch.norm(errors, p=2, dim=-1)
    scores = scores.cpu().detach().numpy()
    return scores
