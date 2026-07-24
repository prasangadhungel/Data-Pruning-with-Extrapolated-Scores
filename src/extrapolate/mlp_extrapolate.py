import datetime
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger
from omegaconf import OmegaConf
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.dataset import prepare_data
from utils.helpers import parse_config, seed_everything
from utils.models import load_model_by_name

logger.remove()
logger.add(sys.stdout, format="{time:MM-DD HH:mm} - {message}")


class MLP(torch.nn.Module):
    def __init__(self, input_dim, dropout=0.5):
        super(MLP, self).__init__()
        # 3 Layer NN: Input -> 256 -> 64 -> 1
        self.fc1 = torch.nn.Linear(input_dim, 256)
        self.fc2 = torch.nn.Linear(256, 64)
        self.fc3 = torch.nn.Linear(64, 1)
        self.dropout = dropout

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.fc3(x)
        return x


def prepare_features(
    embeddings,
    labels_dict,
    num_classes,
    use_labels=True,
    device="cuda",
):
    """
    Prepares the input feature matrix X.
    """
    samples_list = list(range(len(embeddings)))
    labels = [labels_dict[i] for i in samples_list]

    if use_labels:
        # Concatenate One-Hot labels with Embeddings
        one_hot = torch.eye(num_classes)[labels].to(device)
        x = torch.cat((one_hot, embeddings), dim=1)
    else:
        x = embeddings

    return x


@torch.no_grad()
def evaluate(
    model,
    x,
    y_subset, 
    y_full,
    train_mask,
    val_mask,
    test_mask,
    device,
):
    model.eval()
    # Predict on all nodes
    out = model(x).squeeze()
    
    # Convert to CPU numpy for metric calculation
    all_preds = out.detach()
    all_preds_np = all_preds.cpu().numpy()
    
    # --- Train Evaluation (vs Subset Scores) ---
    pred_train = all_preds_np[train_mask.cpu().numpy()]
    orig_train = y_subset[train_mask].cpu().numpy()
    
    corr_train = np.corrcoef(orig_train, pred_train)[0, 1] if len(pred_train) > 1 else 0
    spearman_train = spearmanr(orig_train, pred_train).correlation
    mse_train = np.mean((orig_train - pred_train) ** 2)

    # --- Val Evaluation (vs Subset Scores) ---
    pred_val = all_preds_np[val_mask.cpu().numpy()]
    orig_val = y_subset[val_mask].cpu().numpy()
    
    corr_val = np.corrcoef(orig_val, pred_val)[0, 1] if len(pred_val) > 1 else 0
    spearman_val = spearmanr(orig_val, pred_val).correlation
    mse_val = np.mean((orig_val - pred_val) ** 2)

    # --- Test Evaluation (vs Full Ground Truth Scores) ---
    # Test mask here usually corresponds to unseed samples (extrapolation targets)
    pred_test = all_preds_np[test_mask.cpu().numpy()]
    orig_test = y_full[test_mask].cpu().numpy()
    
    corr_test = np.corrcoef(orig_test, pred_test)[0, 1] if len(pred_test) > 1 else 0
    spearman_test = spearmanr(orig_test, pred_test).correlation
    mse_test = np.mean((orig_test - pred_test) ** 2)

    return (
        all_preds,
        corr_train, spearman_train, mse_train,
        corr_val, spearman_val, mse_val,
        corr_test, spearman_test, mse_test,
    )


def main(cfg_path: str):
    seed_everything(42)

    cfg = OmegaConf.load(cfg_path)
    cfg = cfg.CIFAR10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Dataset Info
    trainset, train_loader, _, num_samples = prepare_data(cfg.dataset, 1024)
    logger.info(f"Loaded dataset: {cfg.dataset.name}, Device: {device}")

    # 2. Load Scores
    with open(cfg.scores.original_scores_file) as f:
        full_scores_dict = json.load(f)

    with open(cfg.scores.subset_scores_file) as f:
        subset_scores_dict = json.load(f)

    logger.info(f"Number of samples in subset: {len(subset_scores_dict)}")

    # Calculate theoretical max correlation (Subset vs Full)
    subset_keys = [k for k in full_scores_dict.keys() if k in subset_scores_dict]
    subset_scores_np = np.array([subset_scores_dict[k] for k in subset_keys])
    full_scores_np_subset = np.array([full_scores_dict[k] for k in subset_keys])

    corr = np.corrcoef(full_scores_np_subset, subset_scores_np)[0, 1]
    spearman = spearmanr(full_scores_np_subset, subset_scores_np).correlation
    mse = np.mean((full_scores_np_subset - subset_scores_np) ** 2)
    logger.info(f"Max achievable correlation (on subset): {corr:.4f} Spearman: {spearman:.4f} MSE: {mse:.4f}")

    # 3. Prepare Labels and Indices
    labels_tensor = torch.zeros(num_samples, dtype=torch.int64, device=device)
    seed_samples = [int(key) for key in subset_scores_dict.keys()]
    num_seed = len(seed_samples)
    unseed_samples = [i for i in range(num_samples) if i not in seed_samples]

    # Prepare Target Tensors
    # y_subset contains values for seed samples, 0.0 for others (used for training)
    y_subset_list = []
    for i in range(num_samples):
        y_subset_list.append(subset_scores_dict.get(str(i), 0.0))
    y_subset_tensor = torch.tensor(y_subset_list, dtype=torch.float, device=device)

    # y_full contains ground truth for everyone (used for final test eval)
    y_full_list = [full_scores_dict.get(str(i), 0.0) for i in range(num_samples)]
    y_full_tensor = torch.tensor(y_full_list, dtype=torch.float, device=device)

    # 4. Extract or Load Embeddings
    for model_name in tqdm(cfg.models.names):
        if cfg.checkpoints.read_embeddings:
            embeddings = torch.load(cfg.checkpoints.embeddings_file, map_location=device)
            logger.info(f"Loaded embeddings from {cfg.checkpoints.embeddings_file}")
            
            # Need to populate labels_tensor even if loading embeddings
            for _, labels, sample_idxs in train_loader:
                labels = labels.to(device)
                sample_idxs = sample_idxs.to(device)
                labels_tensor[sample_idxs] = labels
        else:
            # Extraction logic
            embedding_model = load_model_by_name(
                model_name,
                cfg.dataset.num_classes,
                cfg.dataset.image_size,
                cfg.models.resnet50.path,
                device,
            )
            embedding_model.eval()

            # Get embedding dimension
            sample_input, _, _ = trainset[0]
            sample_input = sample_input.unsqueeze(0).to(device)
            with torch.no_grad():
                sample_output = embedding_model(sample_input)
            embedding_dim = sample_output.shape[1]

            embeddings = torch.zeros(num_samples, embedding_dim, device=device)

            for images, labels, sample_idxs in tqdm(train_loader, mininterval=20, maxinterval=40):
                images = images.to(device)
                sample_idxs = sample_idxs.to(device)
                labels = labels.to(device)
                with torch.no_grad():
                    batch_embeddings = embedding_model(images)

                embeddings[sample_idxs] = batch_embeddings
                labels_tensor[sample_idxs] = labels

            if cfg.checkpoints.save_embeddings:
                torch.save(embeddings, cfg.checkpoints.embeddings_file)

        # 5. Prepare Input Features (Embeddings + Optional Labels)
        # We treat the whole dataset as one big tensor X
        X = prepare_features(
            embeddings,
            labels_tensor.cpu().numpy(), # pass as list/array or handle in function
            cfg.dataset.num_classes,
            use_labels=cfg.hyperparams.use_labels,
            device=device
        )
        logger.info(f"Feature Matrix Shape: {X.shape}")

        # 6. Create Splits (Train/Val from Seed, Test from Unseed)
        val_frac = 0.1 
        val_idxs = random.sample(seed_samples, int(val_frac * len(seed_samples)))
        train_idxs = [i for i in seed_samples if i not in val_idxs]
        test_idxs = unseed_samples # Extrapolation targets

        train_mask = torch.zeros(num_samples, dtype=torch.bool, device=device)
        train_mask[train_idxs] = True
        
        val_mask = torch.zeros(num_samples, dtype=torch.bool, device=device)
        val_mask[val_idxs] = True
        
        test_mask = torch.zeros(num_samples, dtype=torch.bool, device=device)
        test_mask[test_idxs] = True

        # Create DataLoader for Training (only on train_idxs)
        train_X = X[train_mask]
        train_y = y_subset_tensor[train_mask]
        
        train_dataset = TensorDataset(train_X, train_y)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=cfg.hyperparams.batch_size, 
            shuffle=True
        )

        # 7. Initialize Model
        model = MLP(input_dim=X.shape[1], dropout=0.5).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.hyperparams.lr)

        logger.info("Starting MLP Training...")

        # Initial Eval
        (_, _, _, _, _, _, _, corr_test, spearman_test, mse_test) = evaluate(
            model, X, y_subset_tensor, y_full_tensor, train_mask, val_mask, test_mask, device
        )
        logger.info(f"[Before training] Test - Corr: {corr_test:.3f} Spearman: {spearman_test:.3f} MSE: {mse_test:.4f}")

        best_val_corr = -1
        best_test_corr = -1
        best_test_spearman = -1
        saved_scores = subset_scores_dict.copy()

        # 8. Training Loop
        for epoch in range(cfg.hyperparams.epochs):
            model.train()
            train_losses = []

            for batch_x, batch_y in train_dataloader:
                optimizer.zero_grad()
                out = model(batch_x).squeeze()
                loss = F.mse_loss(out, batch_y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            avg_loss = np.mean(train_losses)
            logger.info(f"Epoch: {epoch}, Loss: {avg_loss:.5f}")

            # Evaluate
            (
                all_preds,
                corr_train, spearman_train, mse_train,
                corr_val, spearman_val, mse_val,
                corr_test, spearman_test, mse_test,
            ) = evaluate(
                model, X, y_subset_tensor, y_full_tensor, train_mask, val_mask, test_mask, device
            )

            logger.info(f"Step={epoch} | Val Corr: {corr_val:.3f} | Test Corr: {corr_test:.3f} Spearman: {spearman_test:.3f} MSE: {mse_test:.4f}")

            # Save Best Model logic based on Validation Correlation
            if corr_val > best_val_corr:
                logger.info(f"New best val corr: {corr_val:.4f} (prev: {best_val_corr:.4f})")
                best_val_corr = corr_val
                best_test_corr = corr_test
                best_test_spearman = spearman_test

                # Update saved scores with predictions for unseed samples
                for i in unseed_samples:
                    saved_scores[str(i)] = all_preds[i].item()

        # 9. Final Results & Saving
        logger.info(f"Best Test Corr: {best_test_corr} Spearman: {best_test_spearman}")

        # Calculate final extrapolation stats on full dataset
        extrapolated_scores = np.array([saved_scores[str(i)] for i in range(num_samples)])
        full_scores_final = np.array([full_scores_dict[str(i)] for i in range(num_samples)])
        
        corr_extrapolated = np.corrcoef(full_scores_final, extrapolated_scores)[0, 1]
        spearman_extrapolated = spearmanr(full_scores_final, extrapolated_scores).correlation
        
        logger.info(f"Final Extrapolated Corr: {corr_extrapolated} Spearman: {spearman_extrapolated}")

        filename = f"{cfg.output.gnn_dict_path}_{cfg.scores.type}_{cfg.dataset.name}_{model_name}_MLP_seed_{num_seed}"
        date = datetime.datetime.now()
        filename += f"_{date.month}_{date.day}.json"

        with open(filename, "w") as f:
            json.dump(saved_scores, f)

        logger.info(f"Saved extrapolated scores to {filename}")


if __name__ == "__main__":
    # Assuming you keep the same config layout
    default_config_path = os.path.join(
        os.path.dirname(__file__), "configs", "mlp_config.yaml"
    )
    config_path = parse_config(
        default_config=default_config_path, description="Run MLP Extrapolation"
    )
    main(cfg_path=config_path)