# Effective Data Pruning through Score Extrapolation
Dataset pruning seeks to identify a minimal subset of training samples that preserves (or even improves) model performance. However, most existing algorithms require computing importance scores for every example by training on the full dataset and collecting statistics — a prohibitively expensive pre-pruning step.

Our approach avoids this overhead by first computing scores on a small subset of samples and then extrapolating those scores to the entire dataset. We explore two extrapolation strategies: KNN, and Graph Neural Networks (GNN)

The repository is organized as follows:
```
├── src
│   ├── extrapolate
│   ├── __init__.py    # Standard pruning algorithms (DU, TDDS, Forget, EL2N, Random)
│   ├── prune          # Score extrapolation methods (KNN, GNN)
│   └── utils          # Utilities (data loaders, models, evaluation etc)
```

All standard pruning methods (e.g., [Dynamic Uncertainty](https://arxiv.org/abs/2306.05175)) are under `src/prune`. Configuration files in `src/prune/configs/` allow you to control output directories, and tune algorithm-specific parameters for each dataset.

**Compute Full-Set Scores**

Open the desired config (e.g., `src/prune/configs/du_config.yaml`), and then set:

```yaml
dataset:
  for_extrapolation:
    value: false
```
Adjust any output paths for models and score files.

Run:
```bash
python src/prune/dynamic_uncertainty.py
```

Scores for each training sample will be saved as specified in your config.

Compute Subset Scores

In the same config file, set:
```
dataset:
  for_extrapolation:
    value: true
    subset_size: <N>
```

Update paths for the trained model and output score files.

Run the same command as above. This computes scores on a random subset of size N.

2. Score Extrapolation (src/extrapolate)

This module infers importance scores for the remaining samples using either KNN or GNN.

Configure Extrapolation

Open `src/extrapolate/configs/gnn_config.yaml` (or `knn_config.yaml`).

Set:
```
scores:
  subset_scores_file: path/to/subset_scores.json
  original_scores_file: path/to/full_scores.json  # only used for evaluation
models:
  resnet:
    path: path/to/pretrained_model.pth
output:
  gnn_dict_path: path/to/save/
```

Run Extrapolation
```bash
python src/extrapolate/gnn_extrapolate.py
```

This will save the extrapolated scores according to output.gnn_dict_path in your config.

3. Pruning with the Scores

Once you have a dictionary of scores (either original or extrapolated), you can perform pruning experiments:

Open `src/prune/configs/json_config.yaml` and set:
```
json_path: path/to/score_dict.json
```

Run:
```bash
python src/prune/prune_with_scores.py
```

The script will apply various pruning strategies using the provided scores and report performance metrics.
