import json
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from tqdm import tqdm

import wandb
from prune.utils.argparse import parse_config
from prune.utils.dataset import get_dataset
from prune.utils.models import load_model_by_name

logger = logging.getLogger(__name__)


class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim):
        super(GNN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_layers[0]))

        for i in range(len(hidden_layers) - 1):
            self.convs.append(GCNConv(hidden_layers[i], hidden_layers[i + 1]))

        self.convs.append(GCNConv(hidden_layers[-1], output_dim))

    def forward(self, x, edge_index, edge_attr):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)

        x = self.convs[-1](x, edge_index, edge_attr)
        return x


class GNNFactory:
    @staticmethod
    def create_gnn(gnn_type, input_dim, hidden_layers, output_dim):
        if gnn_type == "gcn":
            return GNN(input_dim, hidden_layers, output_dim)
        elif gnn_type == "gat":
            from torch_geometric.nn import GATConv

            class GAT(torch.nn.Module):
                def __init__(self, input_dim, hidden_layers, output_dim):
                    super(GAT, self).__init__()
                    self.convs = torch.nn.ModuleList()
                    self.convs.append(GATConv(input_dim, hidden_layers[0]))
                    for i in range(len(hidden_layers) - 1):
                        self.convs.append(
                            GATConv(hidden_layers[i], hidden_layers[i + 1])
                        )
                    self.convs.append(GATConv(hidden_layers[-1], output_dim))

                def forward(self, x, edge_index, edge_attr):
                    for conv in self.convs[:-1]:
                        x = conv(x, edge_index)
                        x = F.relu(x)
                    x = self.convs[-1](x, edge_index)
                    return x

            return GAT(input_dim, hidden_layers, output_dim)
        elif gnn_type == "sage":
            from torch_geometric.nn import SAGEConv

            class GraphSAGE(torch.nn.Module):
                def __init__(self, input_dim, hidden_layers, output_dim):
                    super(GraphSAGE, self).__init__()
                    self.convs = torch.nn.ModuleList()
                    self.convs.append(SAGEConv(input_dim, hidden_layers[0]))
                    for i in range(len(hidden_layers) - 1):
                        self.convs.append(
                            SAGEConv(hidden_layers[i], hidden_layers[i + 1])
                        )
                    self.convs.append(SAGEConv(hidden_layers[-1], output_dim))

                def forward(self, x, edge_index):
                    for conv in self.convs[:-1]:
                        x = conv(x, edge_index)
                        x = F.relu(x)
                    x = self.convs[-1](x, edge_index)
                    return x

            return GraphSAGE(input_dim, hidden_layers, output_dim)
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")


def get_edges_and_attributes(embeddings, k=10, distance="euclidean"):
    if distance == "euclidean":
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(
            embeddings
        )
    elif distance == "cosine":
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(
            embeddings
        )

    distances, neighbors = nbrs.kneighbors(embeddings)

    edge_index = []
    edge_attr = []

    for i in range(neighbors.shape[0]):
        for j in range(1, k + 1):
            edge_index.append([i, neighbors[i, j]])
            edge_attr.append(np.exp(-distances[i, j]))

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return edge_index, edge_attr


def prepare_data(
    embeddings_dict, seed_samples, full_scores_dict, k, distance="euclidean"
):
    samples_list = [int(i) for i in embeddings_dict.keys()]
    seed_samples_pos = [samples_list.index(i) for i in seed_samples]

    embeddings = np.array(
        [embeddings_dict[i].cpu().numpy().flatten() for i in samples_list]
    )
    y = [full_scores_dict[str(i)] for i in samples_list]
    y = torch.tensor(y, dtype=torch.float)

    edge_index, edge_attr = get_edges_and_attributes(embeddings, k, distance)
    x = torch.tensor(embeddings, dtype=torch.float)

    mask = torch.zeros(y.size(0), dtype=torch.bool)
    mask[seed_samples_pos] = True

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, train_mask=mask)


def main(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainset, _ = get_dataset(
        cfg.dataset.name,
        partial=cfg.dataset.partial,
        subset_idxs=cfg.dataset.subset_idxs,
    )

    with open(cfg.dataset.scores_file) as f:
        full_scores_dict = json.load(f)

    results = []

    for model_name in tqdm(cfg.models.names):
        embedding_model = load_model_by_name(
            model_name, device, cfg.models.resnet50.path
        )
        embedding_model.eval()
        embeddings_dict = {}
        for i in tqdm(range(len(trainset)), mininterval=10, maxinterval=20):
            sample, _, sample_idx = trainset[i]
            sample = sample.to(device).unsqueeze(0)
            with torch.no_grad():
                embedding_val = embedding_model(sample)
            embeddings_dict[sample_idx] = embedding_val.cpu()

        for k in tqdm(cfg.hyperparams.k_values):
            for num_seed in tqdm(cfg.hyperparams.num_seeds):
                wandb.init(
                    project=cfg.wandb.project,
                    name=f"k-{k}-num_seed-{num_seed}-model-{model_name}",
                )
                wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
                seed_samples = np.random.choice(
                    list(embeddings_dict.keys()), num_seed, replace=False
                )
                data = prepare_data(
                    embeddings_dict,
                    seed_samples,
                    full_scores_dict,
                    k,
                    cfg.hyperparams.distance,
                ).to(device)

                gnn_model = GNNFactory.create_gnn(
                    cfg.hyperparams.gnn.type,
                    input_dim=data.num_node_features,
                    hidden_layers=cfg.hyperparams.gnn.hidden_layers,
                    output_dim=1,
                ).to(device)

                optimizer = torch.optim.Adam(
                    gnn_model.parameters(), lr=cfg.hyperparams.lr
                )

                for epoch in range(cfg.hyperparams.epochs):
                    gnn_model.train()
                    optimizer.zero_grad()
                    out = gnn_model(data.x, data.edge_index, data.edge_attr).squeeze()
                    loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
                    loss.backward()
                    optimizer.step()

                    if epoch % 200 == 0:
                        logger.info(f"GNN Epoch: {epoch}, Loss: {loss.item()}")

                    wandb.log({"Train Loss": loss.item()}, step=epoch)

                gnn_model.eval()
                with torch.no_grad():
                    out = gnn_model(data.x, data.edge_index, data.edge_attr).squeeze()

                corr_avg = np.corrcoef(
                    out[~data.train_mask].cpu().numpy(),
                    data.y[~data.train_mask].cpu().numpy(),
                )[0, 1]
                spearman_avg = spearmanr(
                    out[~data.train_mask].cpu().numpy(),
                    data.y[~data.train_mask].cpu().numpy(),
                ).correlation

                results.append(
                    {
                        "model": model_name,
                        "k": k,
                        "num_seeds": num_seed,
                        "corr_avg": corr_avg,
                        "spearman_avg": spearman_avg,
                    }
                )
                wandb.finish()

    results_df = pd.DataFrame(results)
    results_df.to_csv(cfg.output.results_file, index=False)


if __name__ == "__main__":
    default_config_path = os.path.join(
        os.path.dirname(__file__), "configs", "gnn_config.yaml"
    )
    config_path = parse_config(
        default_config=default_config_path, description="Run GNN Extrapolation"
    )
    main(cfg_path=config_path)
