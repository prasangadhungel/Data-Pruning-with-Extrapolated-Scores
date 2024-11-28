import json

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from omegaconf import DictConfig
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from tqdm import tqdm

import wandb
from utils.dataset import get_dataset
from utils.models import ResNetEmbedding, load_model


class GNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, 512)
        self.conv2 = GCNConv(512, 256)
        self.conv3 = GCNConv(256, output_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        return x


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


@hydra.main(config_path="configs", config_name="gnn_config")
def main(cfg: DictConfig):
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
        if model_name == "resnet50-self-trained":
            model = load_model("ResNet50", 100, cfg.models.resnet50.path, device)
            embedding_model = ResNetEmbedding(model).to(device)

        elif model_name == "resnet18":
            embedding_model = torchvision.models.resnet18(pretrained=True).to(device)

        elif model_name == "resnet50":
            embedding_model = torchvision.models.resnet50(pretrained=True).to(device)

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
                wandb.config.update(cfg)
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

                gnn_model = GNN(input_dim=data.num_node_features, output_dim=1).to(
                    device
                )
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

                    if epoch % 100 == 0:
                        print(f"GNN Epoch: {epoch}, Loss: {loss.item()}")

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
    main()
