import json
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
from tqdm import tqdm
from utils.dataset import get_dataset
from utils.models import ResNetEmbedding, load_model
import wandb

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
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(embeddings)
    elif distance == "cosine":
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(embeddings)
        
    distances, neighbors = nbrs.kneighbors(embeddings)
    
    edge_index = []
    edge_attr = []

    for i in range(neighbors.shape[0]):
        for j in range(1, k + 1):  # Skip the first neighbor (itself)
            edge_index.append([i, neighbors[i, j]])
            edge_attr.append(distances[i, j])
            
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    return edge_index, edge_attr

def prepare_data(embeddings_dict, seed_samples, full_scores_dict, k, distance="euclidean"):
    samples_list = [int(i) for i in embeddings_dict.keys()]
    
    seed_samples_pos = [samples_list.index(i) for i in seed_samples]

    embeddings = np.array([embeddings_dict[i].cpu().numpy().flatten() for i in samples_list])
    y = [full_scores_dict[str(i)] for i in samples_list]
    y = torch.tensor(y, dtype=torch.float)
    
    edge_index, edge_attr = get_edges_and_attributes(embeddings, k, distance)
    x = torch.tensor(embeddings, dtype=torch.float)
    
    mask = torch.zeros(y.size(0), dtype=torch.bool)
    mask[seed_samples_pos] = True
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, train_mask=mask)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainset, _ = get_dataset("SYNTHETIC_CIFAR100_1M", partial=True, subset_idxs=["0"])

    with open("/nfs/homedirs/dhp/unsupervised-data-pruning/scores/SYNTHETIC_CIFAR100_1M_dynamic_uncertainty_train_2.json") as f:
        full_scores_dict = json.load(f)

    model_combination = ["resnet50-trained-to-compute-score", "resnet18", "resnet50"]
    k_combination = [5, 10, 50, 100, 500, 1000]
    num_seeds_combination = [1000, 10000, 20000, 50000, 100000]

    results = []

    for model_name in tqdm(model_combination):
        if model_name == "resnet50-trained-to-compute-score":
            model_path = "/nfs/homedirs/dhp/unsupervised-data-pruning/models/SYNTHETIC_CIFAR100_1M_ResNet50.pt"
            model = load_model("ResNet50", 100, model_path, device)
            embedding_model = ResNetEmbedding(model).to(device)
            embedding_model.eval()

        elif model_name == "resnet18":
            embedding_model = torchvision.models.resnet18(pretrained=True)
            embedding_model = embedding_model.to(device)
            embedding_model.eval()

        elif model_name == "resnet50":
            embedding_model = torchvision.models.resnet50(pretrained=True)
            embedding_model = embedding_model.to(device)
            embedding_model.eval()

        embeddings_dict = {}
        for i in tqdm(range(len(trainset))):
            # if i % 2 == 0:
                sample = trainset[i][0]
                sample_idx = trainset[i][2]
                sample = sample.to(device)
                sample = sample.unsqueeze(0)
                with torch.no_grad():
                    embedding_val = embedding_model(sample)
                embeddings_dict[sample_idx] = embedding_val.cpu()
        
        for k in tqdm(k_combination):
            for num_seed in tqdm(num_seeds_combination):
                    wandb.init(
                            project=f"GNN extrapolate", name=f"k-{k}-num_seed-{num_seed}-model-{model_name}"
                    )
                    seed_samples = np.random.choice(list(embeddings_dict.keys()), num_seed, replace=False)
                    distance_metric = "euclidean"
                    data = prepare_data(embeddings_dict, seed_samples, full_scores_dict, k, distance_metric).to(device)
                    
                    model = GNN(input_dim=data.num_node_features, output_dim=1).to(device)

                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

                    out = model(data.x, data.edge_index, data.edge_attr).squeeze()

                    orig_seed = data.y.cpu().detach().numpy()[data.train_mask.cpu().detach().numpy()]
                    pred_seed = out.cpu().detach().numpy()[data.train_mask.cpu().detach().numpy()]

                    corr_seed = np.corrcoef(orig_seed, pred_seed)[0, 1]
                    spearman_seed = spearmanr(orig_seed, pred_seed).correlation

                    print(f"For seeded samples, Corr: {corr_seed}, Spearman: {spearman_seed}")

                    orig_list = data.y.cpu().detach().numpy()[~data.train_mask.cpu().detach().numpy()]
                    pred_list = out.cpu().detach().numpy()[~data.train_mask.cpu().detach().numpy()]
                    
                    corr_avg = np.corrcoef(orig_list, pred_list)[0, 1]
                    spearman_avg = spearmanr(orig_list, pred_list).correlation


                    print(f"For unseeded samples, Corr: {corr_avg}, Spearman: {spearman_avg}")

                    model.train()
                    for epoch in range(10000):  # Number of epochs
                        optimizer.zero_grad()
                        out = model(data.x, data.edge_index, data.edge_attr).squeeze()
                        loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
                        loss.backward()
                        optimizer.step()
                        if i%100 == 0:
                            print(f"GNN Epoch: {epoch}, Loss: {loss.item()}")
    
                        wandb.log({"Train Loss": loss.item()}, step=epoch)

                        orig_seed = data.y.cpu().detach().numpy()[data.train_mask.cpu().detach().numpy()]
                        pred_seed = out.cpu().detach().numpy()[data.train_mask.cpu().detach().numpy()]

                        corr_seed = np.corrcoef(orig_seed, pred_seed)[0, 1]
                        spearman_seed = spearmanr(orig_seed, pred_seed).correlation

                        wandb.log({"Corr Seed": corr_seed}, step=epoch)
                        wandb.log({"Spearman Seed": spearman_seed}, step=epoch)

                        orig_list = data.y.cpu().detach().numpy()[~data.train_mask.cpu().detach().numpy()]
                        pred_list = out.cpu().detach().numpy()[~data.train_mask.cpu().detach().numpy()]
                        
                        corr_avg = np.corrcoef(orig_list, pred_list)[0, 1]
                        spearman_avg = spearmanr(orig_list, pred_list).correlation

                        wandb.log({"Corr Unseed": corr_avg}, step=epoch)
                        wandb.log({"Spearman Unseed": spearman_avg}, step=epoch)

                    model.eval()
                    with torch.no_grad():
                        out = model(data.x, data.edge_index, data.edge_attr).squeeze()
                    
                    orig_list = data.y.cpu().detach().numpy()[~data.train_mask.cpu().detach().numpy()]
                    pred_list = out.cpu().detach().numpy()[~data.train_mask.cpu().detach().numpy()]
                    
                    corr_avg = np.corrcoef(orig_list, pred_list)[0, 1]
                    spearman_avg = spearmanr(orig_list, pred_list).correlation
                    
                    # check the correlation for the seed samples as well
                    orig_seed = data.y.cpu().detach().numpy()[data.train_mask.cpu().detach().numpy()]
                    pred_seed = out.cpu().detach().numpy()[data.train_mask.cpu().detach().numpy()]

                    corr_seed = np.corrcoef(orig_seed, pred_seed)[0, 1]
                    spearman_seed = spearmanr(orig_seed, pred_seed).correlation

                    print(f"For seeded samples, Corr: {corr_seed}, Spearman: {spearman_seed}")
                    results.append({
                        "model": model_name,
                        "k": k,
                        "num_seeds": num_seed,
                        "corr_avg": corr_avg,
                        "spearman_avg": spearman_avg,
                    })
                    wandb.finish()
                    print(f"Model: {model_name}, k: {k}, Num Seeds: {num_seed}, Distance: {distance_metric} , Corr Avg: {corr_avg}, Spearman Avg: {spearman_avg}")

    results_df = pd.DataFrame(results)
    results_df.to_csv("/nfs/homedirs/dhp/unsupervised-data-pruning/data/gnn_extrapolation.csv", index=False)
