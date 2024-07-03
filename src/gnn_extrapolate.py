import json
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from scipy.spatial import distance_matrix
from scipy.stats import spearmanr
from tqdm import tqdm
from utils.dataset import get_dataset
from utils.models import ResNetEmbedding, load_model

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def get_edges(embeddings, k=10):
    dist_matrix = distance_matrix(embeddings, embeddings)
    edges = []
    for i in range(len(embeddings)):
        neighbors = np.argsort(dist_matrix[i])[:k+1]  # +1 because the closest neighbor is the point itself
        for neighbor in neighbors:
            if i != neighbor:  # skip self-loops
                edges.append((i, neighbor))
    return torch.tensor(edges).t().contiguous()

def prepare_data(embeddings_dict, seed_samples, full_scores_dict, k):
    samples_list = [int(i) for i in embeddings_dict.keys()]
    unseeded_samples = [i for i in samples_list if i not in seed_samples]
    
    embeddings = [embeddings_dict[i].cpu().numpy().flatten() for i in samples_list]
    y = [full_scores_dict[str(i)] for i in samples_list]
    y = torch.tensor(y, dtype=torch.float)
    
    edge_index = get_edges(embeddings, k)
    x = torch.tensor(embeddings, dtype=torch.float)
    
    mask = torch.zeros(y.size(0), dtype=torch.bool)
    mask[seed_samples] = True
    
    return Data(x=x, edge_index=edge_index, y=y, train_mask=mask)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainset, _ = get_dataset("SYNTHETIC_CIFAR100_1M", partial=True, subset_idxs=["50k"])

with open("/nfs/homedirs/dhp/unsupervised-data-pruning/scores/SYNTHETIC_CIFAR100_1M_dynamic_uncertainty_train_2.json") as f:
    full_scores_dict = json.load(f)

model_combination = ["resnet18"]
k_combination = [5, 20, 50]
num_seeds_combination = [1000, 10000, 20000]
distance_combination = ["cosine", "euclidean"]

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
        sample = trainset[i][0]
        sample_idx = trainset[i][2]
        sample = sample.to(device)
        sample = sample.unsqueeze(0)
        with torch.no_grad():
            embedding_val = embedding_model(sample)
        embeddings_dict[sample_idx] = embedding_val.cpu()
    
    for k in tqdm(k_combination):
        for num_seed in tqdm(num_seeds_combination):
            seed_samples = np.random.choice(list(embeddings_dict.keys()), num_seed, replace=False)
            data = prepare_data(embeddings_dict, seed_samples, full_scores_dict, k).to(device)
            
            model = GNN(input_dim=data.num_node_features, hidden_dim=64, output_dim=1).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            model.train()
            for epoch in range(200):  # Number of epochs
                optimizer.zero_grad()
                out = model(data.x, data.edge_index).squeeze()
                loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index).squeeze()
            
            orig_list = data.y.cpu().numpy()[~data.train_mask.cpu().numpy()]
            pred_list = out.cpu().numpy()[~data.train_mask.cpu().numpy()]
            
            corr_avg = np.corrcoef(orig_list, pred_list)[0, 1]
            spearman_avg = spearmanr(orig_list, pred_list).correlation
            
            results.append({
                "model": model_name,
                "num_seeds": num_seed,
                "corr_avg": corr_avg,
                "spearman_avg": spearman_avg
            })
            print(f"Model: {model_name}, Num Seeds: {num_seed}, Corr Avg: {corr_avg}, Spearman Avg: {spearman_avg}")

results_df = pd.DataFrame(results)
results_df.to_csv("/nfs/homedirs/dhp/unsupervised-data-pruning/data/gnn_extrapolation.csv", index=False)
