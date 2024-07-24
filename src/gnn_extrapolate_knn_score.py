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
        self.conv2 = GCNConv(512, 512)
        self.conv3 = GCNConv(512, 512)
        self.conv4 = GCNConv(512, 256)
        self.conv5 = GCNConv(256, output_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv4(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv5(x, edge_index, edge_attr)
        return x

def get_edges_and_attributes(embeddings, k=10, distance="euclidean"):
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(embeddings)
        
    distances, neighbors = nbrs.kneighbors(embeddings)
    
    edge_index = []
    edge_attr = []

    for i in range(neighbors.shape[0]):
        for j in range(1, k + 1):
            edge_index.append([i, neighbors[i, j]])
            edge_attr.append(1/distances[i, j])
            
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    return edge_index, edge_attr

def prepare_data(embeddings_dict, seed_samples, full_scores_dict, knn_extrapolated_score, k, distance="euclidean"):
    samples_list = [int(i) for i in embeddings_dict.keys()]
    
    train_samples_pos = [samples_list.index(i) for i in seed_samples]

    embeddings = np.array([embeddings_dict[i].cpu().numpy().flatten() for i in samples_list])

    edge_index, edge_attr = get_edges_and_attributes(embeddings, k, distance)

    true_scores = [full_scores_dict[str(i)] for i in samples_list]
    binary_features = [1 if i in seed_samples else 0 for i in samples_list]

    knn_scores = [knn_extrapolated_score[str(i)] for i in samples_list]
    
    # use the knn scores as features     
    node_features = torch.tensor(knn_scores, dtype=torch.float).unsqueeze(1)

    # node_features = np.vstack([knn_scores, binary_features]).T
    # node_features = torch.tensor(node_features, dtype=torch.float)
    
    y = torch.tensor(true_scores, dtype=torch.float)

    mask = torch.zeros(y.size(0), dtype=torch.bool)
    mask[train_samples_pos] = True
    
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=y, train_mask=mask)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainset, _ = get_dataset("SYNTHETIC_CIFAR100_1M", partial=True, subset_idxs=["0"])

    with open("/nfs/homedirs/dhp/unsupervised-data-pruning/scores/SYNTHETIC_CIFAR100_1M_dynamic_uncertainty_train_2.json") as f:
        full_scores_dict = json.load(f)

    with open("/nfs/homedirs/dhp/unsupervised-data-pruning/scores/SYNTHETIC_CIFAR100_1M_dynamic_uncertainty_knn_extrapolated.json") as f:
        knn_extrapolated_score = json.load(f)

    model_combination = ["resnet50-trained-to-compute-score", "resnet18", "resnet50"]
    k_combination = [50, 20, 50, 10]
    trainset_size = [100_000, 50_000, 20_000, 10_000]

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
        for i in tqdm(range(len(trainset)), mininterval=10, maxinterval=20):
            sample = trainset[i][0]
            sample_idx = trainset[i][2]
            sample = sample.to(device)
            sample = sample.unsqueeze(0)
            with torch.no_grad():
                embedding_val = embedding_model(sample)
            embeddings_dict[sample_idx] = embedding_val.cpu()
        
        for k in tqdm(k_combination):
            for num_seed in tqdm(trainset_size):
                    wandb.init(
                            project=f"GNN extrapolate", name=f"k-{k}-num_seed-{num_seed}-model-{model_name}"
                    )
                    train_samples = np.random.choice(list(embeddings_dict.keys()), num_seed, replace=False)
                    distance_metric = "euclidean"
                    data = prepare_data(embeddings_dict, train_samples, full_scores_dict, knn_extrapolated_score, k, distance_metric).to(device)
                    
                    model = GNN(input_dim=data.num_node_features, output_dim=1).to(device)

                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

                    out = model(data.x, data.edge_index, data.edge_attr).squeeze()

                    orig_train = data.y.cpu().detach().numpy()[data.train_mask.cpu().detach().numpy()]
                    pred_train = out.cpu().detach().numpy()[data.train_mask.cpu().detach().numpy()]

                    corr_train = np.corrcoef(orig_train, pred_train)[0, 1]
                    spearman_train = spearmanr(orig_train, pred_train).correlation

                    print(f"For seeded samples, Corr: {corr_train}, Spearman: {spearman_train}")

                    orig_test = data.y.cpu().detach().numpy()[~data.train_mask.cpu().detach().numpy()]
                    pred_test = out.cpu().detach().numpy()[~data.train_mask.cpu().detach().numpy()]
                    
                    corr_test = np.corrcoef(orig_test, pred_test)[0, 1]
                    spearman_test = spearmanr(orig_test, pred_test).correlation

                    print(f"For unseeded samples, Corr: {corr_test}, Spearman: {spearman_test}")

                    model.train()
                    for epoch in range(5000):
                        optimizer.zero_grad()
                        out = model(data.x, data.edge_index, data.edge_attr).squeeze()
                        loss = F.mse_loss(out[~data.train_mask], data.y[~data.train_mask])
                        loss.backward()
                        optimizer.step()
                        if epoch % 100 == 0:
                            print(f"GNN Epoch: {epoch}, Loss: {loss.item()}")
    
                        wandb.log({"Train Loss": loss.item()}, step=epoch)

                        orig_train = data.y.cpu().detach().numpy()[data.train_mask.cpu().detach().numpy()]
                        pred_train = out.cpu().detach().numpy()[data.train_mask.cpu().detach().numpy()]

                        corr_train = np.corrcoef(orig_train, pred_train)[0, 1]
                        spearman_train = spearmanr(orig_train, pred_train).correlation

                        wandb.log({"Corr Trainset": corr_test}, step=epoch)
                        wandb.log({"Spearman Trainset": spearman_test}, step=epoch)

                        orig_test = data.y.cpu().detach().numpy()[~data.train_mask.cpu().detach().numpy()]
                        pred_test = out.cpu().detach().numpy()[~data.train_mask.cpu().detach().numpy()]
                        
                        corr_test = np.corrcoef(orig_test, pred_test)[0, 1]
                        spearman_test = spearmanr(orig_test, pred_test).correlation

                        wandb.log({"Corr Testset": corr_train}, step=epoch)
                        wandb.log({"Spearman Testset": spearman_train}, step=epoch)

                    model.eval()
                    with torch.no_grad():
                        out = model(data.x, data.edge_index, data.edge_attr).squeeze()
                    
                    orig_test = data.y.cpu().detach().numpy()[~data.train_mask.cpu().detach().numpy()]
                    pred_test = out.cpu().detach().numpy()[~data.train_mask.cpu().detach().numpy()]
                    
                    corr_test = np.corrcoef(orig_test, pred_test)[0, 1]
                    spearman_test = spearmanr(orig_test, pred_test).correlation
                    
                    # check the correlation for the seed samples as well
                    orig_train = data.y.cpu().detach().numpy()[data.train_mask.cpu().detach().numpy()]
                    pred_train = out.cpu().detach().numpy()[data.train_mask.cpu().detach().numpy()]

                    corr_train = np.corrcoef(orig_train, pred_train)[0, 1]
                    spearman_train = spearmanr(orig_train, pred_train).correlation

                    print(f"For seeded samples, Corr: {corr_train}, Spearman: {spearman_train}")
                    results.append({
                        "model": model_name,
                        "k": k,
                        "num_seeds": num_seed,
                        "corr_avg": corr_test,
                        "spearman_avg": spearman_test,
                    })
                    wandb.finish()
                    print(f"Model: {model_name}, k: {k}, Num Seeds: {num_seed}, Distance: {distance_metric} , Corr Avg: {corr_test}, Spearman: {spearman_test}")

    results_df = pd.DataFrame(results)
    results_df.to_csv("/nfs/homedirs/dhp/unsupervised-data-pruning/data/gnn_extrapolation.csv", index=False)
