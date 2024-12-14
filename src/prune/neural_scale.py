import json
import logging
import os

import torch
from omegaconf import OmegaConf
from sklearn.cluster import KMeans
from utils.argparse import parse_config
from utils.dataset import prepare_data
from utils.prune_utils import get_embeddings, prune

logger = logging.getLogger(__name__)


def main(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainset, train_loader, test_loader, _ = prepare_data(
        cfg.dataset, cfg.training.batch_size, embedding=True
    )
    logger.info(f"Loaded dataset: {cfg.dataset.name}, Device: {device}")

    for num_itr in range(cfg.experiment.num_iterations):

        # Load pre-trained self-supervised model (SWaV)
        model = torch.hub.load(
            "facebookresearch/swav:main", "resnet50", pretrained=True
        )
        model = model.to(device)

        # Get embeddings
        features = get_embeddings(model, train_loader, device)
        embeddings = torch.tensor([features[i] for i in range(len(trainset))])

        # Cluster the embeddings using KMeans
        kmeans = KMeans(n_clusters=cfg.pruning.num_clusters, random_state=cfg.seed)
        kmeans.fit(embeddings.numpy())

        # Compute distances for pruning scores
        distances = {
            i: (
                1
                - torch.nn.functional.cosine_similarity(
                    embedding.unsqueeze(0),
                    torch.tensor(kmeans.cluster_centers_[kmeans.labels_[i]]),
                    dim=1,
                ).item()
            )
            for i, embedding in enumerate(embeddings)
        }

        # Save distances as JSON
        output_path = (
            f"{cfg.paths.scores}/{cfg.dataset.name}_embedding_distances_{num_itr}.json"
        )
        with open(output_path, "w") as f:
            json.dump(distances, f)

        # Perform pruning and retraining for each prune percentage
        prune(
            trainset=trainset,
            test_loader=test_loader,
            scores_dict=distances,
            cfg=cfg,
            wandb_name="neural-scale-prune",
            device=device,
        )


if __name__ == "__main__":
    default_config_path = os.path.join(
        os.path.dirname(__file__), "configs", "neural_scale_config.yaml"
    )
    config_path = parse_config(
        default_config=default_config_path, description="Run Neural Scale Pruning"
    )
    main(cfg_path=config_path)
