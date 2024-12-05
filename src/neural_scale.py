import json
import logging

import hydra
import torch
from omegaconf import DictConfig
from sklearn.cluster import KMeans

from utils.dataset import prepare_data
from utils.prune_utils import get_embeddings, prune

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="neural_scale_config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainset, train_loader, test_loader = prepare_data(
        cfg.dataset, cfg.training.batch_size, embedding=True
    )
    logger.info(f"loaded dataset: {cfg.dataset.name}, device: {device}")

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
        distances = {}
        for i, embedding in enumerate(embeddings):
            distances[i] = (
                1
                - torch.nn.functional.cosine_similarity(
                    embedding.unsqueeze(0),
                    torch.tensor(kmeans.cluster_centers_[kmeans.labels_[i]]),
                    dim=1,
                ).item()
            )

        # Save distances as JSON
        with open(
            f"{cfg.paths.scores}/{cfg.dataset.name}_embedding_distances_{num_itr}.json",
            "w",
        ) as f:
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
    main()
