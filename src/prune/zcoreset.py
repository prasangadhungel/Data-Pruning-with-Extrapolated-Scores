import datetime
import json
import os
import sys

import numpy as np
# If your CLIP is in a custom path, add it
# import clip
import open_clip
import torch
from loguru import logger
from omegaconf import OmegaConf
from torchvision import models
from torchvision.datasets import CIFAR10
from tqdm import tqdm

# Add your src folder for prune/utils
sys.path.append("/nfs/homedirs/dhp/unsupervised-data-pruning/src")

from utils.dataset import prepare_data
from utils.helpers import parse_config, seed_everything
from utils.prune_utils import prune

logger.remove()
logger.add(sys.stdout, format="{time:MM-DD HH:mm} - {message}")


def get_embeddings_clip(train_loader, device):
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    clip_model = clip_model.to(device)

    # clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    from torchvision.transforms import ToPILImage

    to_pil = ToPILImage()

    num_samples = len(train_loader.dataset)
    embedding_dim = clip_model.visual.output_dim
    embeddings = torch.zeros(num_samples, embedding_dim, device=device)

    with torch.no_grad():
        for images, _, sample_idxs in tqdm(
            train_loader, desc="Extracting CLIP embeddings"
        ):
            # Convert tensor image to PIL for preprocessing
            processed_images = torch.stack(
                [preprocess(to_pil(image.cpu())) for image in images]
            )
            processed_images = processed_images.to(device)
            batch_embeddings = clip_model.encode_image(processed_images).float()
            embeddings[sample_idxs] = batch_embeddings

    return embeddings.cpu().numpy()


def get_embeddings_resnet18(train_loader, device):
    # Load pretrained ResNet-18 and remove the final FC layer
    resnet18 = models.resnet18(pretrained=True)
    resnet18.fc = torch.nn.Identity()  # Penultimate layer output
    resnet18 = resnet18.to(device)
    resnet18.eval()

    num_samples = len(train_loader.dataset)
    embedding_dim = 512  # ResNet-18 penultimate layer
    embeddings = torch.zeros(num_samples, embedding_dim, device=device)

    with torch.no_grad():
        for images, _, sample_idxs in tqdm(
            train_loader, desc="Extracting ResNet18 embeddings"
        ):
            images = images.to(device)
            batch_embeddings = resnet18(images).float()  # (batch_size, 512)
            embeddings[sample_idxs] = batch_embeddings

    return embeddings.cpu().numpy()


# ---- ZCoreSet Scoring ----
def embedding_preprocess(embeddings):
    return {
        "n": len(embeddings),
        "n_dim": embeddings.shape[1],
        "min": np.min(embeddings, axis=0),
        "max": np.max(embeddings, axis=0),
        "med": np.median(embeddings, axis=0),
    }


def sample_score(args, embed_info, n_sample, n=0):
    global embeddings
    scores = np.zeros(embed_info["n"])
    for i in range(n_sample):
        dim = np.random.choice(embed_info["n_dim"], args["sample_dim"], replace=False)
        sample = np.random.triangular(
            embed_info["min"][dim], embed_info["med"][dim], embed_info["max"][dim]
        )
        embed_dist = np.sum(np.abs(embeddings[:, dim] - sample), axis=1)
        idx = np.argmin(embed_dist)
        scores[idx] += 1
        cover_sample = embeddings[idx, dim]
        nn_dist = np.sum(np.abs(embeddings[:, dim] - cover_sample), axis=1)
        nn = np.argsort(nn_dist)[1:]
        if nn_dist[nn[0]] == 0:
            scores[nn[0]] -= 1
        else:
            nn = nn[: args["redund_nn"]]
            dist_penalty = 1 / (nn_dist[nn] ** args["redund_exp"])
            dist_penalty /= np.sum(dist_penalty)
            scores[nn] -= dist_penalty
    return scores


def init_worker(embeddings_):
    global embeddings
    embeddings = embeddings_


def zcore_score(args, embeddings_):
    embed_info = embedding_preprocess(embeddings_)
    n_parallel_sample = int(args["n_sample"] / args["num_workers"])
    parallel_input = [
        (args, embed_info, n_parallel_sample, n) for n in range(args["num_workers"])
    ]
    import multiprocessing

    pool = multiprocessing.Pool(
        args["num_workers"], initializer=init_worker, initargs=(embeddings_,)
    )
    parallel_scores = pool.starmap(sample_score, parallel_input)
    pool.close()
    pool.join()
    if args["rand_init"]:
        scores = np.random.uniform(0, 1, embed_info["n"])
        for s in parallel_scores:
            scores += s
    else:
        scores = np.sum(parallel_scores, axis=0)
    score_min = np.min(scores)
    scores = (scores - score_min) / (np.max(scores) - score_min)
    return scores.astype(np.float32)


# ---- Main ZCoreSet Interface ----
def get_zcoreset_scores(cfg, train_loader, device):
    """
    Returns a dictionary mapping from dataset indices to ZCoreSet scores.
    Only CLIP implemented, but you can add other models as needed.
    """
    embedding_models = cfg.zcoreset.get("embedding", ["resnet18", "clip"])
    embeddings_list = []
    for model_name in embedding_models:
        if model_name == "clip":
            emb = get_embeddings_clip(train_loader, device)
            logger.info(f"Extracted embeddings using {model_name} : {emb.shape}")
        elif model_name == "resnet18":
            emb = get_embeddings_resnet18(train_loader, device)
            logger.info(f"Extracted embeddings using {model_name} : {emb.shape}")
        else:
            raise ValueError(
                f"Embedding model '{model_name}' not implemented in this script."
            )
        embeddings_list.append(emb)
    embeddings = np.concatenate(embeddings_list, axis=1)
    logger.info(f"Combined embeddings shape: {embeddings.shape}")

    zcore_args = dict(
        dataset=cfg.dataset.name,
        n_sample=getattr(cfg.zcoreset, "n_sample", 20000),
        num_workers=getattr(cfg.zcoreset, "num_workers", 4),
        sample_dim=getattr(cfg.zcoreset, "sample_dim", 32),
        rand_init=getattr(cfg.zcoreset, "rand_init", False),
        redund_nn=getattr(cfg.zcoreset, "redund_nn", 8),
        redund_exp=getattr(cfg.zcoreset, "redund_exp", 4),
        trial=getattr(cfg.zcoreset, "trial", 1),
    )
    scores = zcore_score(zcore_args, embeddings)
    zcoreset_scores = {int(i): float(scores[i]) for i in range(len(scores))}
    return zcoreset_scores


# ---- Main Entrypoint ----
def main(cfg_path: str):
    seed_everything(10)

    cfg = OmegaConf.load(cfg_path)
    cfg = cfg.IMAGENET
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainset, train_loader, test_loader, _ = prepare_data(
        cfg.dataset, cfg.training.batch_size
    )
    logger.info(f"Loaded dataset: {cfg.dataset.name}, Device: {device}")

    for num_itr in range(cfg.experiment.num_iterations):
        zcoreset = get_zcoreset_scores(cfg, train_loader, device)

        output_path = f"{cfg.paths.scores}/{cfg.dataset.name}_zcoreset_{num_itr}"

        date = datetime.datetime.now()
        output_path += f"_{date.month}_{date.day}.json"

        with open(output_path, "w") as f:
            json.dump(zcoreset, f)

        logger.info(f"Saved Zcoreset scores to {output_path}")

        # Pruning and evaluation
        if cfg.pruning.prune is True:
            prune(
                trainset=trainset,
                test_loader=test_loader,
                scores_dict=zcoreset,
                cfg=cfg,
                wandb_name="zcoreset",
                device=device,
            )


if __name__ == "__main__":
    default_config_path = os.path.join(
        os.path.dirname(__file__), "configs", "zcoreset.yaml"
    )
    config_path = parse_config(
        default_config=default_config_path,
        description="Run Zcoreset",
    )
    main(cfg_path=config_path)
