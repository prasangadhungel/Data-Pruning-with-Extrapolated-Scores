import os

import torch

import torchvision
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
import torchvision.transforms.functional as TF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_embeddings(model, data_loader):
    model.eval()
    features = []
    with torch.no_grad():
        for images, _ in data_loader:
            outputs = model(images)  # Replace with appropriate output extraction for your model
            features.append(outputs.cpu().detach())
    features = torch.cat(features, dim=0)
    return features

# Load CIFAR-10 dataset
if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Load pre-trained self-supervised model (SWaV)
    model = torch.hub.load('facebookresearch/swav:main', 'resnet50', pretrained=True)
    model = model.to(device)

    embeddings = get_embeddings(model, trainloader)
    kmeans = KMeans(n_clusters=10, random_state=0)
    kmeans.fit(embeddings.numpy())

    distances = []
    for embedding in embeddings:
        distances.append(1 - torch.nn.functional.cosine_similarity(embedding.unsqueeze(0), torch.tensor(kmeans.cluster_centers_), dim=1).item())
    
    difficulty = torch.tensor(distances)

    classes = trainset.classes

    for prune_percentage in [0, 0.1, 0.2, 0.5]:
        num_to_prune = int(prune_percentage * len(difficulty))
        indices_to_prune = torch.argsort(difficulty)[:num_to_prune]
        prune_indices = difficulty.argsort()[:num_to_prune]
        trainset_pruned = torch.utils.data.Subset(trainset, prune_indices.complement())  # Keep data with higher difficulty scores

        total_samples_kept = len(trainset_pruned)
        print(f"{total_samples_kept} samples kept")

        data_dir = "./diff_samples_" + str(int(prune_percentage * 100))
        for cls in classes:
            os.makedirs(os.path.join(data_dir, cls), exist_ok=True)
        
        for idx, (image, label) in enumerate(trainset_pruned):
            class_dir = os.path.join(data_dir, classes[label])
            image_path = os.path.join(class_dir, f"img_{idx}.jpg")
            tensor_image = TF.to_tensor(image)  # Convert PIL image to tensor
            torchvision.utils.save_image(tensor_image, image_path)





