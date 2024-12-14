import torch


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def get_top_k_accuracy(model, test_loader, device, k=5):
    model.eval()
    correct = 0
    top_k_correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            _, top_k_predictions = torch.topk(outputs, k, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            top_k_correct += (top_k_predictions == labels.view(-1, 1)).sum().item()

    accuracy = 100 * correct / total
    top_k_accuracy = 100 * top_k_correct / total

    return accuracy, top_k_accuracy


@torch.no_grad()
def evaluate(model, test_loader, device):
    model.eval()
    outputs = []

    for batch in test_loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        out = model(images)
        acc = accuracy(out, labels)
        outputs.append(acc)

    epoch_acc = torch.stack(outputs).mean()
    return epoch_acc.item()
