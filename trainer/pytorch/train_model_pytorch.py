import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import model_arch.pytorch as torch_models
from pytorch_datasets import ImageKeyDataset
from tqdm import tqdm


def get_model(model_class: nn.Module, device):
    model = model_class().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    return model, criterion, optimizer


def train(train_loader, model, criterion, optimizer, epoch):
    for epoch in range(epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({"Loss": loss.item()})

        progress_bar.close()


def test(test_loader, model, device):
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        _, labels = torch.max(labels, 1)

        output = model(images)
        _, predicted = torch.max(output, 1)
        correct += (predicted == labels).sum()
        total += labels.size(0)

    print("Accuracy of the model: %.3f %%" % ((100 * correct) / (total + 1)))


if __name__ == "__main__":
    model_class = torch_models.EfficientNetV2S
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    width = 210
    height = 60
    epochs = 100
    batch_size = 256

    transform = transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_data = ImageKeyDataset(root="data", train=True, transform=transform)
    test_data = ImageKeyDataset(root="data", train=False, transform=transform)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size)

    model, criterion, optimizer = get_model(model_class, device=device)

    train(train_loader, model, criterion, optimizer, epochs)
    test(test_loader, model, device)

    torch.save(model, "model.pt")
