import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import model_arch.pytorch as torch_models
from pytorch_datasets import ImageKeyDataset
from tqdm import tqdm
from key_map import base_key


def get_weights():
    key_weights_dino = {
        base_key.code2key["0"]: 0.1,  # none
        base_key.code2key["38"]: 3,  # up_arrow
        base_key.code2key["40"]: 1,  # down_arrow
    }
    all_key_weights_dino = [
        key_weights_dino.get(
            key_and_type.rsplit("_", 1)[0], base_key.default_key_weights
        )
        for key_and_type in base_key.all_key_and_type_comb
    ]
    return torch.tensor(all_key_weights_dino)


def get_model(model_class: nn.Module, **kwargs):
    device = kwargs.get(
        "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    weights = kwargs.get("weights")

    model = model_class(**kwargs).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    return model, criterion, optimizer


def train(train_loader, model, criterion, optimizer, epoch):
    min_loss = float("inf")
    for epoch in range(epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

        for data, labels in progress_bar:
            prev_keys = data["prev_keys"].to(device)
            curr_frame = data["curr_frame"].to(device)
            curr_key = labels.to(device)

            optimizer.zero_grad()
            outputs = model(prev_keys, curr_frame)
            loss = criterion(outputs, curr_key)
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({"Loss": loss.item()})

        progress_bar.close()

        # if loss.item() < min_loss:
        #     min_loss = loss.item()
        #     torch.save(model, f"model-{epoch}-{min_loss}.pt")


def test(test_loader, model, device):
    correct = 0
    total = 0
    for data, labels in test_loader:
        prev_keys = data["prev_keys"].to(device)
        curr_frame = data["curr_frame"].to(device)
        curr_key = labels.to(device)
        _, curr_key = torch.max(curr_key, 1)

        output = model(prev_keys, curr_frame)
        _, predicted = torch.max(output, 1)
        correct += (predicted == curr_key).sum()
        total += curr_key.size(0)

    print("Accuracy of the model: %.3f %%" % ((100 * correct) / (total + 1)))


if __name__ == "__main__":
    model_class = torch_models.LSTM_EfficientNetV2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    width = 210
    height = 60
    epochs = 100
    batch_size = 256
    FPS = 30
    seconds = 5

    transform = transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    data_settings = ImageKeyDataset.DataSettings(
        model="lstm_efficientNetV2",
        width=width,
        height=height,
        FPS=FPS,
        seconds=seconds,
    )

    train_data = ImageKeyDataset(
        root="data", train=True, transform=transform, data_settings=data_settings
    )
    test_data = ImageKeyDataset(
        root="data", train=False, transform=transform, data_settings=data_settings
    )

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size)

    model, criterion, optimizer = get_model(
        model_class, device=device, data_settings=data_settings, weights=get_weights()
    )

    train(train_loader, model, criterion, optimizer, epochs)
    test(test_loader, model, device)

    torch.save(model, "model-last.pt")
