import torch
import torchvision
from torchvision.datasets.vision import VisionDataset
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Tuple
import os
import json
import glob
import warnings
from dataclasses import dataclass
from key_map import base_key


class ImageKeyDataset(VisionDataset):
    """
    Args:
        root (string): Root directory of dataset.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    classes = base_key.all_key_and_type_comb

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    @dataclass
    class DataSettings:
        model: str
        width: int
        height: int
        FPS: int
        seconds: int

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        data_settings: DataSettings = None,
    ) -> None:
        super().__init__(root, transform=transform)
        self.train = train
        self.data_settings = data_settings
        self.data, self.targets = self._load_data()

    def _load_data(self):
        is_train = "train" if self.train else "test"
        data_paths = glob.glob(self.root + f"/{is_train}/*")
        data = []
        targets = []

        for data_path in data_paths:
            with open(f"{data_path}/record.jsonl", "r") as f:
                records: list[dict] = [json.loads(line) for line in f.readlines()]

            for record in records:
                tick = record.get("tick")
                image_path = f"{data_path}/images/{tick}.png"
                image = Image.open(image_path)
                image_transformed = self.transform(image)

                keyboard = record.get("keyboard")[0]
                label = keyboard["key"] + "_" + keyboard["type"]

                data.append(image_transformed)
                targets.append(label)

        targets = onehot_labels(self.classes, targets)

        data = torch.stack(data, dim=0)
        targets = torch.from_numpy(targets).to(torch.float32)

        return data, targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        if self.data_settings:
            data, target = self._transform_data_integration(index)
        else:
            data, target = self.data[index], self.targets[index]

        return data, target

    def __len__(self) -> int:
        if self.data_settings:
            FPS = self.data_settings.FPS
            seconds = self.data_settings.seconds
            window_size = FPS * seconds
            return len(self.data) - window_size
        else:
            return len(self.data)

    def _transform_data_integration(self, index):
        FPS = self.data_settings.FPS
        seconds = self.data_settings.seconds
        window_size = FPS * seconds

        data = {}

        if self.data_settings.model == "integration":
            prev_frames = []
            for i in range(0, window_size, FPS):
                prev_frames.append(self.data[index + i])
            prev_frames = torch.stack(prev_frames, dim=0)
            data["prev_frames"] = prev_frames

        prev_keys = self.targets[index : index + window_size]
        curr_frame = self.data[index + window_size]
        curr_key = self.targets[index + window_size]

        data["prev_keys"] = prev_keys
        data["curr_frame"] = curr_frame
        target = curr_key

        return data, target


def onehot_labels(categories, labels):
    labels = [[key] for key in labels]
    onehot_encoder = OneHotEncoder(sparse=False, categories=[categories])
    onehot_labels = onehot_encoder.fit_transform(labels)
    return onehot_labels
