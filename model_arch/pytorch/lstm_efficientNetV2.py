import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from model_arch.pytorch import ConvLSTM
from torchvision.models import efficientnet_v2_s
from key_map import base_key
from pytorch_datasets import ImageKeyDataset


class LSTM_EfficientNetV2(nn.Module):
    """
    LSTM + EfficientNetV2 model
    """

    def __init__(self, **kwargs):
        super(LSTM_EfficientNetV2, self).__init__()

        data_settings: ImageKeyDataset.DataSettings = kwargs["data_settings"]
        width = data_settings.width
        height = data_settings.height
        FPS = data_settings.FPS
        seconds = data_settings.seconds
        frame_size = FPS * seconds

        num_classes = len(base_key.all_key_and_type_comb)

        self.lstm_hidden_size = 64
        self.lstm_bidirectional = True
        self.lstm = nn.LSTM(
            input_size=num_classes,
            hidden_size=self.lstm_hidden_size,
            num_layers=3,
            batch_first=True,
            bidirectional=self.lstm_bidirectional,
        )
        self.fc1 = nn.Linear(
            in_features=self.lstm_hidden_size * 2 if self.lstm_bidirectional else 1,
            out_features=num_classes,
        )

        self.efficient_net = efficientnet_v2_s(pretrained=True)
        num_features = self.efficient_net.classifier[-1].in_features
        self.efficient_net.classifier = nn.Linear(num_features, num_classes)

        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(
            in_features=self.fc1.out_features
            + self.efficient_net.classifier.out_features,
            out_features=num_classes,
        )

    def forward(self, input1, input2):
        lstm_output, (lstm_h, lstm_c) = self.lstm(input1)
        o = lstm_output[:, -1, :]
        prev_keys = self.fc1(o)

        curr_frame = self.efficient_net(input2)

        concatenated = torch.cat((prev_keys, curr_frame), dim=1)
        concatenated = F.relu(concatenated)
        concatenated = self.dropout(concatenated)

        curr_key = self.fc2(concatenated)

        return curr_key
