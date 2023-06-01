import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from model_arch.pytorch import ConvLSTM
from torchvision.models import efficientnet_v2_s
from key_map import base_key
from pytorch_datasets import ImageKeyDataset


class Integration(nn.Module):
    """
    ConvLSTM, LSTM, EfficientNetV2 integrated model
    """

    def __init__(self, **kwargs):
        super(Integration, self).__init__()

        data_settings: ImageKeyDataset.DataSettings = kwargs["data_settings"]
        width = data_settings.width
        height = data_settings.height
        FPS = data_settings.FPS
        seconds = data_settings.seconds
        frame_size = FPS * seconds

        num_classes = len(base_key.all_key_and_type_comb)

        conv_lstm_hidden_dim = [64, 64, 128]
        self.conv_lstm = ConvLSTM(
            input_dim=3,
            hidden_dim=conv_lstm_hidden_dim,
            kernel_size=(3, 3),
            num_layers=3,
            batch_first=True,
            bias=True,
            return_all_layers=False,
        )
        self.batch_norm = nn.BatchNorm2d(num_features=conv_lstm_hidden_dim[-1])
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(
            in_features=conv_lstm_hidden_dim[-1] * width * height,
            out_features=num_classes,
        )

        self.lstm_hidden_size = 64
        self.lstm_bidirectional = True
        self.lstm = nn.LSTM(
            input_size=num_classes,
            hidden_size=self.lstm_hidden_size,
            num_layers=3,
            batch_first=True,
            bidirectional=self.lstm_bidirectional,
        )
        self.fc2 = nn.Linear(
            in_features=self.lstm_hidden_size * 2 if self.lstm_bidirectional else 1,
            out_features=num_classes,
        )

        self.efficient_net = efficientnet_v2_s(pretrained=True)
        num_features = self.efficient_net.classifier[-1].in_features
        self.efficient_net.classifier = nn.Linear(num_features, num_classes)

        self.dropout = nn.Dropout(p=0.1)
        self.last_fc = nn.Linear(
            in_features=self.fc1.out_features
            + self.fc2.out_features
            + self.efficient_net.classifier.out_features,
            out_features=num_classes,
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input1, input2, input3):
        [convlstm_output], [(convlstm_h, convlstm_c)] = self.conv_lstm(input1)
        o = convlstm_output[:, -1, :, :, :]
        prev_frames = self.batch_norm(o)
        prev_frames = self.flatten(prev_frames)
        prev_frames = self.fc1(prev_frames)

        lstm_output, (lstm_h, lstm_c) = self.lstm(input2)
        o = lstm_output[:, -1, :]
        prev_keys = self.fc2(o)

        curr_frame = self.efficient_net(input3)

        concatenated = torch.cat((prev_frames, prev_keys, curr_frame), dim=1)
        concatenated = F.relu(concatenated)
        concatenated = self.dropout(concatenated)

        curr_key = self.last_fc(concatenated)
        # curr_key = self.softmax(curr_key)

        return curr_key
