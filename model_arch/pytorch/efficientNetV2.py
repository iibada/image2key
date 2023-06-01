import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    efficientnet_v2_s,
    efficientnet_v2_m,
    efficientnet_v2_l,
)
from key_map import base_key


class EfficientNetV2(nn.Module):
    def __init__(self, model_fn):
        super(EfficientNetV2, self).__init__()

        self.efficient_net = model_fn(pretrained=True)
        num_features = self.efficient_net.classifier[-1].in_features
        num_classes = len(base_key.all_key_and_type_comb)
        self.efficient_net.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.efficient_net(x)


class EfficientNetV2S(EfficientNetV2):
    def __init__(self):
        super().__init__(efficientnet_v2_s)


class EfficientNetV2M(EfficientNetV2):
    def __init__(self):
        super().__init__(efficientnet_v2_m)


class EfficientNetV2L(EfficientNetV2):
    def __init__(self):
        super().__init__(efficientnet_v2_l)
