import types
import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision.models import (
    ConvNeXt, ConvNeXt_Small_Weights, convnext_small,
    ConvNeXt_Base_Weights, convnext_base,
    EfficientNet_B6_Weights, efficientnet_b6,
    Inception_V3_Weights, inception_v3,
    RegNet_X_16GF_Weights, regnet_x_16gf,
)


#TODO: get more models
# from https://docs.pytorch.org/vision/main/models.html
# should have: image classification, ...

class SimpleNet(nn.Module):
    def __init__(self, device):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

        if device is not None:
            self.device = device
            self.to(self.device)

    def rand_inputs(self):
        return torch.randn(1, 1, 16, 16).to(self.device)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#################################################

def load_conv_next(device: str = "cpu") -> ConvNeXt:
    weights = ConvNeXt_Small_Weights.IMAGENET1K_V1
    model = convnext_small(weights=weights).eval()
    if device is not None:
        model.to(device)
    return model

def conv_next_rand_inputs() -> torch.Tensor:
    return torch.randn(1, 3, 224, 224)

def load_simple_net(device: str = "cpu") -> nn.Module:
    torch.manual_seed(0)
    return SimpleNet(device).eval()

def simple_net_rand_inputs() -> torch.Tensor:
    return torch.randn(1, 1, 16, 16)

def load_conv_next_base(device: str = "cpu") -> nn.Module:
    weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
    model = convnext_base(weights=weights).eval()
    if device is not None:
        model.to(device)
    return model

def conv_next_base_rand_inputs() -> torch.Tensor:
    return torch.randn(1, 3, 224, 224)

def load_efficientnet_b6(device: str = "cpu") -> nn.Module:
    weights = EfficientNet_B6_Weights.IMAGENET1K_V1
    model = efficientnet_b6(weights=weights).eval()
    if device is not None:
        model.to(device)
    return model

def efficientnet_b6_rand_inputs() -> torch.Tensor:
    return torch.randn(1, 3, 528, 528)

def load_inception_v3(device: str = "cpu") -> nn.Module:
    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights).eval()
    if device is not None:
        model.to(device)
    return model

def inception_v3_rand_inputs() -> torch.Tensor:
    return torch.randn(1, 3, 299, 299)

def load_regnet_x_16gf(device: str = "cpu") -> nn.Module:
    weights = RegNet_X_16GF_Weights.IMAGENET1K_V2
    model = regnet_x_16gf(weights=weights).eval()
    if device is not None:
        model.to(device)
    return model

def regnet_x_16gf_rand_inputs() -> torch.Tensor:
    return torch.randn(1, 3, 224, 224)

#################################################

evaluation_models: list[tuple[str, types.FunctionType, types.FunctionType]] = [
    ("simple_net", load_simple_net, simple_net_rand_inputs),
    ("conv_next", load_conv_next, conv_next_rand_inputs),
    ("conv_next_base", load_conv_next_base, conv_next_base_rand_inputs),
    ("efficientnet_b6", load_efficientnet_b6, efficientnet_b6_rand_inputs),
    ("inception_v3", load_inception_v3, inception_v3_rand_inputs),
    ("regnet_x_16gf", load_regnet_x_16gf, regnet_x_16gf_rand_inputs),
]
