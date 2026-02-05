import types
import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision.models import ConvNeXt, ConvNeXt_Small_Weights, convnext_small


#TODO: get more models
# from https://docs.pytorch.org/vision/main/models.html
# should have: image classification, ...

class SimpleNet(nn.Module):
    #TODO: note that this is not trained

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

#################################################

evaluation_models: list[tuple[str, types.FunctionType, types.FunctionType]] = [
    ("simple_net", load_simple_net, simple_net_rand_inputs),
    ("conv_next", load_conv_next, conv_next_rand_inputs)
]
