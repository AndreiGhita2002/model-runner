import types
import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision.models import (
    ConvNeXt, ConvNeXt_Small_Weights, convnext_small,
    ConvNeXt_Base_Weights, convnext_base,
    EfficientNet_B6_Weights, efficientnet_b6,
    EfficientNet_V2_S_Weights, efficientnet_v2_s,
    Inception_V3_Weights, inception_v3,
    MaxVit_T_Weights, maxvit_t,
    MobileNet_V3_Large_Weights, mobilenet_v3_large,
    RegNet_X_16GF_Weights, regnet_x_16gf,
    ResNet50_Weights, resnet50,
    Swin_T_Weights, swin_t,
    ViT_B_16_Weights, vit_b_16,
)


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

def load_simple_net(device: str = "cpu") -> nn.Module:
    torch.manual_seed(0)
    return SimpleNet(device).eval()

def simple_net_rand_inputs() -> torch.Tensor:
    return torch.randn(1, 1, 16, 16)

# ConvNeXt Small — modern CNN with inverted bottlenecks and large kernels (~50M params)
def load_conv_next(device: str = "cpu") -> ConvNeXt:
    weights = ConvNeXt_Small_Weights.IMAGENET1K_V1
    model = convnext_small(weights=weights).eval()
    if device is not None:
        model.to(device)
    return model

def conv_next_rand_inputs() -> torch.Tensor:
    return torch.randn(1, 3, 224, 224)

# ConvNeXt Base — larger variant of ConvNeXt (~89M params)
def load_conv_next_base(device: str = "cpu") -> nn.Module:
    weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
    model = convnext_base(weights=weights).eval()
    if device is not None:
        model.to(device)
    return model

def conv_next_base_rand_inputs() -> torch.Tensor:
    return torch.randn(1, 3, 224, 224)

# EfficientNet B6 — compound-scaled CNN, larger input resolution (~43M params)
def load_efficientnet_b6(device: str = "cpu") -> nn.Module:
    weights = EfficientNet_B6_Weights.IMAGENET1K_V1
    model = efficientnet_b6(weights=weights).eval()
    if device is not None:
        model.to(device)
    return model

def efficientnet_b6_rand_inputs() -> torch.Tensor:
    return torch.randn(1, 3, 528, 528)

# Inception V3 — multi-branch architecture with factored convolutions (~24M params)
def load_inception_v3(device: str = "cpu") -> nn.Module:
    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights).eval()
    if device is not None:
        model.to(device)
    return model

def inception_v3_rand_inputs() -> torch.Tensor:
    return torch.randn(1, 3, 299, 299)

# RegNet X 16GF — regular network designed by architecture search, uniform structure (~54M params)
def load_regnet_x_16gf(device: str = "cpu") -> nn.Module:
    weights = RegNet_X_16GF_Weights.IMAGENET1K_V2
    model = regnet_x_16gf(weights=weights).eval()
    if device is not None:
        model.to(device)
    return model

def regnet_x_16gf_rand_inputs() -> torch.Tensor:
    return torch.randn(1, 3, 224, 224)

# ResNet-50 — classic residual network, most common CV baseline (~25M params)
def load_resnet50(device: str = "cpu") -> nn.Module:
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights).eval()
    if device is not None:
        model.to(device)
    return model

def resnet50_rand_inputs() -> torch.Tensor:
    return torch.randn(1, 3, 224, 224)

# MobileNet V3 Large — lightweight edge model, depthwise separable convolutions (~5.5M params)
def load_mobilenet_v3_large(device: str = "cpu") -> nn.Module:
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
    model = mobilenet_v3_large(weights=weights).eval()
    if device is not None:
        model.to(device)
    return model

def mobilenet_v3_large_rand_inputs() -> torch.Tensor:
    return torch.randn(1, 3, 224, 224)

# EfficientNet V2 Small — fused-MBConv, faster training than V1 (~22M params)
def load_efficientnet_v2_s(device: str = "cpu") -> nn.Module:
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = efficientnet_v2_s(weights=weights).eval()
    if device is not None:
        model.to(device)
    return model

def efficientnet_v2_s_rand_inputs() -> torch.Tensor:
    return torch.randn(1, 3, 384, 384)

# ViT B/16 — standard Vision Transformer, patches image into 16x16 tokens (~86M params)
def load_vit_b_16(device: str = "cpu") -> nn.Module:
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights).eval()
    if device is not None:
        model.to(device)
    return model

def vit_b_16_rand_inputs() -> torch.Tensor:
    return torch.randn(1, 3, 224, 224)

# Swin-T — shifted-window transformer, hierarchical features like CNNs (~28M params)
def load_swin_t(device: str = "cpu") -> nn.Module:
    weights = Swin_T_Weights.IMAGENET1K_V1
    model = swin_t(weights=weights).eval()
    if device is not None:
        model.to(device)
    return model

def swin_t_rand_inputs() -> torch.Tensor:
    return torch.randn(1, 3, 224, 224)

# MaxViT-T — hybrid block-local + dilated global attention (~31M params)
def load_maxvit_t(device: str = "cpu") -> nn.Module:
    weights = MaxVit_T_Weights.IMAGENET1K_V1
    model = maxvit_t(weights=weights).eval()
    if device is not None:
        model.to(device)
    return model

def maxvit_t_rand_inputs() -> torch.Tensor:
    return torch.randn(1, 3, 224, 224)

#################################################

evaluation_models: list[tuple[str, types.FunctionType, types.FunctionType]] = [
    ("conv_next", load_conv_next, conv_next_rand_inputs),              # modern CNN
    ("mobilenet_v3_large", load_mobilenet_v3_large, mobilenet_v3_large_rand_inputs),  # lightweight edge (~5.5M)
    ("efficientnet_b6", load_efficientnet_b6, efficientnet_b6_rand_inputs),  # efficient CNN
    ("regnet_x_16gf", load_regnet_x_16gf, regnet_x_16gf_rand_inputs),  # regular CNN
]

evaluation_models_extended: list[tuple[str, types.FunctionType, types.FunctionType]] = [
    # CNNs
    ("conv_next", load_conv_next, conv_next_rand_inputs),              # modern CNN (~50M)
    ("conv_next_base", load_conv_next_base, conv_next_base_rand_inputs),  # modern CNN (~89M)
    ("efficientnet_b6", load_efficientnet_b6, efficientnet_b6_rand_inputs),  # efficient CNN (~43M)
    ("regnet_x_16gf", load_regnet_x_16gf, regnet_x_16gf_rand_inputs),  # regular CNN (~54M)
    ("resnet50", load_resnet50, resnet50_rand_inputs),                 # classic residual (~25M)
    ("mobilenet_v3_large", load_mobilenet_v3_large, mobilenet_v3_large_rand_inputs),  # lightweight edge (~5.5M)
    ("efficientnet_v2_s", load_efficientnet_v2_s, efficientnet_v2_s_rand_inputs),  # fused-MBConv (~22M)
    # Transformers
    ("vit_b_16", load_vit_b_16, vit_b_16_rand_inputs),                # pure transformer (~86M)
    ("swin_t", load_swin_t, swin_t_rand_inputs),                      # windowed transformer (~28M)
    # ("maxvit_t", load_maxvit_t, maxvit_t_rand_inputs),                # hybrid attention (~31M)
]

evaluation_models_reduced: list[tuple[str, types.FunctionType, types.FunctionType]] = [
    ("conv_next", load_conv_next, conv_next_rand_inputs),
    ("efficientnet_b6", load_efficientnet_b6, efficientnet_b6_rand_inputs),
]

other_models: list[tuple[str, types.FunctionType, types.FunctionType]] = [
    ("mobilenet_v3_large", load_mobilenet_v3_large, mobilenet_v3_large_rand_inputs),  # lightweight edge (~5.5M)
    ("regnet_x_16gf", load_regnet_x_16gf, regnet_x_16gf_rand_inputs),  # regular CNN
]

MODEL_SETS = {
    "small": evaluation_models,
    "extended": evaluation_models_extended,
    "reduced": evaluation_models_reduced,
    "other": other_models,
}
