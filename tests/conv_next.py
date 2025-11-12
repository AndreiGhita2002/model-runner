import torchvision.models as models
from torchvision.models import ConvNeXt_Small_Weights, ConvNeXt
import torch

from src.logger import ModelLogger


class ConvNext():
    model: ConvNeXt

    def __init__(self):
        weights = ConvNeXt_Small_Weights.IMAGENET1K_V1
        self.model = models.convnext_small(weights=weights)
        self.model.eval()

        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        self.model.to(self.device)

    def rand_inputs(self):
        return torch.randn(1, 3, 224, 224).to(self.device)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

