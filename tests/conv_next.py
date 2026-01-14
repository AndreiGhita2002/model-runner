import torchvision.models as models
from torchvision.models import ConvNeXt_Small_Weights, ConvNeXt
import torch


class ConvNext:
    # TODO: Why is this wrapper necessary?
    model: ConvNeXt

    def __init__(self, device):
        weights = ConvNeXt_Small_Weights.IMAGENET1K_V1
        self.model = models.convnext_small(weights=weights)
        self.model.eval()

        self.device = device
        self.model.to(self.device)

    def rand_inputs(self):
        return torch.randn(1, 3, 224, 224).to(self.device)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

