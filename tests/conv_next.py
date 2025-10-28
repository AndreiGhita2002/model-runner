import torchvision.models as models
from torchvision.models import ConvNeXt_Small_Weights, ConvNeXt
import torch

from src.logger import ModelLogger
from src.model_wrapper import ModelWrapper


class ConvNextWrapper(ModelWrapper):
    model: ConvNeXt

    def __init__(self, logger: ModelLogger):
        weights = ConvNeXt_Small_Weights.IMAGENET1K_V1
        self.model = models.convnext_small(weights=weights)
        self.model.eval()

        logger.patch_module(self.model)

        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        self.model.to(self.device)

    def rand_inputs(self):
        return torch.randn(1, 3, 224, 224).to(self.device)

    def forward(self, x):
        output = self.model(x)
        predicted_class = output.argmax(dim=1)
        return predicted_class

