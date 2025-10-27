import torchvision.models as models
from torchvision.models import ConvNeXt_Small_Weights
import torch

from src.logger import ModelLogger

def conv_next_run():
    # Load model with the new weights API
    weights = ConvNeXt_Small_Weights.IMAGENET1K_V1
    model = models.convnext_small(weights=weights)
    model.eval()

    logger = ModelLogger()

    print("Model ConvNeXt_Small_Weights loaded.")

    print("Model patched with logger")
    logger.patch_module(model)

    print(torch.accelerator.is_available())
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model.to(device)
    print(f"Using {device} device")

    # random input
    random_input = torch.randn(1, 3, 224, 224).to(device)

    # Run inference
    print("Running inference..")
    with torch.no_grad():
        output = model(random_input)
        predicted_class = output.argmax(dim=1)
        print("Result:", predicted_class)

    return logger.to_json()


if __name__ == '__main__':
    print(conv_next_run())
