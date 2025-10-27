import torch
import torch.nn as nn
import torch.nn.functional as F

from src.logger import ModelLogger, LOGGER
from src.model_wrapper import ModelWrapper


class SimpleNet(nn.Module, ModelWrapper):
    def __init__(self, logger: ModelLogger):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

        logger.patch_module(self, name='SimpleNet')

        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        self.to(self.device)

    def rand_inputs(self):
        return torch.randn(1, 1, 16, 16).to(self.device)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

