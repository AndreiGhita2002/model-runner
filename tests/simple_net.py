import torch
import torch.nn as nn
import torch.nn.functional as F

from src.logger import ModelLogger


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def simple_net_run():
    model = SimpleNet()
    logger = ModelLogger()
    logger.patch_module(model, name='SimpleNet')

    input_tensor = torch.randn(1, 1, 16, 16)
    output = model(input_tensor)
    print("Model Output 1:", output)

    input_tensor = torch.randn(1, 1, 16, 16)
    output = model(input_tensor)
    print("Model Output 2:", output)

    return logger.to_json()

if __name__ == '__main__':
    print(simple_net_run())
