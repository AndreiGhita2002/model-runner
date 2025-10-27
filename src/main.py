from typing import Any

from src.logger import ModelLogger
from src.model_wrapper import ModelWrapper
from tests.conv_next import ConvNextWrapper
from tests.simple_net import SimpleNet


class MainService:
    logger = ModelLogger()
    models: dict[str, ModelWrapper] = {}

    def __init__(self):
        # initialise models
        self.models['simple-net'] = SimpleNet(self.logger)
        self.models['conv-next'] = ConvNextWrapper(self.logger)

    def run_model(self, model_name: str, x: Any, randomise_input=False):
        model = self.models.get(model_name, None)

        if model is None:
            print("MainService.run_model: provided model_name does not correspond to any known model!\n"
                  " provided model_name: ", model_name)
            return None

        if randomise_input:
            ri = model.rand_inputs()
            return model.forward(ri)

        if x is None:
            print("MainService.run_model: provided input is None!")

        return model.forward(x)


if __name__ == '__main__':
    _main = MainService()
