from typing import Any

import torch

# from src.logger import ModelLogger
from src.timer_wrapper import TimerWrapper
from tests.conv_next import ConvNext
from tests.simple_net import SimpleNet


class MainService:
    # logger = ModelLogger()
    models: dict[str, TimerWrapper] = {}

    def __init__(self):
        # initialise models
        self.models['simple-net'] = TimerWrapper(SimpleNet())
        self.models['conv-next'] = TimerWrapper(ConvNext())

    def run_model(self, model_name: str, x: Any, randomise_input=False):
        model = self.models.get(model_name, None)

        if model is None:
            print("MainService.run_model: provided model_name does not correspond to any known model!\n"
                  " provided model_name: ", model_name)
            return None

        if randomise_input or x is None:
            if callable(model.rand_inputs):
                x = model.rand_inputs()
            if x is None or not callable(model.rand_inputs):
                return {'error': 'Input was not provided, or the model does not define rand_inputs function!'}

        if x is None:
            print("MainService.run_model: provided input is None!")

        with torch.no_grad():
            return model(x)

    def get_logs(self):
        #todo
        # return self.logger.to_dict()
        return {}

    def get_model_names(self):
        return self.models.keys()


if __name__ == '__main__':
    _main = MainService()
