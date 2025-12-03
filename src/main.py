import pprint
from typing import Any

import torch
from torch import nn

# from src.logger import ModelLogger
from src.timed_module import TimedModule, make_module_timed
from tests.conv_next import ConvNext
from tests.simple_net import SimpleNet


class MainService:
    # logger = ModelLogger()
    models: dict[str, nn.Module] = {}

    def __init__(self, depth=2):
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        self.depth = depth

        # initialise models
        self.models['simple-net'] = make_module_timed(SimpleNet(self.device), device=self.device, depth=depth)
        self.models['conv-next'] = make_module_timed(ConvNext(self.device), device=self.device, depth=depth)

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

        # pipe = pipeline(
        #     module=model,
        #     mb_args=(x,),
        #     # split_spec={
        #     #     "layers.1": SplitPoint.BEGINNING,
        #     # }
        # )
        # print("pipe:", pipe)

        return model.run()

    def get_logs(self):
        #todo
        # return self.logger.to_dict()
        l = {}
        for (model_name, model) in self.models.items():
            if isinstance(model, TimedModule):
                l[model_name] = model.get_logs()
            else:
                l[model_name] = None
        return l

    def get_model_names(self):
        return self.models.keys()


if __name__ == '__main__':
    main = MainService()

    RESULT_FILE = "results.txt"

    a = main.run_model("simple-net", None, randomise_input=True)
    logs = main.get_logs()
    pretty_logs1 = pprint.pformat(logs)
    b = main.run_model("simple-net", None, randomise_input=True)
    logs = main.get_logs()
    pretty_logs2 = pprint.pformat(logs)

    with open(RESULT_FILE, "w") as f:
        f.write(f"Result a: {a}\n")
        f.write(f"Result b: {b}\n")
        f.write("Logs 1:\n")
        f.write(pretty_logs1)
        f.write("Logs 2:\n")
        f.write(pretty_logs2)
