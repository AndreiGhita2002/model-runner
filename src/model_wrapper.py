from abc import ABC, abstractmethod
from typing import Any


# TODO user should not be required to implement something like this, figure out a way to do this automatically

class ModelWrapper(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def rand_inputs(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    # @abstractmethod
    # def forward(self, x: Any):
    #     pass
