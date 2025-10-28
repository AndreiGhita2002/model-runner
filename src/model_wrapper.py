from abc import ABC, abstractmethod
from typing import Any


class ModelWrapper(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def rand_inputs(self):
        pass

    @abstractmethod
    def forward(self, x: Any):
        pass
