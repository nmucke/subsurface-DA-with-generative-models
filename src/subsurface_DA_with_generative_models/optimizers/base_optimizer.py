from pickletools import optimize
import numpy as np
import torch
import torch.optim as optim
from abc import abstractmethod



class Optimizer():

    def __init__(
        self,
        model: torch.nn.Module,
        args: dict,
    ) -> None:
        
        self.model = model
        self.args = args

    @abstractmethod
    def zero_grad(self) -> None:
        pass

    @abstractmethod
    def step(self) -> None:
        pass

    def step_scheduler(self, loss: float=None) -> None:
        if self.step_scheduler is None:
            pass