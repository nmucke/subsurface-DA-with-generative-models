from pickletools import optimize
import torch

from subsurface_DA_with_generative_models.optimizers.base_optimizer import Optimizer


class BaseTrainStepper():

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
    ) -> None:
        pass
    

    def train_step(self) -> None:
        pass

    def val_step(self) -> None:
        pass