from abc import abstractmethod
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
    

    @abstractmethod
    def train_step(
        self,
        batch: dict,
        ) -> None:

        raise NotImplementedError

    @abstractmethod
    def val_step(
        self,
        dataloader: torch.utils.data.DataLoader,
        ) -> None:

        raise NotImplementedError

    @abstractmethod
    def start_epoch(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def end_epoch(self, val_loss) -> None:
        raise NotImplementedError
    





