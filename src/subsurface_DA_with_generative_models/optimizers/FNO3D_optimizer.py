

import pdb
from pickletools import optimize
import numpy as np
import torch
import torch.optim as optim

from subsurface_DA_with_generative_models.optimizers.base_optimizer import Optimizer
from subsurface_DA_with_generative_models.optimizers.optimizer_utils import get_learning_rate_scheduler

class FNO3dOptimizer(Optimizer):
    def __init__(
        self,
        model: torch.nn.Module,
        args: dict,
    ) -> None:
    
        self.model = model
        self.args = args

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args['learning_rate'],
            weight_decay=self.args['weight_decay'],
        )

        if self.args['scheduler_args'] is not None:
            self._set_scheduler()

    def load_state_dict(self, state_dict: dict) -> None:
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])

    def _set_scheduler(self) -> None:
        self.scheduler = get_learning_rate_scheduler(
            type=self.args['scheduler_args']['type'],
            optimizer=self.optimizer,
            **self.args['scheduler_args']['args']
        )

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def step(self) -> None:
        self.optimizer.step()

    def step_scheduler(self, loss: float=None) -> None:
        if self.args['scheduler_args']['type'] != 'plateau':
            self.scheduler.step()
        else:
            self.scheduler.step(loss)