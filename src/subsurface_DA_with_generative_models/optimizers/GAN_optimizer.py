import pdb
from pickletools import optimize
import numpy as np
import torch
import torch.optim as optim

from subsurface_DA_with_generative_models.optimizers.base_optimizer import Optimizer
from subsurface_DA_with_generative_models.optimizers.optimizer_utils import get_learning_rate_scheduler
from pickletools import optimize
import numpy as np
import torch
import torch.optim as optim

from subsurface_DA_with_generative_models.optimizers.base_optimizer import Optimizer
from subsurface_DA_with_generative_models.optimizers.optimizer_utils import get_learning_rate_scheduler

class GANOptimizer(Optimizer):

    def __init__(
        self,
        model: torch.nn.Module,
        args: dict,
    ) -> None:
    
        self.model = model
        self.args = args

        self.generator = torch.optim.Adam(
            self.model.generator.parameters(),
            lr=self.args['learning_rate'],
            weight_decay=self.args['weight_decay'],
        )

        self.critic = torch.optim.Adam(
            self.model.critic.parameters(),
            lr=self.args['learning_rate'],
            weight_decay=self.args['weight_decay'],
        )

        if self.args['scheduler_args'] is not None:
            self._set_scheduler()

    def load_state_dict(self, state_dict: dict) -> None:
        self.generator.load_state_dict(state_dict['generator_optimizer_state_dict'])
        self.critic.load_state_dict(state_dict['critic_optimizer_state_dict'])
        self.generator_scheduler.load_state_dict(state_dict['generator_scheduler_state_dict'])
        self.critic_scheduler.load_state_dict(state_dict['critic_scheduler_state_dict'])

    def _set_scheduler(self) -> None:
        self.generator_scheduler = get_learning_rate_scheduler(
            type=self.args['scheduler_args']['type'],
            optimizer=self.generator,
            **self.args['scheduler_args']['args']
        )

        self.critic_scheduler = get_learning_rate_scheduler(
            type=self.args['scheduler_args']['type'],
            optimizer=self.critic,
            **self.args['scheduler_args']['args']
        )

    def zero_grad(self) -> None:
        self.generator.zero_grad()
        self.critic.zero_grad()

    def step(self) -> None:
        self.generator.step()
        self.critic.step()

    def step_scheduler(self, loss: float=None) -> None:
        if self.args['scheduler_args']['type'] != 'plateau':
            self.generator_scheduler.step()
            self.critic_scheduler.step()
        else:
            self.generator_scheduler.step(loss)
            self.critic_scheduler.step(loss)

