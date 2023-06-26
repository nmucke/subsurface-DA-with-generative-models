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

class WAEOptimizer(Optimizer):

    def __init__(
        self,
        model: torch.nn.Module,
        args: dict,
    ) -> None:
    
        self.model = model
        self.args = args

        self.decoder = torch.optim.Adam(
            self.model.decoder.parameters(),
            lr=self.args['learning_rate'],
            weight_decay=self.args['weight_decay'],
        )

        self.encoder = torch.optim.Adam(
            self.model.encoder.parameters(),
            lr=self.args['learning_rate'],
            weight_decay=self.args['weight_decay'],
        )

        if self.args['scheduler_args'] is not None:
            self._set_scheduler()

    def _set_scheduler(self) -> None:
        self.decoder_scheduler = get_learning_rate_scheduler(
            type=self.args['scheduler_args']['type'],
            optimizer=self.decoder,
            **self.args['scheduler_args']['args']
        )

        self.encoder_scheduler = get_learning_rate_scheduler(
            type=self.args['scheduler_args']['type'],
            optimizer=self.encoder,
            **self.args['scheduler_args']['args']
        )

    def zero_grad(self) -> None:
        self.decoder.zero_grad()
        self.encoder.zero_grad()

    def step(self) -> None:
        self.decoder.step()
        self.encoder.step()

    def step_scheduler(self, loss: float=None) -> None:
        if loss is None:
            self.decoder_scheduler.step()
            self.encoder_scheduler.step()
        else:
            self.decoder_scheduler.step(loss)
            self.encoder_scheduler.step(loss)
 