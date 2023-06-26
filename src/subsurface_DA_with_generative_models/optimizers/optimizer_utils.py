import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

class LRScheduler(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        # Check if using mixed precision training
        self.mixed_training = False
        base_optimizer = optimizer
 
        # Check that optimizer param is valid
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))

        super(LRScheduler, self).__init__(base_optimizer, last_epoch)

    def step(self, epoch=None):
        # Set the current training step
        # ('epoch' is used to be consistent with _LRScheduler)
        if self.mixed_training:
            # The assumption is that the step will be constant
            state_dict = self.optimizer.state[self.optimizer.param_groups[0]['params'][0]]
            if 'step' in state_dict:
                self.last_epoch = state_dict['step'] + 1
            else:
                self.last_epoch = 1
        else:
            self.last_epoch = epoch if epoch is not None else self.last_epoch + 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class CosineWarmUpScheduler(LRScheduler):
    """
    Applies a warm up period to the learning rate.
    """

    def __init__(self, optimizer, warmup, total_steps, last_epoch=-1):
        self.warmup = warmup
        self.total_steps = total_steps
        super(CosineWarmUpScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        progress = self.last_epoch / self.total_steps
        if progress < self.warmup:
            return [base_lr * progress / self.warmup for base_lr in self.base_lrs]
        else:
            return [base_lr * (0.5 * (1.0 + torch.cos(math.pi + progress))) for base_lr in self.base_lrs]


def get_learning_rate_scheduler(
    type: str,
    optimizer: torch.optim.Optimizer,
    **args
    ):

    if type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, 
            **args
            )
    elif type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            **args
            )
    elif type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            **args
            )
    elif type == 'cosine_warmup':
        return CosineWarmUpScheduler(
            optimizer, 
            **args
            )
    else:
        raise NotImplementedError