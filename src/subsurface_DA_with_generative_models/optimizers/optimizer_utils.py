import torch
import torch.nn as nn


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
    else:
        raise NotImplementedError