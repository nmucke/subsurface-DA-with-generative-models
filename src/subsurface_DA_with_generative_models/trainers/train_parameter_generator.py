#from attr import dataclass
import os
from torch import nn
import torch
from tqdm import tqdm
import pdb
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')

from subsurface_DA_with_generative_models.train_steppers.base_train_stepper import BaseTrainStepper


class EarlyStopping:
    def __init__(self) -> None:
        num_non_improving_epochs: int = 0
        best_loss: float = float('inf')
        patience: int = 10
    
    def __call__(self, loss: float) -> None:
        if loss < self.best_loss:
            self.best_loss = loss
            self.num_non_improving_epochs = 0
        else:
            self.num_non_improving_epochs += 1

            if self.num_non_improving_epochs == self.patience:
                print('Early stopping triggered')
                return True

class MetricLogger():
    def __init__(self) -> None:
        self.total_loss = 0
        self.num_batches = 1

    def update(self, loss: float):
        self.total_loss += loss
        self.num_batches += 1

    @property
    def average_loss(self):
        return self.total_loss / self.num_batches



def train_parameter_generator(
    train_dataloader: torch.utils.data.DataLoader,
    num_epochs: int,
    train_stepper: BaseTrainStepper,
    val_dataloader: torch.utils.data.DataLoader = None,
    print_progress: bool = True,
    patience: int = None,
    plot_path: str = None,
) -> None:
    

    if plot_path is not None:
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        plot_batch = next(iter(train_dataloader))
    
    # Set up early stopping
    if patience is not None:
        early_stopper = EarlyStopping(patience=patience)

    # Training loop
    for epoch in range(num_epochs):
        
        # Set up progress bar
        if print_progress:
            pbar = tqdm(
                    enumerate(train_dataloader),
                    total=int(len(train_dataloader.dataset)/train_dataloader.batch_size),
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'
                )
        else:
            pbar = enumerate(train_dataloader)
        
        
        train_stepper.start_epoch()

        # Training
        for i, batch in pbar:

            loss = train_stepper.train_step(batch=batch)

            if i % 2 == 0:
                loss_logger_dict = {}
                for key, value in loss.items():
                    loss_logger_dict[key] = value
                pbar.set_postfix(loss_logger_dict)
        
        # Validation
        if val_dataloader is not None:
            loss = train_stepper.val_step(val_dataloader)

            # Early stopping
            if patience is not None:
                stop = early_stopper(loss)
                if stop:
                    train_stepper.end_epoch(loss)
                    break
        
        train_stepper.end_epoch(loss)

        # Plotting
        if plot_path is not None:
            train_stepper.plot(
                batch=plot_batch, 
                plot_path=plot_path
                )                
