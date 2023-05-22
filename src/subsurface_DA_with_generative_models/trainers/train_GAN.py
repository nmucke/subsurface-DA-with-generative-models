#from attr import dataclass
from torch import nn
import torch
from tqdm import tqdm
import pdb
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

from subsurface_DA_with_generative_models.train_steppers.base_train_stepper import BaseTrainStepper


class EarlyStopping:
    def __call__(self) -> None:
        num_non_improving_epochs: int = 0
        best_loss: float = float('inf')
        patience: int = 10

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



def train_GAN(
    train_dataloader: torch.utils.data.DataLoader,
    num_epochs: int,
    train_stepper: BaseTrainStepper,
    val_dataloader: torch.utils.data.DataLoader = None,
    print_progress: bool = True,
    patience: int = None,
    model_save_path: str = None,
    save_output: bool = False,
) -> None:
    
    fixed_z = train_stepper._sample_latent(
        (train_dataloader.batch_size, train_stepper.model.latent_dim),
        )
    fixed_input = next(iter(train_dataloader))[0]
    fixed_input = fixed_input[:, 40]

    if patience is not None:
        early_stopper = EarlyStopping(patience=patience)

    device = train_stepper.device

    for epoch in range(num_epochs):

        if print_progress:
            pbar = tqdm(
                    enumerate(train_dataloader),
                    total=int(len(train_dataloader.dataset)/train_dataloader.batch_size),
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'
                )
        else:
            pbar = enumerate(train_dataloader)
        
        generator_loss_logger = MetricLogger()
        critic_loss_logger = MetricLogger()
        critic_grad_penalty_logger = MetricLogger()

        for i, (input_data, output_data) in pbar:

            input_data = input_data.view(-1, input_data.shape[2], input_data.shape[3], input_data.shape[4])
            output_data = output_data.view(-1, output_data.shape[2], output_data.shape[3], output_data.shape[4])
            
            input_data = input_data.to(device)
            output_data = output_data.to(device)

            loss = train_stepper.train_step(
                input_data=input_data,
                output_data=output_data,
                )

            if loss['generator_loss'] is not None:
                generator_loss_logger.update(
                    loss=loss['generator_loss'],
                    )
            critic_loss_logger.update(
                loss=loss['critic_loss'],
                )
            critic_grad_penalty_logger.update(
                loss=loss['gradient_penalty'],
                )

            if i % 10 == 0:
                pbar.set_postfix({
                    'generator_loss': generator_loss_logger.average_loss,
                    'critic_loss': critic_loss_logger.average_loss,
                    'gradient_penalty': critic_grad_penalty_logger.average_loss,
                })
        
        if save_output:
            # Save generated images
            generated_img = train_stepper.model.generator(fixed_z, fixed_input)
            generated_img = generated_img.to('cpu').detach()
            generated_img = make_grid(output_data[4, 1, 32, 32])
            save_image(generated_img, f'gan_output/generated_images_{epoch}.png')
            
        #train_stepper.step_scheduler()
        
    if patience is None:
        train_stepper.save_model(model_save_path)
        