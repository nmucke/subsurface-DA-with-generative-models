#from attr import dataclass
from torch import nn
import torch
from tqdm import tqdm
import pdb
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
plt.switch_backend('agg')

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



def train_WAE(
    train_dataloader: torch.utils.data.DataLoader,
    num_epochs: int,
    train_stepper: BaseTrainStepper,
    val_dataloader: torch.utils.data.DataLoader = None,
    print_progress: bool = True,
    patience: int = None,
    model_save_path: str = None,
    save_output: bool = False,
    only_input: bool = False,
) -> None:
    
    plot_batch_size = 4
    fixed_z = train_stepper._sample_latent(
        (plot_batch_size, train_stepper.model.latent_dim),
        )
    fixed_input, fixed_dynamic_input, fixed_output = next(iter(train_dataloader))
    fixed_input = fixed_input[0:plot_batch_size, 30].to(train_stepper.device)
    fixed_output = fixed_output[0:plot_batch_size, 30]

    if only_input:
        plot_data = fixed_input.to('cpu')
    else:
        plot_data = fixed_output.to('cpu')

    if fixed_dynamic_input is not None:
        fixed_dynamic_input = fixed_dynamic_input[:, 30].to(train_stepper.device)


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
        
        recon_loss_logger = MetricLogger()
        latent_loss_logger = MetricLogger()

        for i, (input_data, dynamic_input_data, output_data) in pbar:

            if only_input:
                input_data = input_data[:, 0]
            else:    
                input_data = input_data.view(-1, input_data.shape[2], input_data.shape[3], input_data.shape[4])
                output_data = output_data.view(-1, output_data.shape[2], output_data.shape[3], output_data.shape[4])
                output_data = output_data.to(device)

            if dynamic_input_data is not None:
                dynamic_input_data = dynamic_input_data.view(-1, dynamic_input_data.shape[2], dynamic_input_data.shape[3])
                dynamic_input_data = dynamic_input_data.to(device)
            
            input_data = input_data.to(device)

            if only_input:
                loss = train_stepper.train_step(
                    input_data=input_data,
                    )
            else:
                loss = train_stepper.train_step(
                    input_data=input_data,
                    output_data=output_data,
                    dynamic_input_data=dynamic_input_data,
                    )

            recon_loss_logger.update(
                loss=loss['recon_loss'],
                )
            latent_loss_logger.update(
                loss=loss['latent_loss'],
                )

            if i % 10 == 0:
                pbar.set_postfix({
                    'recon_loss': recon_loss_logger.average_loss,
                    'latent_loss': latent_loss_logger.average_loss,
                })

        # save model statedict
        save_dict = {
            'model_state_dict': train_stepper.model.state_dict(),
            'encoder_optimizer_state_dict': train_stepper.optimizer.encoder.state_dict(),
            'decoder_optimizer_state_dict': train_stepper.optimizer.decoder.state_dict(),
            'encoder_scheduler_state_dict': train_stepper.optimizer.encoder_scheduler.state_dict(),
            'decoder_scheduler_state_dict': train_stepper.optimizer.decoder_scheduler.state_dict(),
        }
        torch.save(save_dict, f'trained_models/WAE.pt')

        train_stepper.optimizer.step_scheduler(recon_loss_logger.average_loss)
        
        if save_output:

            train_stepper.model.eval()
            
            # Save generated images
            if only_input:
                generated_img = train_stepper.model.decoder(
                    latent_samples=fixed_z, 
                    )
            else:
                generated_img = train_stepper.model.decoder(
                    latent_samples=fixed_z, 
                    input_data=fixed_input,
                    dynamic_input_data=fixed_dynamic_input,
                    )
            generated_img = generated_img.to('cpu').detach()

            train_stepper.model.train()

            # Plot the images
            fig = plt.figure(figsize=(plot_batch_size, plot_batch_size))
            plt.axis("off")
            for i in range(plot_batch_size):
                fig.add_subplot(4, plot_batch_size, i+1)
                plt.imshow(
                    plot_data[i, 0, :, :],
                    cmap='viridis',
                    interpolation='none',
                    )
                fig.add_subplot(4, plot_batch_size, i+plot_batch_size+1)
                plt.imshow(
                    generated_img[i, 0, :, :],
                    cmap='viridis',
                    interpolation='none',
                    )
            for i in range(plot_batch_size):
                fig.add_subplot(4, plot_batch_size, i+1+2*plot_batch_size)
                plt.imshow(
                    plot_data[i, 1, :, :],
                    cmap='viridis',
                    interpolation='none',
                    )
                fig.add_subplot(4, plot_batch_size, i+1+3*plot_batch_size)
                plt.imshow(
                    generated_img[i, 1, :, :],
                    cmap='viridis',
                    interpolation='none',
                    )
            
            plt.savefig(f'wae_output/generated_images_{epoch}.png')
            plt.close()
            #generated_img = make_grid(images_to_save[:, 0:1, :, :], nrow=train_dataloader.batch_size, normalize=True)
            #save_image(generated_img, f'gan_output/generated_images_{epoch}.png')
            
        
    if patience is None:
        train_stepper.save_model(model_save_path)
        