import pdb
import numpy as np
import torch
import torch.nn as nn
import os
from torch import autocast
import matplotlib.pyplot as plt

from subsurface_DA_with_generative_models.optimizers.GAN_optimizer import GANOptimizer
from subsurface_DA_with_generative_models.train_steppers.base_train_stepper import BaseTrainStepper

def MMD_loss(
    x: torch.Tensor, 
    y: torch.Tensor,
    kernel: str = "multiscale",
    device: str = "cpu",
    ) -> torch.Tensor:
    """
    Emprical maximum mean discrepancy. The lower the result, 
    the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
        '''
        C = 2*x.shape[-1]*1
        XX += C * (C + dxx)**-1
        YY += C * (C + dyy)**-1
        XY += C * (C + dxy)**-1
        '''
            
    if kernel == "rbf":
      
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)

    return torch.mean(XX + YY - 2. * XY)



class WAETrainStepper(BaseTrainStepper):

    def __init__(
        self,
        model: nn.Module,
        optimizer: GANOptimizer,
        model_save_path: str,
        MMD_regu: float,
    ) -> None:

        self.model = model
        self.optimizer = optimizer

        self.MMD_regu = MMD_regu

        self.device = model.device

        self.critic_train_count = 0

        self.generator_scaler = torch.cuda.amp.GradScaler()
        self.critic_scaler = torch.cuda.amp.GradScaler()

        self.epoch_count = 0

        self.model_save_path = model_save_path

        self.best_loss = float('inf')

        self.MSE_loss = nn.MSELoss()

    def _sample_latent(self, shape: torch.Tensor) -> torch.Tensor:
        return torch.randn(shape, device=self.device)
    
    def start_epoch(self) -> None:
        self.epoch_count += 1
        self.generator_loss = 0 

    def end_epoch(self, val_loss: float = None) -> None:

        self.optimizer.step_scheduler()

        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'decoder_optimizer_state_dict': self.optimizer.generator.state_dict(),
            'encoder_optimizer_state_dict': self.optimizer.critic.state_dict(),
        }

        if self.optimizer.args['scheduler_args'] is not None:
            save_dict['decoder_scheduler_state_dict'] = \
                self.optimizer.generator_scheduler.state_dict()
            save_dict['encoder_scheduler_state_dict'] = \
                self.optimizer.critic_scheduler.state_dict()

        torch.save(save_dict, f'{self.model_save_path}/model.pt')

        # save best loss to file
        with open(f'{self.model_save_path}/loss.txt', 'w') as f:
            f.write(str(self.best_loss))
                                    
    def train_step(
        self,
        batch: dict,
        ) -> dict:

        # unpack batch
        static_spatial_parameters = batch.get('static_spatial_parameters')

        # send to device
        static_spatial_parameters = static_spatial_parameters.to(self.device)

        generated_latent = self.model.encoder(static_spatial_parameters)

        generated_output_data = self.model.generator(generated_latent)

        true_latent = self._sample_latent(
            shape=(generated_latent.shape)
        )

        # compute loss
        latent_loss = MMD_loss(generated_latent, true_latent)

        recon_loss = self.MSE_loss(
            generated_output_data, 
            static_spatial_parameters
            )
        
        loss = recon_loss + self.MMD_regu*latent_loss




            
        

    def val_step(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> None:
        
        return None

        '''
        
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:

                # unpack batch
                static_spatial_parameters = batch.get('static_spatial_parameters')

                # send to device
                static_spatial_parameters = static_spatial_parameters.to(self.device)
                
                generated_output_data = self.model.generator(
                    static_spatial_parameters=static_spatial_parameters,
                    )
                
                total_loss += loss.detach().item()
                num_batches += 1

        print(f'Val loss: {total_loss / num_batches: 0.4f}, epoch: {self.epoch_count}')

        return total_loss / num_batches
        '''
    
    def plot(
        self, 
        batch: dict,
        plot_path: str = None,
        ):
        # unpack batch


        # unpack batch
        static_spatial_parameters = batch.get('static_spatial_parameters')


        # send to device
        static_spatial_parameters = static_spatial_parameters.to(self.device)

        latent = self._sample_latent(
            shape=(static_spatial_parameters.shape[0], self.model.latent_dim)
            )
        
        with torch.no_grad():
            generated_output_data = self.model.generator(
                latent=latent
            )

        generated_output_data = generated_output_data.detach().cpu().numpy()
        static_spatial_parameters = static_spatial_parameters.detach().cpu().numpy()

        plt.figure(figsize=(10, 10))
        for i in range(4):
            plt.subplot(4, 4, i+1)
            plt.imshow(generated_output_data[i, 0])
            plt.colorbar()
            plt.title(f'Generated')


            plt.subplot(4, 4, i+4+1)
            plt.imshow(static_spatial_parameters[i, 0])
            plt.colorbar()
            plt.title(f'True')


            plt.subplot(4, 4, i+8+1)
            plt.imshow(generated_output_data[i, 1])
            plt.colorbar()
            plt.title(f'Generated')

            plt.subplot(4, 4, i+12+1)
            plt.imshow(static_spatial_parameters[i, 1])
            plt.colorbar()
            plt.title(f'True')

        plt.savefig(f'{plot_path}/plot_{self.epoch_count}.png')     
        
        plt.close()   
