import pdb
import numpy as np
import torch
import torch.nn as nn
import os
from torch import autocast
import matplotlib.pyplot as plt

from subsurface_DA_with_generative_models.optimizers.GAN_optimizer import GANOptimizer
from subsurface_DA_with_generative_models.train_steppers.base_train_stepper import BaseTrainStepper

class ParameterGANTrainStepper(BaseTrainStepper):

    def __init__(
        self,
        model: nn.Module,
        optimizer: GANOptimizer,
        gradient_penalty_regu: str,
        num_critic_steps: int,
        model_save_path: str,
        wasserstein: bool = False,
    ) -> None:

        self.model = model
        self.optimizer = optimizer
        self.gradient_penalty_regu = gradient_penalty_regu
        self.wasserstein = wasserstein

        self.device = model.device

        self.critic_train_count = 0
        self.num_critic_steps = num_critic_steps

        self.generator_scaler = torch.cuda.amp.GradScaler()
        self.critic_scaler = torch.cuda.amp.GradScaler()

        self.epoch_count = 0

        self.model_save_path = model_save_path

        self.best_loss = float('inf')
    
    def _sample_latent(self, shape: torch.Tensor) -> torch.Tensor:
        return torch.randn(shape, device=self.device)

    def start_epoch(self) -> None:
        self.epoch_count += 1
        self.generator_loss = 0 

    def end_epoch(self, val_loss: float = None) -> None:

        self.optimizer.step_scheduler()

        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'generator_optimizer_state_dict': self.optimizer.generator.state_dict(),
            'critic_optimizer_state_dict': self.optimizer.critic.state_dict(),
        }

        if self.optimizer.args['scheduler_args'] is not None:
            save_dict['generator_scheduler_state_dict'] = \
                self.optimizer.generator_scheduler.state_dict()
            save_dict['critic_scheduler_state_dict'] = \
                self.optimizer.critic_scheduler.state_dict()

        torch.save(save_dict, f'{self.model_save_path}/model.pt')

        # save best loss to file
        with open(f'{self.model_save_path}/loss.txt', 'w') as f:
            f.write(str(self.best_loss))
                                     
    def _compute_gradient_penalty(
        self, 
        static_spatial_parameters: torch.Tensor = None,
        generated_spatial_parameters: torch.Tensor = None,
    ):
        
        # Random weight term for interpolation between real and fake data
        alpha = torch.randn(static_spatial_parameters.size(), device=self.device)
        
        # Get random interpolation between real and fake data
        interpolates = (
            alpha * static_spatial_parameters + ((1 - alpha) * generated_spatial_parameters)
            ).requires_grad_(True)
        
        model_interpolates = self.model.critic(
            static_spatial_parameters=interpolates,
            )

        grad_outputs = torch.ones(
            model_interpolates.size(), 
            device=self.device, 
            requires_grad=False
            )

        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=model_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.reshape(gradients.size(0), -1)
        gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
        
        return gradient_penalty

    def _critic_train_step(
        self, 
        static_spatial_parameters: torch.Tensor = None,
    ):
        self.model.critic.train()
        self.model.generator.eval()


        # compute critic loss for real data
        critic_output_real_data = self.model.critic(
            static_spatial_parameters=static_spatial_parameters,
            )
        
        latent = self._sample_latent(
            shape=(static_spatial_parameters.shape[0], self.model.latent_dim)
            )
        generated_spatial_parameters = self.model.generator(latent=latent)
        
        critic_output_fake_data = self.model.critic(
            static_spatial_parameters=generated_spatial_parameters,
            )

        if self.wasserstein:
            critic_loss_real = torch.mean(critic_output_real_data)
            critic_loss_fake = torch.mean(critic_output_fake_data)

            # compute gradient penalty
            gradient_penalty = self._compute_gradient_penalty(
                static_spatial_parameters=static_spatial_parameters,
                generated_spatial_parameters=generated_spatial_parameters,
                )
            # compute critic loss
            critic_loss = -critic_loss_real + critic_loss_fake \
                + self.gradient_penalty_regu * gradient_penalty

        else:
            critic_loss_real = nn.BCELoss()(
                    critic_output_real_data, 
                    torch.ones_like(critic_output_real_data)
                    )
            critic_loss_fake = nn.BCELoss()(
                    critic_output_fake_data, 
                    torch.zeros_like(critic_output_fake_data)
                    )
                
            # compute critic loss
            critic_loss = critic_loss_real + critic_loss_fake 


        # update critic
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), 0.5)
        self.optimizer.critic.step()

        if self.wasserstein:
            return critic_loss.detach().item(), gradient_penalty.detach().item()
        
        else:
            return critic_loss.detach().item(), 0.0
 
    def _generator_train_step(
        self, 
        static_spatial_parameters: torch.Tensor = None,
    ):
        self.model.critic.eval()
        self.model.generator.train()
        self.optimizer.generator.zero_grad()

        latent = self._sample_latent(
            shape=(static_spatial_parameters.shape[0], self.model.latent_dim)
            )
        
        generated_spatial_parameters = self.model.generator(latent=latent)
        
        critic_output_data = self.model.critic(
            static_spatial_parameters=generated_spatial_parameters,
            )

        if self.wasserstein:
            GAN_loss = -torch.mean(critic_output_data)
        else:
            GAN_loss = nn.BCELoss()(
                    critic_output_data, 
                    torch.ones_like(critic_output_data)
                    )
                

        # update generator
        GAN_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.generator.parameters(), 0.5)
        self.optimizer.generator.step()

        return GAN_loss.detach().item()

    def train_step(
        self,
        batch: dict,
        ) -> dict:

        # unpack batch
        static_spatial_parameters = batch.get('static_spatial_parameters')

        # send to device
        static_spatial_parameters = static_spatial_parameters.to(self.device)
            
        # train critic
        self.optimizer.critic.zero_grad()        
        critic_loss, gradient_penalty = self._critic_train_step(
            static_spatial_parameters=static_spatial_parameters,
        )

        self.critic_train_count += 1
        

        # train generator
        if self.critic_train_count == self.num_critic_steps:
            generator_loss_i = self._generator_train_step(
                static_spatial_parameters=static_spatial_parameters,
            )
            self.critic_train_count = 0

            self.generator_loss += generator_loss_i

            return {
                'generator_loss': self.generator_loss/self.epoch_count,
                'critic_loss': critic_loss,
                'gradient_penalty': gradient_penalty
            }
        else:
            return {
                'generator_loss': self.generator_loss/self.epoch_count,
                'critic_loss': critic_loss,
                'gradient_penalty': gradient_penalty
            }

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
