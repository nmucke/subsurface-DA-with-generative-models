import pdb
import numpy as np
import torch
import torch.nn as nn
import os
from torch import autocast
import matplotlib.pyplot as plt

from subsurface_DA_with_generative_models.optimizers.GAN_optimizer import GANOptimizer
from subsurface_DA_with_generative_models.train_steppers.base_train_stepper import BaseTrainStepper

def prepare_batch(batch: dict, device: str) -> dict:

    # unpack batch
    static_point_parameters = batch.get('static_point_parameters')
    static_spatial_parameters = batch.get('static_spatial_parameters')
    dynamic_point_parameters = batch.get('dynamic_point_parameters')
    dynamic_spatial_parameters = batch.get('dynamic_spatial_parameters')
    output_variables = batch.get('output_variables')

    # get dimensions sizes
    num_x = static_spatial_parameters.shape[2]
    num_y = static_spatial_parameters.shape[3]
    num_time_steps = dynamic_point_parameters.shape[-1]

    # send to device
    if static_point_parameters is not None:
        static_point_parameters = static_point_parameters.to(device)
    if static_spatial_parameters is not None:
        static_spatial_parameters = static_spatial_parameters.to(device)
    if dynamic_point_parameters is not None:
        dynamic_point_parameters = dynamic_point_parameters.to(device)
    if dynamic_spatial_parameters is not None:
        dynamic_spatial_parameters = dynamic_spatial_parameters.to(device)
    if output_variables is not None:
        output_variables = output_variables.to(device)

    return (
        static_point_parameters,
        static_spatial_parameters,
        dynamic_point_parameters,
        dynamic_spatial_parameters,
        output_variables,
    )

class ForwardGANTrainStepper(BaseTrainStepper):

    def __init__(
        self,
        model: nn.Module,
        optimizer: GANOptimizer,
        gradient_penalty_regu: str,
        num_critic_steps: int,
        model_save_path: str,
        GAN_regularization: float = 1e-3,
        with_GAN_loss: bool = False,
    ) -> None:

        self.model = model
        self.optimizer = optimizer
        self.gradient_penalty_regu = gradient_penalty_regu
        self.GAN_regularization = GAN_regularization
        self.with_GAN_loss = with_GAN_loss

        self.device = model.device

        self.critic_train_count = 0
        self.num_critic_steps = num_critic_steps

        self.generator_scaler = torch.cuda.amp.GradScaler()
        self.critic_scaler = torch.cuda.amp.GradScaler()

        self.epoch_count = 0

        self.model_save_path = model_save_path

        self.best_loss = float('inf')

    def start_epoch(self) -> None:
        self.epoch_count += 1

    def end_epoch(self, val_loss: float = None) -> None:

        self.optimizer.step_scheduler(val_loss)

        if val_loss < self.best_loss:
            save_dict = {
                'model_state_dict': self.model.state_dict(),
                'generator_optimizer_state_dict': self.optimizer.generator.state_dict(),
                'critic_optimizer_state_dict': self.optimizer.critic.state_dict(),
            }

            if self.optimizer.args['scheduler_args'] is not None:
                save_dict['generator_scheduler_state_dict'] = \
                    self.optimizer.generator_scheduler.state_dict()
                save_dict['critic_scheduler_state_dict'] = \
                    self.optimizer.critic_scheduler.state_dict(),

            torch.save(save_dict, f'{self.model_save_path}/model.pt')

            self.best_loss = val_loss

            # save best loss to file
            with open(f'{self.model_save_path}/loss.txt', 'w') as f:
                f.write(str(self.best_loss))
                                     
    def _compute_gradient_penalty(
        self, 
        static_point_parameters: torch.Tensor = None,
        static_spatial_parameters: torch.Tensor = None,
        dynamic_point_parameters: torch.Tensor = None,
        dynamic_spatial_parameters: torch.Tensor = None,
        real_output_variables: torch.Tensor = None,
        fake_output_variables: torch.Tensor = None,
    ):
        
        # Random weight term for interpolation between real and fake data
        alpha = torch.randn(real_output_variables.size(), device=self.device)
        
        # Get random interpolation between real and fake data
        interpolates = (
            alpha * real_output_variables + ((1 - alpha) * fake_output_variables)
            ).requires_grad_(True)
        
        model_interpolates = self.model.critic(
            output_variables=interpolates,
            static_point_parameters=static_point_parameters,
            static_spatial_parameters=static_spatial_parameters,
            dynamic_point_parameters=dynamic_point_parameters,
            dynamic_spatial_parameters=dynamic_spatial_parameters,
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
        static_point_parameters: torch.Tensor = None,
        static_spatial_parameters: torch.Tensor = None,
        dynamic_point_parameters: torch.Tensor = None,
        dynamic_spatial_parameters: torch.Tensor = None,
        output_variables: torch.Tensor = None,
    ):
        self.model.critic.train()
        self.model.generator.eval()

        # compute critic loss for real data
        critic_output_real_data = self.model.critic(
            output_variables=output_variables,
            static_point_parameters=static_point_parameters,
            static_spatial_parameters=static_spatial_parameters,
            dynamic_point_parameters=dynamic_point_parameters,
            dynamic_spatial_parameters=dynamic_spatial_parameters,
            )
        
        generated_output_variables = self.model.generator(
            static_point_parameters=static_point_parameters,
            static_spatial_parameters=static_spatial_parameters,
            dynamic_point_parameters=dynamic_point_parameters,
            dynamic_spatial_parameters=dynamic_spatial_parameters
            ).reshape(output_variables.shape)
        
        critic_output_fake_data = self.model.critic(
            output_variables=generated_output_variables,
            static_point_parameters=static_point_parameters,
            static_spatial_parameters=static_spatial_parameters,
            dynamic_point_parameters=dynamic_point_parameters,
            dynamic_spatial_parameters=dynamic_spatial_parameters,
            )

        critic_loss_real = torch.mean(critic_output_real_data)
        critic_loss_fake = torch.mean(critic_output_fake_data)

        # compute gradient penalty
        gradient_penalty = self._compute_gradient_penalty(
            static_point_parameters=static_point_parameters,
            static_spatial_parameters=static_spatial_parameters,
            dynamic_point_parameters=dynamic_point_parameters,
            dynamic_spatial_parameters=dynamic_spatial_parameters,
            real_output_variables=output_variables,
            fake_output_variables=generated_output_variables,
            )
        
        # compute critic loss
        critic_loss = -critic_loss_real + critic_loss_fake \
            + self.gradient_penalty_regu * gradient_penalty

        # update critic
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), 0.5)
        self.optimizer.critic.step()

        return critic_loss.detach().item(), gradient_penalty.detach().item()
 
    def _generator_train_step(
        self, 
        static_point_parameters: torch.Tensor = None,
        static_spatial_parameters: torch.Tensor = None,
        dynamic_point_parameters: torch.Tensor = None,
        dynamic_spatial_parameters: torch.Tensor = None,
        output_variables: torch.Tensor = None,
    ):
        self.model.critic.eval()
        self.model.generator.train()
        self.optimizer.generator.zero_grad()

        generated_output_variables = self.model.generator(
            static_point_parameters=static_point_parameters,
            static_spatial_parameters=static_spatial_parameters,
            dynamic_point_parameters=dynamic_point_parameters,
            dynamic_spatial_parameters=dynamic_spatial_parameters
            ).reshape(output_variables.shape)
        
        if self.with_GAN_loss:
            critic_output_data = self.model.critic(
                output_variables=generated_output_variables,
                static_point_parameters=static_point_parameters,
                static_spatial_parameters=static_spatial_parameters,
                dynamic_point_parameters=dynamic_point_parameters,
                dynamic_spatial_parameters=dynamic_spatial_parameters
                )
        
            GAN_loss = -torch.mean(critic_output_data)
        MSE_loss = nn.MSELoss()(generated_output_variables, output_variables)

        generator_loss = MSE_loss 
        
        if self.with_GAN_loss:
            generator_loss += self.GAN_regularization * GAN_loss 
        else:
            GAN_loss = torch.tensor(0.0)

        # update generator
        generator_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.generator.parameters(), 0.5)
        self.optimizer.generator.step()

        return GAN_loss.detach().item(), MSE_loss.detach().item()

    def train_step(
        self,
        batch: dict,
        ) -> dict:

        # unpack batch
        static_point_parameters, static_spatial_parameters, \
        dynamic_point_parameters, dynamic_spatial_parameters, output_variables = \
            prepare_batch(
                batch=batch,
                device=self.device,
            )
        
        if self.with_GAN_loss:        
            # train critic
            self.optimizer.critic.zero_grad()        
            critic_loss, gradient_penalty = self._critic_train_step(
                static_point_parameters=static_point_parameters,
                static_spatial_parameters=static_spatial_parameters,
                dynamic_point_parameters=dynamic_point_parameters,
                dynamic_spatial_parameters=dynamic_spatial_parameters,
                output_variables=output_variables,
            )
        else:
            critic_loss = None
            gradient_penalty = None

        self.critic_train_count += 1
        

        # train generator
        if self.critic_train_count == self.num_critic_steps or not self.with_GAN_loss:
            GAN_loss, MSE_loss = self._generator_train_step(
                static_point_parameters=static_point_parameters,
                static_spatial_parameters=static_spatial_parameters,
                dynamic_point_parameters=dynamic_point_parameters,
                dynamic_spatial_parameters=dynamic_spatial_parameters,
                output_variables=output_variables,
            )
            self.critic_train_count = 0

            return {
                'GAN_loss': GAN_loss,
                'MSE_loss': MSE_loss,
                'critic_loss': critic_loss,
                'gradient_penalty': gradient_penalty
            }
        else:
            return {
                'GAN_loss': None,
                'MSE_loss': None,
                'critic_loss': critic_loss,
                'gradient_penalty': gradient_penalty
            }

    def val_step(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> None:
        
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                # unpack batch
                static_point_parameters, static_spatial_parameters, \
                dynamic_point_parameters, dynamic_spatial_parameters, output_variables = \
                    prepare_batch(
                        batch=batch,
                        device=self.device,
                    )
                
                generated_output_data = self.model.generator(
                    static_point_parameters=static_point_parameters,
                    static_spatial_parameters=static_spatial_parameters,
                    dynamic_point_parameters=dynamic_point_parameters,
                    dynamic_spatial_parameters=dynamic_spatial_parameters
                    ).reshape(output_variables.shape)
                
                loss = nn.MSELoss()(generated_output_data, output_variables)
                total_loss += loss.detach().item()
                num_batches += 1

        print(f'Validation loss: {total_loss / num_batches}')

        return total_loss / num_batches
    
    def plot(
        self, 
        batch: dict,
        plot_path: str = None,
        ):
        # unpack batch
        static_point_parameters, static_spatial_parameters, \
        dynamic_point_parameters, dynamic_spatial_parameters, output_variables = \
            prepare_batch(
                batch=batch,
                device=self.device,
            )
        with torch.no_grad():
            generated_output_data = self.model.generator(
                static_point_parameters=static_point_parameters,
                static_spatial_parameters=static_spatial_parameters,
                dynamic_point_parameters=dynamic_point_parameters,
                dynamic_spatial_parameters=dynamic_spatial_parameters
                ).reshape(output_variables.shape)

        plot_time = 30
        plot_x = 40
        plot_y = 10

        plt.figure(figsize=(10, 10))
        plt.subplot(3, 3, 1)
        plt.imshow(generated_output_data[0, 0, plot_time, :, :].cpu().numpy())
        plt.colorbar()
        plt.title('Generated pressure')

        plt.subplot(3, 3, 2)
        plt.imshow(output_variables[0, 0, plot_time, :, :].cpu().numpy())
        plt.colorbar()
        plt.title('True pressure')

        plt.subplot(3, 3, 3)
        plt.imshow(np.abs(generated_output_data[0, 0, plot_time, :, :].cpu().numpy() - output_variables[0, 0, plot_time, :, :].cpu().numpy()))
        plt.colorbar()
        plt.title('Pressure Error')

        plt.subplot(3, 3, 4)
        plt.imshow(generated_output_data[0, 1, plot_time, :, :].cpu().numpy())
        plt.colorbar()
        plt.title('Generated CO2')

        plt.subplot(3, 3, 5)
        plt.imshow(output_variables[0, 1, plot_time, :, :].cpu().numpy())
        plt.colorbar()
        plt.title('True CO2')

        plt.subplot(3, 3, 6)
        plt.imshow(np.abs(generated_output_data[0, 1, plot_time, :, :].cpu().numpy() - output_variables[0, 1, plot_time, :, :].cpu().numpy()))
        plt.colorbar()
        plt.title('CO2 Error')

        plt.subplot(3, 3, 7)
        plt.plot(generated_output_data[0, 1, :, plot_x, plot_y].cpu().numpy(), label='Generated CO2')
        plt.plot(output_variables[0, 1, :, plot_x, plot_y].cpu().numpy(), label='True CO2')
        plt.legend()
        plt.grid()
        plt.title(f'CO2 at ({plot_x}, {plot_y})')

        plt.subplot(3, 3, 8)
        plt.plot(generated_output_data[0, 0, :, plot_x, plot_y].cpu().numpy(), label='Generated pressure')
        plt.plot(output_variables[0, 0, :, plot_x, plot_y].cpu().numpy(), label='True pressure')
        plt.legend()
        plt.grid()
        plt.title(f'Pressure at ({plot_x}, {plot_y})')

        plt.savefig(f'{plot_path}/plot_{self.epoch_count}.png')     
        
        plt.close()   
