import pdb
import torch
import torch.nn as nn
import matplotlib.pyplot as plt 

from subsurface_DA_with_generative_models.optimizers.GAN_optimizer import GANOptimizer
from subsurface_DA_with_generative_models.train_steppers.base_train_stepper import BaseTrainStepper


class ParemeterGANTrainStepper(BaseTrainStepper):

    def __init__(
        self,
        model: nn.Module,
        optimizer: GANOptimizer,
        gradient_penalty_regu: str,
        num_critic_steps: int,
    ) -> None:

        self.model = model
        self.optimizer = optimizer
        self.gradient_penalty_regu = gradient_penalty_regu

        self.device = model.device

        self.critic_train_count = 0
        self.num_critic_steps = num_critic_steps

    def _sample_latent(self, shape: torch.Tensor) -> torch.Tensor:
        return torch.randn(shape, device=self.device)
    
    def step_scheduler(self) -> None:
        self.optimizer.step_scheduler()

    def _compute_gradient_penalty(
        self, 
        input_data: torch.Tensor = None,
        generated_input_data: torch.Tensor = None,
    ):
        
        # Random weight term for interpolation between real and fake data
        alpha = torch.randn(input_data.size(), device=self.device)
        
        # Get random interpolation between real and fake data
        interpolates = (
            alpha * generated_input_data + ((1 - alpha) * input_data)
            ).requires_grad_(True)

        model_interpolates = self.model.critic(input_data=interpolates)

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

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
        
        return gradient_penalty

    def _critic_train_step(
        self, 
        input_data: torch.Tensor = None,
    ):
        self.model.critic.train()
        self.model.generator.eval()

        # compute critic loss for fake data
        latent_samples = self._sample_latent(
            shape=(input_data.shape[0], self.model.latent_dim)
        )

        # compute critic loss for real data
        critic_input_real_data = self.model.critic(
            input_data=input_data,
            )
        generated_input_data = self.model.generator(
            latent_samples=latent_samples,
            )
        
        critic_input_fake_data = self.model.critic(
            input_data=generated_input_data,
            )
        

        critic_loss_real = torch.mean(critic_input_real_data)
        critic_loss_fake = torch.mean(critic_input_fake_data)

        # compute gradient penalty
        gradient_penalty = self._compute_gradient_penalty(
            input_data=input_data,
            generated_input_data=generated_input_data,
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
        input_data: torch.Tensor = None,
    ):
        self.model.critic.eval()
        self.model.generator.train()

        # compute critic loss for fake data
        latent_samples = self._sample_latent(
            shape=(input_data.shape[0], self.model.latent_dim)
        )
        
        generated_input_data = self.model.generator(
            latent_samples=latent_samples, 
            )
        critic_input_data = self.model.critic(
            input_data=generated_input_data, 
            )

        generator_loss = -torch.mean(critic_input_data)

        # update generator
        generator_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.generator.parameters(), 0.5)
        self.optimizer.generator.step()

        return generator_loss.detach().item()

    def train_step(
        self,
        input_data: torch.Tensor,
        ) -> None:

        self.optimizer.critic.zero_grad()

        # train critic
        critic_loss, gradient_penalty = self._critic_train_step(
            input_data=input_data,
            )

        self.critic_train_count += 1

        # train generator
        if self.critic_train_count == self.num_critic_steps:
            self.optimizer.generator.zero_grad()
            generator_loss = self._generator_train_step(
                input_data=input_data,
                )
            self.critic_train_count = 0

            return {
                'generator_loss': generator_loss,
                'critic_loss': critic_loss,
                'gradient_penalty': gradient_penalty,
                'MSE_loss': 0,
            }
        else:
            return {
                'generator_loss': None,
                'critic_loss': critic_loss,
                'gradient_penalty': gradient_penalty,
                'MSE_loss': 0,
            }

    def val_step(
        self,
        output_data: torch.Tensor,
        input_data: torch.Tensor,
    ) -> None:
        
        return {
            'gen_loss': 0,
            'critic_loss': 0
        }

    def save_model(self, path: str) -> None:
        torch.save(self.model, path)