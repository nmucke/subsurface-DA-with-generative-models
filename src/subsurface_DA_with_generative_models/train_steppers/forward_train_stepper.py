import pdb
import pickle
import numpy as np
import torch
import torch.nn as nn
import os
from torch import autocast
import matplotlib.pyplot as plt

from subsurface_DA_with_generative_models.optimizers.GAN_optimizer import GANOptimizer
from subsurface_DA_with_generative_models.plotting_utils import plot_output
from subsurface_DA_with_generative_models.train_steppers.base_train_stepper import BaseTrainStepper
from subsurface_DA_with_generative_models.data_handling.data_utils import prepare_batch



class ForwardTrainStepper(BaseTrainStepper):

    def __init__(
        self,
        model: nn.Module,
        optimizer: GANOptimizer,
        model_save_path: str,
        **kwargs,
    ) -> None:

        self.model = model
        self.optimizer = optimizer

        self.device = model.device

        self.generator_scaler = torch.cuda.amp.GradScaler()
        self.critic_scaler = torch.cuda.amp.GradScaler()

        self.epoch_count = 0

        self.model_save_path = model_save_path

        self.best_loss = float('inf')

    def start_epoch(self) -> None:
        self.epoch_count += 1
        self.model.train()

    def end_epoch(self, val_loss: float = None) -> None:

        self.optimizer.step_scheduler(val_loss)

        if val_loss < self.best_loss:
            save_dict = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.optimizer.state_dict(),
            }

            if self.optimizer.args['scheduler_args'] is not None:
                save_dict['scheduler_state_dict'] = \
                    self.optimizer.scheduler.state_dict()

            torch.save(save_dict, f'{self.model_save_path}/model.pt')

            self.best_loss = val_loss

            # save best loss to file
            with open(f'{self.model_save_path}/loss.txt', 'w') as f:
                f.write(str(self.best_loss))
                                     
 
    def _train_step(
        self, 
        static_point_parameters: torch.Tensor = None,
        static_spatial_parameters: torch.Tensor = None,
        dynamic_point_parameters: torch.Tensor = None,
        dynamic_spatial_parameters: torch.Tensor = None,
        output_variables: torch.Tensor = None,
    ):
        self.optimizer.zero_grad()

        pred_output_variables = self.model(
            static_point_parameters=static_point_parameters,
            static_spatial_parameters=static_spatial_parameters,
            dynamic_point_parameters=dynamic_point_parameters,
            dynamic_spatial_parameters=dynamic_spatial_parameters
        )
        
        MSE_loss = nn.MSELoss()(pred_output_variables, output_variables)

        # update generator
        MSE_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        return MSE_loss.detach().item()

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
        
        MSE_loss = self._train_step(
            static_point_parameters=static_point_parameters,
            static_spatial_parameters=static_spatial_parameters,
            dynamic_point_parameters=dynamic_point_parameters,
            dynamic_spatial_parameters=dynamic_spatial_parameters,
            output_variables=output_variables,
        )

        return {
            'MSE_loss': MSE_loss,
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
                
                generated_output_data = self.model(
                    static_point_parameters=static_point_parameters,
                    static_spatial_parameters=static_spatial_parameters,
                    dynamic_point_parameters=dynamic_point_parameters,
                    dynamic_spatial_parameters=dynamic_spatial_parameters
                    )#.reshape(output_variables.shape)
                
                loss = nn.MSELoss()(generated_output_data, output_variables)
                total_loss += loss.detach().item()
                num_batches += 1

        print(f'Val loss: {total_loss / num_batches: 0.4f}, epoch: {self.epoch_count}')

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
        
        # Load up preprocessor
        with open('trained_preprocessors/preprocessor_64.pkl', 'rb') as f:
            preprocessor = pickle.load(f)

        with torch.no_grad():
            generated_output_data = self.model(
                static_point_parameters=static_point_parameters,
                static_spatial_parameters=static_spatial_parameters,
                dynamic_point_parameters=dynamic_point_parameters,
                dynamic_spatial_parameters=dynamic_spatial_parameters
                )#.reshape(output_variables.shape)

        plot_time = 30
        plot_x = 20
        plot_y = 10

        output_variables = output_variables.cpu()
        generated_output_data = generated_output_data.cpu()

        for i in range(output_variables.shape[0]):
            output_variables[i] = preprocessor.output.inverse_transform(output_variables[i])
            generated_output_data[i] = preprocessor.output.inverse_transform(generated_output_data[i])

        output_variables = output_variables.numpy()
        generated_output_data = generated_output_data.numpy()

        plot_path = f'{plot_path}/plot_{self.epoch_count}'

        plot_output(
            generated_output_data=generated_output_data,
            output_variables=output_variables,
            plot_path=plot_path,
            plot_time=plot_time,
            plot_x_y=(plot_x, plot_y),
        )
            