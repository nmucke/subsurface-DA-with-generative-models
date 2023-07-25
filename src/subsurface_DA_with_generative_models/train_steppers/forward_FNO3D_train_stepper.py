import pdb
import pickle
import numpy as np
import torch
import torch.nn as nn
import os
from torch import autocast
import matplotlib.pyplot as plt
import torch.nn.functional as F
from subsurface_DA_with_generative_models.optimizers.FNO3D_optimizer import FNO3dOptimizer
from subsurface_DA_with_generative_models.plotting_utils import plot_output
from subsurface_DA_with_generative_models.train_steppers.base_train_stepper import BaseTrainStepper


#loss function with rel/abs Lp loss
class LpLoss(nn.Module):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):

        diff_norms = torch.norm(x - y, self.p, 1)
        y_norms = torch.norm(y, self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def forward(self, x, y):
        return self.rel(x, y)

def prepare_batch(batch: dict, device: str) -> dict:

    # unpack batch
    static_point_parameters = batch.get('static_point_parameters')
    static_spatial_parameters = batch.get('static_spatial_parameters')
    dynamic_point_parameters = batch.get('dynamic_point_parameters')
    dynamic_spatial_parameters = batch.get('dynamic_spatial_parameters')
    output_variables = batch.get('output_variables')

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

class FNO3DTrainStepper(BaseTrainStepper):
    def __init__(
        self,
        model: nn.Module,
        optimizer: FNO3dOptimizer,        
        model_save_path: str,
        device: str = 'cuda',
    ) -> None:

        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_count = 0
        self.epoch_count = 0
        self.model_save_path = model_save_path
        self.best_loss = float('inf')
        self.myloss = LpLoss(size_average=False)

    def start_epoch(self) -> None:
        self.epoch_count += 1

    def end_epoch(self, val_loss: float = None) -> None:

            if val_loss is not None:
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


    def _compute_loss(
        self,
        y: torch.Tensor = None,
        y_hat: torch.Tensor = None,   
    ) -> torch.Tensor:    
        l2 = self.myloss(y_hat.view(self.batch_size, -1), y.view(self.batch_size, -1))
        return l2
                                    
    def train_step(
        self,
        batch: dict,        
        ) -> dict:       

        # unpack batch
        static_point_parameters, static_spatial_parameters, \
        dynamic_point_parameters, dynamic_spatial_parameters, \
        output_variables = \
            prepare_batch(
                batch=batch,
                device=self.device,
            )
        self.batch_size = static_spatial_parameters.shape[0] 
                   
        # train 
        self.optimizer.zero_grad()   
        #with autocast():
        y_hat = self.model(
            static_point_parameters=static_point_parameters,
            static_spatial_parameters=static_spatial_parameters,
            dynamic_point_parameters=dynamic_point_parameters,
            dynamic_spatial_parameters=dynamic_spatial_parameters,
        )
        
        l2_loss = self._compute_loss(y=output_variables, y_hat=y_hat)
        l2_loss.backward()
        self.optimizer.step()     
        mse_loss = F.mse_loss(y_hat, output_variables, reduction='mean')

        self.train_count += 1       


        return {
            'train_loss': l2_loss.detach().item(),
            'train_mse': mse_loss.detach().item(),
        }

    def val_step(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> None:
        
        self.model.eval()
        total_l2_loss = 0
        total_mse_loss = 0
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
                y_hat = self.model(
                    static_point_parameters=static_point_parameters,
                    static_spatial_parameters=static_spatial_parameters,
                    dynamic_point_parameters=dynamic_point_parameters,
                    dynamic_spatial_parameters=dynamic_spatial_parameters,
                )
                l2_loss = self._compute_loss(y=output_variables, y_hat=y_hat)
                mse_loss = F.mse_loss(y_hat, output_variables, reduction='mean')    

                total_l2_loss += l2_loss.detach().item()
                total_mse_loss += mse_loss.detach().item()      

                num_batches += 1

        print(f'Validation L2 loss: {total_l2_loss / num_batches} \n  Validation MSE loss: {total_mse_loss / num_batches} \n')

        return total_l2_loss / num_batches
    
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
        with open('trained_preprocessors/preprocessor_64_pressure_output.pkl', 'rb') as f:
            preprocessor = pickle.load(f)

        with torch.no_grad():
            generated_output_data = self.model(
                static_point_parameters=static_point_parameters,
                static_spatial_parameters=static_spatial_parameters,
                dynamic_point_parameters=dynamic_point_parameters,
                dynamic_spatial_parameters=dynamic_spatial_parameters
                )#.reshape(output_variables.shape)

        plot_time = 30
        plot_x = 40
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