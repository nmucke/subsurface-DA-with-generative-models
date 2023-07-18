import pdb
import numpy as np
import torch
import torch.nn as nn
import os
from torch import autocast
import matplotlib.pyplot as plt
import torch.nn.functional as F
from subsurface_DA_with_generative_models.optimizers.FNO3D_optimizer import FNO3dOptimizer
from subsurface_DA_with_generative_models.train_steppers.base_train_stepper import BaseTrainStepper


#loss function with rel/abs Lp loss
class LpLoss(object):
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
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def process_batch(batch: dict) -> dict:

    # unpack batch
    static_spatial_parameters = batch.get('static_spatial_parameters') # (num_channels, num_x, num_y)
    dynamic_point_parameters = batch.get('dynamic_point_parameters')  # (num_channels, num_time_steps)
    dynamic_spatial_parameters = batch.get('dynamic_spatial_parameters') # (num_channels, num_time_steps, num_x, num_y)
    output_variables = batch.get('output_variables') # (num_channels, num_time_steps, num_x, num_y)
    #print the name and the shape of all the tensors
    #pdb.set_trace()

    # get dimensions sizes
    num_channels = static_spatial_parameters.shape[1]
    num_time_steps = dynamic_point_parameters.shape[2]
    num_x = dynamic_spatial_parameters.shape[3]
    num_y = dynamic_spatial_parameters.shape[4]


    # We need to expand the dimensions of 'static_spatial_parameters' and 'dynamic_point_parameters' to match
    # with 'dynamic_spatial_parameters' for concatenation
  
    static_spatial = static_spatial_parameters.unsqueeze(2).repeat(1, 1, num_time_steps, 1, 1)

    dynamic_point = dynamic_point_parameters.unsqueeze(-1).unsqueeze(-1).repeat(1 ,1, 1, num_x, num_y)
   
    # Concatenating along the first dimension (channels)
    #print the shape of all the tensors

    x = torch.cat([static_spatial, dynamic_point, dynamic_spatial_parameters], dim=1)
    #permute the channels from dim 1 to last dim

    x = x.permute(0, 4, 2, 3, 1).permute(0,2,1,3,4)

    # y is the output_variables
    y = output_variables.permute(0, 4, 2, 3, 1).permute(0,2,1,3,4)

    return x, y


def prepare_batch(batch: dict, device: str):

    # Process the batch and get x and y
    x, y = process_batch(batch)

    # send to device
    if x is not None:
        x = x.to(device)
    if y is not None:
        y = y.to(device)

    return x, y

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
                        # 'optimizer_state_dict': self.optimizer.state_dict(),    
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
        #print the shape of y and y_hat tensors     
        l2 = self.myloss(y_hat.reshape(self.batch_size, -1), y.reshape(self.batch_size, -1))
        return l2
                                    
    def train_step(
        self,
        batch: dict,        
        ) -> dict:       

        # unpack batch
        x, y = prepare_batch(
                batch=batch,
                device=self.device,
            )
        self.batch_size = x.shape[0] 
                   
        # train 
        self.optimizer.zero_grad()   
        #with autocast():
        y_hat = self.model(x)           
        l2_loss = self._compute_loss(y=y, y_hat=y_hat)
        l2_loss.backward()
        self.optimizer.step()     
        mse_loss = F.mse_loss(y_hat, y, reduction='mean')    

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
                x,y= prepare_batch(
                    batch=batch,
                    device=self.device,
                )
                y_hat = self.model(x)
                l2_loss = self._compute_loss(y=y, y_hat=y_hat)
                mse_loss = F.mse_loss(y_hat, y, reduction='mean')    

                total_l2_loss += l2_loss.detach().item()
                total_mse_loss += mse_loss.detach().item()      

                num_batches += 1

        print(f'Validation L2 loss: {total_l2_loss / num_batches} \n  Validation MSE loss: {total_mse_loss / num_batches} \n')

        return total_l2_loss / num_batches
    
    def plot(
        self, 
        batch: dict,
        plot_path: str = '',
            ):
            x,output_variables = prepare_batch(
            batch=batch,
            device=self.device,
                    )
       
            with torch.no_grad():
                generated_output_data = self.model(x).reshape(output_variables.shape)

                plot_time = -1
                plot_x = 16
                plot_y = 16

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

            # plt.subplot(3, 3, 4)
            # plt.imshow(generated_output_data[0, 1, plot_time, :, :].cpu().numpy())
            # plt.colorbar()
            # plt.title('Generated CO2')

            # plt.subplot(3, 3, 5)
            # plt.imshow(output_variables[0, 1, plot_time, :, :].cpu().numpy())
            # plt.colorbar()
            # plt.title('True CO2')

            # plt.subplot(3, 3, 6)
            # plt.imshow(np.abs(generated_output_data[0, 1, plot_time, :, :].cpu().numpy() - output_variables[0, 1, plot_time, :, :].cpu().numpy()))
            # plt.colorbar()
            # plt.title('CO2 Error')

            # plt.subplot(3, 3, 7)
            # plt.plot(generated_output_data[0, 1, :, plot_x, plot_y].cpu().numpy(), label='Generated CO2')
            # plt.plot(output_variables[0, 1, :, plot_x, plot_y].cpu().numpy(), label='True CO2')
            # plt.legend()
            # plt.grid()
            # plt.title(f'CO2 at ({plot_x}, {plot_y})')

                plt.subplot(3, 3, 4)
                plt.plot(generated_output_data[0, 0, :, plot_x, plot_y].cpu().numpy(), label='Generated pressure')
                plt.plot(output_variables[0, 0, :, plot_x, plot_y].cpu().numpy(), label='True pressure')
                plt.legend()
                plt.grid()
                plt.title(f'Pressure at ({plot_x}, {plot_y})')

                plt.savefig(f'{plot_path}/plot_{self.epoch_count}.png')     
                
                plt.close()   
