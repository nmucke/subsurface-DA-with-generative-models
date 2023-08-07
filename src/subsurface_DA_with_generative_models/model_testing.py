from dataclasses import dataclass
import pdb
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from subsurface_DA_with_generative_models.data_handling.data_utils import prepare_batch
from subsurface_DA_with_generative_models.metrics import (
    RMSE_loss, 
    RRMSE_loss
)
from subsurface_DA_with_generative_models.plotting_utils import plot_output
from subsurface_DA_with_generative_models.preprocessor import Preprocessor


@dataclass
class MetricLogger():
    value: float = 0.0
    num_batches: int = 1

    def update(self, loss: float):
        self.value += loss
        self.num_batches += 1

    @property
    def average(self):
        return self.value / self.num_batches

def get_metric_function(metric: str) -> nn.Module:

    if metric == 'RMSE':
        return RMSE_loss
    elif metric == 'RRMSE':
        return RRMSE_loss


class ForwardModelTester():
    def __init__(
        self,
        forward_model: nn.Module,
        preprocessor: Preprocessor = None,
        device: str = 'cuda',
        metrics: list = None,
        **kwargs,
    ) -> None:
        
        self.forward_model = forward_model
        self.device = device

        self.metrics = metrics

        self.preprocessor = preprocessor

        self.metric_functions = {
            metric: get_metric_function(metric)
            for metric in self.metrics
        }


    def _compute_metrics(
        self,
        output_pred: torch.Tensor,
        output_true: torch.Tensor,
    ) -> dict:
        
        for metric in self.metrics:
            self.metrics[metric].update(
                self.metric_functions[metric](
                    pred=output_pred,
                    target=output_true,
                ).item()
            )
        

    def test_model(
        self, 
        dataloader: DataLoader,
        print_metrics: bool = True,
        return_metrics: bool = False,
        ) -> dict:

        # initialize metrics
        self.metrics = {
            metric: MetricLogger()
            for metric in self.metrics
        }

        with torch.no_grad():
            for batch in dataloader:
                # unpack batch
                static_point_parameters, static_spatial_parameters, \
                dynamic_point_parameters, dynamic_spatial_parameters, \
                output_variables = prepare_batch(
                    batch=batch,
                    device=self.device,
                )
                
                output_pred = self.forward_model(
                    static_point_parameters=static_point_parameters,
                    static_spatial_parameters=static_spatial_parameters,
                    dynamic_point_parameters=dynamic_point_parameters,
                    dynamic_spatial_parameters=dynamic_spatial_parameters,
                )


                if self.preprocessor is not None:
                    output_variables = self.preprocessor.output.inverse_transform(output_variables[0])
                    output_pred = self.preprocessor.output.inverse_transform(output_pred[0])
                    
                # compute metrics
                self._compute_metrics(
                    output_pred=output_pred,
                    output_true=output_variables,
                )

        if print_metrics:
            print('Metrics:')
            for metric in self.metrics:
                print(f'{metric}: {self.metrics[metric].average: 0.4f}')        

        if return_metrics:
            return self.metrics

    def plot(
        self, 
        dataloader: DataLoader,
        save_path: str = None, 
        plot_time: int = 0,
        plot_x_y: tuple = (0, 0),
        sample_index: int = 0,
        show_plot: bool = True,
        ) -> None:


        batch = dataloader.dataset[sample_index]

        # unpack batch
        static_point_parameters, static_spatial_parameters, \
        dynamic_point_parameters, dynamic_spatial_parameters, \
        output_variables = prepare_batch(
            batch=batch,
            device=self.device,
        )

        if static_point_parameters is not None:
            static_point_parameters = static_point_parameters.unsqueeze(0)
        if static_spatial_parameters is not None:
            static_spatial_parameters = static_spatial_parameters.unsqueeze(0)
        if dynamic_point_parameters is not None:
            dynamic_point_parameters = dynamic_point_parameters.unsqueeze(0)
        if dynamic_spatial_parameters is not None:
            dynamic_spatial_parameters = dynamic_spatial_parameters.unsqueeze(0)
        if output_variables is not None:
            output_variables = output_variables.unsqueeze(0)
        
        with torch.no_grad():
            generated_output_data = self.forward_model(
                static_point_parameters=static_point_parameters,
                static_spatial_parameters=static_spatial_parameters,
                dynamic_point_parameters=dynamic_point_parameters,
                dynamic_spatial_parameters=dynamic_spatial_parameters
            )

        output_variables = output_variables.cpu()
        generated_output_data = generated_output_data.cpu()

        if self.preprocessor is not None:
            generated_output_data[0] = self.preprocessor.output.inverse_transform(generated_output_data[0])
            output_variables[0] = self.preprocessor.output.inverse_transform(output_variables[0])
                    
        output_variables = output_variables.numpy()
        generated_output_data = generated_output_data.numpy()

        plot_output(
            generated_output_data=generated_output_data,
            output_variables=output_variables,
            plot_path=save_path,
            plot_time=plot_time,
            plot_x_y=plot_x_y,
            show_plot=show_plot,
        )