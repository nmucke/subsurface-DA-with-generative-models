import pdb
import torch
import torch.nn as nn


class Preprocessor():
    def __init__(
        self,
        type: str,
        static_point: bool = True,
        static_spatial: bool = True,
        dynamic_point: bool = True,
        dynamic_spatial: bool = True,
        output: bool = True
    ) -> None:
        
        self.static_point = static_point
        self.static_spatial = static_spatial
        self.dynamic_point = dynamic_point
        self.dynamic_spatial = dynamic_spatial
        self.output = output
        
        if self.static_point:
            self.static_point = MinMaxTransformer()     

        if self.static_spatial:
            self.static_spatial = MinMaxTransformer()

        if self.dynamic_point:
            self.dynamic_point = MinMaxTransformer()

        if self.dynamic_spatial:
            self.dynamic_spatial = MinMaxTransformer()

        if self.output:
            self.output = MinMaxTransformer()

    def partial_fit(self, data_batch: dict, variable_names: str = None):

        if variable_names is None:
                    
            if self.static_point:
                self.static_point.partial_fit(data_batch['static_point_parameters'])

            if self.static_spatial:
                self.static_spatial.partial_fit(data_batch['static_spatial_parameters'])

            if self.dynamic_point:
                self.dynamic_point.partial_fit(data_batch['dynamic_point_parameters'])

            if self.dynamic_spatial:
                self.dynamic_spatial.partial_fit(data_batch['dynamic_spatial_parameters'])

            if self.output:
                self.output.partial_fit(data_batch['output_variables'])
        
        else:
            if self.static_point and 'static_point' in variable_names:
                self.static_point.partial_fit(data_batch['static_point_parameters'])

            if self.static_spatial and 'static_spatial' in variable_names:
                self.static_spatial.partial_fit(data_batch['static_spatial_parameters'])

            if self.dynamic_point and 'dynamic_point' in variable_names:
                self.dynamic_point.partial_fit(data_batch['dynamic_point_parameters'])

            if self.dynamic_spatial and 'dynamic_spatial' in variable_names:
                self.dynamic_spatial.partial_fit(data_batch['dynamic_spatial_parameters'])

            if self.output and 'output' in variable_names:
                self.output.partial_fit(data_batch['output_variables'])
    

      
class MinMaxTransformer():
    def __init__(
        self,
    ) -> None:
        super(MinMaxTransformer, self).__init__()

        self.num_channels = None
    
    def transform(self, data):
         
        for i in range(self.num_channels):  
            data[i] = (data[i] - self.min[i]) / (self.max[i] - self.min[i])
        return data        

    def inverse_transform(self, data):

        for i in range(self.num_channels):
            data[i] = data[i] * (self.max[i] - self.min[i]) + self.min[i]

        return data        

    def partial_fit(self, batch):

        if self.num_channels is None:
            self.num_channels = batch.shape[1]
            self.min = torch.zeros(self.num_channels)
            self.max = torch.zeros(self.num_channels)

        for i in range(batch.shape[1]):
            batch_min = torch.min(batch[:, i])
            batch_max = torch.max(batch[:, i])

            if batch_min < self.min[i]:
                self.min[i] = batch_min
            if batch_max > self.max[i]:
                self.max[i] = batch_max   

        


        

