import pdb
import torch
import torch.nn as nn


class Preprocessor():
    def __init__(
        self,
        type: str = 'MinMax',
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
        
        TransformerClass = MinMaxTransformer if type == 'MinMax' else GaussianNormalizer

        
        if self.static_point:
            self.static_point = TransformerClass()    

        if self.static_spatial:
            self.static_spatial = TransformerClass()

        if self.dynamic_point:
            self.dynamic_point = TransformerClass()

        if self.dynamic_spatial:
            self.dynamic_spatial = TransformerClass()

        if self.output:
            self.output = TransformerClass()

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

    

class GaussianNormalizer():
    def __init__(self, eps=0.0001) -> None:
        self.mean = None
        self.std = None
        self.num_channels = None
        self.fst_moment = None
        self.snd_moment = None
        self.count = 0
        self.eps = eps

    def transform(self, data):
        for i in range(self.num_channels):  
            data[i] = (data[i] - self.mean[i]) / (self.std[i] + self.eps)
        return data        

    def inverse_transform(self, data):
        for i in range(self.num_channels):
            data[i] = data[i] * (self.std[i] + self.eps) + self.mean[i]
        return data

    def partial_fit(self, batch):
        if self.num_channels is None:
            self.num_channels = batch.shape[1]
            self.mean = torch.zeros(self.num_channels)
            self.std = torch.zeros(self.num_channels)
            self.fst_moment = torch.zeros(self.num_channels)
            self.snd_moment = torch.zeros(self.num_channels)

        for i in range(batch.shape[1]):
            batch_mean = torch.mean(batch[:, i])
            batch_var = torch.var(batch[:, i])
            
            batch_fst_moment = batch_mean
            batch_snd_moment = batch_var + batch_mean ** 2

            # Update the count
            self.count += 1

            # Update the first and second moments
            self.fst_moment[i] = (self.fst_moment[i] * (self.count - 1) + batch_fst_moment) / self.count
            self.snd_moment[i] = (self.snd_moment[i] * (self.count - 1) + batch_snd_moment) / self.count

            # Compute the new mean and standard deviation
            self.mean[i] = self.fst_moment[i]
            self.std[i] = torch.sqrt(self.snd_moment[i] - self.fst_moment[i] ** 2 + self.eps)

        

