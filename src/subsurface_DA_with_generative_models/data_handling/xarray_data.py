import os
import pdb
import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import pickle
import matplotlib.pyplot as plt

from subsurface_DA_with_generative_models.data_handling.data_utils import (
    get_static_point_parameters,
    get_static_spatial_parameters,
    get_dynamic_point_parameters,
    get_dynamic_spatial_parameters,
    get_output_variables,
    add_meshgrids_to_data,
)
from subsurface_DA_with_generative_models.preprocessor import Preprocessor

    
class XarrayDataset(Dataset):
    def __init__(
        self, 
        data_folder: str = 'results64',
        parameter_vars: dict = {
            'static_point': None,
            'static_spatial': ['Por', 'Perm'],
            'dynamic_point': ['gas_rate'],
            'dynamic_spatial': ['time_encoding'],
        },
        output_vars: list = ['Pressure', 'CO2'],
        preprocessor: Preprocessor =  None,
        num_samples: int = -1,
        ):

        self.data_folder = data_folder
        self.parameter_vars = parameter_vars
        self.output_vars = output_vars

        self.file_list = os.listdir(self.data_folder)
        self.file_list = self.file_list[:num_samples]

        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):

        # Load data
        file_path = os.path.join(self.data_folder, self.file_list[idx])
        data = xr.open_dataset(file_path)

        # Add space encodings
        if 'x_encoding' in self.parameter_vars['static_spatial']:
            data = add_meshgrids_to_data(data, x=True, y=True, time=False)

        # Add time encodings
        if 'time_encoding' in self.parameter_vars['dynamic_spatial']:
            data = add_meshgrids_to_data(data, x=False, y=False, time=True)

        # Get parameters
        static_point_parameters = get_static_point_parameters(
            data=data, 
            static_point_vars=self.parameter_vars['static_point']
        ) # (num_channels, 1)

        static_spatial_parameters = get_static_spatial_parameters(
            data=data, 
            static_spatial_vars=self.parameter_vars['static_spatial']
        ) # (num_channels, num_x, num_y)

        dynamic_point_parameters = get_dynamic_point_parameters(
            data=data, 
            dynamic_point_vars=self.parameter_vars['dynamic_point']
        ) # (num_channels, num_time_steps)

        dynamic_spatial_parameters = get_dynamic_spatial_parameters(
            data=data, 
            dynamic_spatial_vars=self.parameter_vars['dynamic_spatial']
        ) # (num_channels, num_time_steps, num_x, num_y)

        output_variables = get_output_variables(
            data=data,
            output_vars=self.output_vars
        ) # (num_channels, num_time_steps, num_x, num_y)
        
        # Preprocess data
        if self.preprocessor is not None:
            if self.preprocessor.static_point:
                static_point_parameters = self.preprocessor.static_point.transform(static_point_parameters)
            if self.preprocessor.static_spatial:
                static_spatial_parameters = self.preprocessor.static_spatial.transform(static_spatial_parameters)
            if self.preprocessor.dynamic_point:
                dynamic_point_parameters = self.preprocessor.dynamic_point.transform(dynamic_point_parameters)
            if self.preprocessor.dynamic_spatial:
                dynamic_spatial_parameters = self.preprocessor.dynamic_spatial.transform(dynamic_spatial_parameters)
            if self.preprocessor.output:              
                output_variables = self.preprocessor.output.transform(output_variables)

        # Collect output in dictionary
        return_dict = {}

        for var_type in self.parameter_vars.keys():
            if self.parameter_vars[var_type] is not None:
                return_dict[f'{var_type}_parameters'] = eval(f'{var_type}_parameters')
        
        if output_variables is not None:
            return_dict['output_variables'] = output_variables
            
        return return_dict


class ParameterDataset(Dataset):
    def __init__(
        self, 
        preprocessor: Preprocessor =  None,
        path: str = None, 
        ):

        size = '64x64x1'

        if path is None:
            self.path = f'data/parameters/2Dresampled_{size}_realization_'
        else:
            self.path = path + f'/{size}_'

        self.preprocessor = preprocessor


    def __len__(self):
        return 5934
    
    def __getitem__(self, idx):

        data = xr.open_dataset(f'{self.path}{idx+1}.nc')

        static_spatial_parameters = get_static_spatial_parameters(
            data=data, 
            static_spatial_vars=['Porosity', 'Permeability']
        )
        static_spatial_parameters = static_spatial_parameters.squeeze(1)

        if self.preprocessor is not None:
            static_spatial_parameters = self.preprocessor.static_spatial.transform(static_spatial_parameters)

        return {'static_spatial_parameters': static_spatial_parameters}


'''
class XarrayDataset_old(Dataset):  #deprecated
    def __init__(
        self, 
        folder, 
        input_vars, 
        output_vars,
        dynamic_input_vars=None,
        include_spatial_coords=True,
        include_time=True,
        input_preprocessor_load_path=None,
        output_preprocessor_load_path=None,
        dynamic_input_preprocessor_load_path=None,
        ):
        
        self.folder = folder
        self.file_list = os.listdir(folder)
        self.input_vars = input_vars
        if include_spatial_coords:
            self.input_vars.append('x_encoding')
            self.input_vars.append('y_encoding')
        if include_time:
            self.input_vars.append('time_encoding')
        self.output_vars = output_vars

        self.dynamic_input_vars = dynamic_input_vars

        # Load preprocessors
        if input_preprocessor_load_path is not None:
            with open(input_preprocessor_load_path, 'rb') as f:
                self.input_preprocessor = pickle.load(f)
        else:
            self.input_preprocessor = None
        
        if dynamic_input_preprocessor_load_path is not None:
            with open(dynamic_input_preprocessor_load_path, 'rb') as f:
                self.dynamic_input_preprocessor = pickle.load(f)
        else:
            self.dynamic_input_preprocessor = None

        if output_preprocessor_load_path is not None:
            with open(output_preprocessor_load_path, 'rb') as f:
                self.output_preprocessor = pickle.load(f)
        else:
            self.output_preprocessor = None

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        
        file_path = os.path.join(self.folder, self.file_list[idx])
        data = xr.open_dataset(file_path)
        # Add the meshgrids as new data variables in the xarray
        X = data['X'].values
        Y = data['Y'].values
        TIME = data['time'].values

        # Create meshgrids for X and Y dimensions
        x_mesh, y_mesh = np.meshgrid(data.X, data.Y, indexing='ij')
        TIME_MESH = np.meshgrid(data.time, data.X, data.Y, indexing='ij')
        data = data.assign(x_encoding=xr.DataArray(x_mesh, coords=[("X", X), ("Y", Y)]))
        data = data.assign(y_encoding=xr.DataArray(y_mesh, coords=[("X", X), ("Y", Y)]))
        data = data.assign(time_encoding=xr.DataArray(TIME_MESH[0], coords=[("time", TIME), ("X", X), ("Y", Y)]))

        # Concatenate input variables along a new dimension
        input_data_list = []
        for var in self.input_vars:
            if 'time' in data[var].dims:
                input_data_list.append(torch.tensor(data[var].values, dtype=torch.float32).unsqueeze(0))
            else:
                scalar_matrix = torch.tensor(data[var].values, dtype=torch.float32).expand(data.time.size, -1, -1)
                input_data_list.append(scalar_matrix.unsqueeze(0))

        input_data = torch.cat(input_data_list, dim=0)
        if self.input_preprocessor is not None:
            for i in range(input_data.shape[1]):
                input_data[:, i] = self.input_preprocessor.transform(input_data[:, i])
        input_data = input_data.permute(1, 0, 2, 3)  

        # Concatenate dynamic input variables along a new dimension

        if self.dynamic_input_vars is not None:
            dynamic_input_data_list = []
            for var in self.dynamic_input_vars:
                if 'time' in data[var].dims:
                    dynamic_input_data_list.append(torch.tensor(data[var].values, dtype=torch.float32).unsqueeze(0))
                else:
                    scalar_matrix = torch.tensor(data[var].values, dtype=torch.float32).expand(data.time.size, -1, -1)
                    dynamic_input_data_list.append(scalar_matrix.unsqueeze(0))

            dynamic_input_data = torch.cat(dynamic_input_data_list, dim=0)

            if self.dynamic_input_preprocessor is not None:
                for i in range(dynamic_input_data.shape[1]):
                    dynamic_input_data[:, i] = self.dynamic_input_preprocessor.transform(dynamic_input_data[:, i])
            dynamic_input_data = dynamic_input_data.squeeze(1)
            dynamic_input_data = torch.tile(dynamic_input_data, (dynamic_input_data.shape[-1], 1, 1))
            
        # Concatenate output variables along a new dimension
        output_data_list = [torch.tensor(data[var].values, dtype=torch.float32).unsqueeze(0) for var in self.output_vars]
        output_data = torch.cat(output_data_list, dim=0)

        if self.output_preprocessor is not None:
            for i in range(output_data.shape[1]):
                output_data[:, i] = self.output_preprocessor.transform(output_data[:, i])
        output_data = output_data.permute(1, 0, 2, 3)

        if self.dynamic_input_vars is not None:
            return input_data, dynamic_input_data, output_data
        else:
            return input_data, None, output_data



'''