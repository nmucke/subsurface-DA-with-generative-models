import os
import pdb
import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import pickle
import matplotlib.pyplot as plt


class XarrayDataset(Dataset):  #deprecated
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



