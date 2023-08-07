
import pdb
import numpy as np
import torch
import xarray as xr

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

def get_variable(data, var_name):
    if var_name in data:
        return torch.tensor(data[var_name].values, dtype=torch.get_default_dtype())
    else:
        return None
    
def get_static_point_parameters(data, static_point_vars):
    if static_point_vars is None:
        return None
    
    static_point_parameters = []
    for var_name in static_point_vars:
        variable_data = get_variable(data, var_name)
        static_point_parameters.append(variable_data)
    
    static_point_parameters = torch.stack(static_point_parameters, dim=0)

    return static_point_parameters

def get_static_spatial_parameters(data, static_spatial_vars):
    if static_spatial_vars is None:
        return None

    static_spatial_parameters = []
    for var_name in static_spatial_vars:
        variable_data = get_variable(data, var_name)
        static_spatial_parameters.append(variable_data)
    
    static_spatial_parameters = torch.stack(static_spatial_parameters, dim=0)

    return static_spatial_parameters


def get_dynamic_point_parameters(data, dynamic_point_vars):
    if dynamic_point_vars is None:
        return None
    
    dynamic_point_parameters = []
    for var_name in dynamic_point_vars:
        variable_data = get_variable(data, var_name)
        variable_data = variable_data.squeeze(0)
        dynamic_point_parameters.append(variable_data)
    
    dynamic_point_parameters = torch.stack(dynamic_point_parameters, dim=0)

    return dynamic_point_parameters

def get_dynamic_spatial_parameters(data, dynamic_spatial_vars):
    if dynamic_spatial_vars is None:
        return None

    dynamic_spatial_parameters = []
    for var_name in dynamic_spatial_vars:
        variable_data = get_variable(data, var_name)
        dynamic_spatial_parameters.append(variable_data)
    
    dynamic_spatial_parameters = torch.stack(dynamic_spatial_parameters, dim=0)

    return dynamic_spatial_parameters

def get_output_variables(data, output_vars):
    if output_vars is None:
        return None

    output_variables = []
    for var_name in output_vars:
        variable_data = get_variable(data, var_name)
        output_variables.append(variable_data)
    
    output_variables = torch.stack(output_variables, dim=0)

    return output_variables


    
def reshape_spatial_variable(variable_data):
    if len(variable_data.shape) == 2:
        return variable_data.unsqueeze(0)
    else:
        return variable_data

def reshape_time_variable(variable_data):
    if len(variable_data.shape) == 2:
        return variable_data.unsqueeze(1)
    else:
        return variable_data
    
def get_meshgrids(data):
    X = data['X'].values
    Y = data['Y'].values
    TIME = data['time'].values

    # Create meshgrids for X and Y dimensions
    x_mesh, y_mesh = np.meshgrid(data.X, data.Y, indexing='ij')
    time_mesh = np.meshgrid(data.time, data.X, data.Y, indexing='ij')

    return x_mesh, y_mesh, time_mesh
    
def add_meshgrids_to_data(data, x=True, y=True, time=True):

    X = data['X'].values
    Y = data['Y'].values
    TIME = data['time'].values

    x_mesh, y_mesh, time_mesh = get_meshgrids(data)

    if x:
        data = data.assign(x_encoding=xr.DataArray(x_mesh, coords=[("X", X), ("Y", Y)]))
    if y:
        data = data.assign(y_encoding=xr.DataArray(y_mesh, coords=[("X", X), ("Y", Y)]))
    if time:
        data = data.assign(time_encoding=xr.DataArray(time_mesh[0], coords=[("time", TIME), ("X", X), ("Y", Y)]))

    return data