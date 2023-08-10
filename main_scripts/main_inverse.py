import os
import pickle
import xarray as xr
import numpy as np
import torch
import yaml
import pdb
import ray
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from subsurface_DA_with_generative_models.data_handling.data_utils import prepare_batch
from subsurface_DA_with_generative_models.inversion.ensemble_kalman_inversion import EnsembleKalmanInversion
from subsurface_DA_with_generative_models.model_testing import ForwardModelTester
from subsurface_DA_with_generative_models.models.forward_models.FNO3D import FNO3d
from subsurface_DA_with_generative_models.models.latent_to_output_model import LatentToOutputModel
from subsurface_DA_with_generative_models.models.parameter_models.parameter_diffusion import ParameterDiffusion


# use matplotlib agg backend so figures can be saved in the background
plt.switch_backend('Qt5Agg')

from subsurface_DA_with_generative_models.models.forward_models import u_net_GAN
from subsurface_DA_with_generative_models.models.forward_models import u_net
from subsurface_DA_with_generative_models.data_handling.xarray_data import XarrayDataset

torch.set_default_dtype(torch.float32)

torch.backends.cuda.enable_flash_sdp(enabled=True)
torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True

MODELS_TO_TEST = ['UNet']

WITH_PARAMETER_MODEL = True

DEVICE = 'cuda'

INVERSE_SOLVER = 'EKI'

FOLDER = "data/results64"
STATIC_POINT_VARS = None
STATIC_SPATIAL_VARS = ['Por', 'Perm']
DYNAMIC_SPATIAL_VARS = ['time_encoding']
DYNAMIC_POINT_VARS = ['gas_rate']
OUTPUT_VARS = ['Pressure', 'CO_2']

parameter_vars = {
    'static_point': STATIC_POINT_VARS,
    'static_spatial': STATIC_SPATIAL_VARS,
    'dynamic_point': DYNAMIC_POINT_VARS,
    'dynamic_spatial': DYNAMIC_SPATIAL_VARS,
}

NUM_PARTICLES = 5000
NUM_ITERATIONS = 50

OBS_X = [20, 20, 40, 40]
OBS_Y = [20, 40, 20, 40]
#OBS_X = [10, 10, 20, 20, 40, 40, 50, 50]
#OBS_Y = [10, 20, 40, 20, 40, 50, 10, 50]

'''
OBS_X = range(0, 64, 10)
OBS_Y = range(0, 64, 10)

OBS_X, OBS_Y = np.meshgrid(OBS_X, OBS_Y)
OBS_X = OBS_X.flatten()
OBS_Y = OBS_Y.flatten()
'''

def observation_operator(x):
    return x[0, :, OBS_X, OBS_Y].flatten()

def main():
    
    for model_type in MODELS_TO_TEST:
        load_path = f'trained_models/{model_type}'

        if model_type == 'UNetGAN' or model_type == 'UNet':
            preprocessor_load_path = 'trained_preprocessors/preprocessor_64.pkl'
        elif model_type == 'FNO3D':
            preprocessor_load_path = 'trained_preprocessors/preprocessor_64_FNO.pkl'
            STATIC_SPATIAL_VARS += ['x_encoding', 'y_encoding']


        # Load up preprocessor
        with open(preprocessor_load_path, 'rb') as f:
            preprocessor = pickle.load(f)

        
        # Load data
        dataset = XarrayDataset(
            data_folder=FOLDER,
            parameter_vars=parameter_vars,
            output_vars=OUTPUT_VARS,
            preprocessor=preprocessor,
            num_samples=200,
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        config_path = f"configs/{model_type}.yml"
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

        # Set up model
        if model_type == 'UNetGAN':
            model = u_net_GAN.UNetGAN(
                generator_args=config['model_args']['generator_args'],
                critic_args=config['model_args']['critic_args'],
            )
        elif model_type == 'FNO3D':
            model = FNO3d(
                **config['model_args'],
            )
        elif model_type == 'UNet':
            model = u_net.UNet(
                **config['model_args'],
            )
        model.to(DEVICE)

        model.eval()
        
        # Load tained model weights
        state_dict = torch.load(f'{load_path}/model.pt', map_location=DEVICE)
        model.load_state_dict(state_dict['model_state_dict'])


        if WITH_PARAMETER_MODEL:
            parameter_model = ParameterDiffusion(
                device=DEVICE,
            )
            parameter_model.load_state_dict(torch.load('trained_models/diffusion.pt'))
            parameter_model.to(DEVICE)


            model = LatentToOutputModel(
                parameter_model=parameter_model.latent_to_sample,
                forward_model=model,
            )

        # Get batch
        batch = dataloader.dataset[1]

        # unpack batch
        static_point_parameters, static_spatial_parameters, \
        dynamic_point_parameters, dynamic_spatial_parameters, \
        output_variables = prepare_batch(
            batch=batch,
            device='cpu',
        )

        #output_variables = preprocessor.output.inverse_transform(output_variables)

        true_static_spatial_parameters = static_spatial_parameters

        fixed_input = {
            'static_point_parameters': static_point_parameters,
            'dynamic_point_parameters': dynamic_point_parameters,
            'dynamic_spatial_parameters': dynamic_spatial_parameters,
        }

        observations = observation_operator(output_variables) 
        observations_with_noise = observations + torch.normal(
            mean=torch.zeros_like(observations), 
            std=0.005*torch.ones_like(observations)
            )

        if INVERSE_SOLVER == 'EKI':
            inverse_solver = EnsembleKalmanInversion(
                forward_model=model,
                fixed_input=fixed_input,
                #preprocessor=preprocessor,
                num_particles=NUM_PARTICLES,
                num_iterations=NUM_ITERATIONS,
                parameter_dim=(2, 64, 64),
                device=DEVICE,
            )

        parameter_ensemble, output_ensemble = inverse_solver.solve(
            observation_operator=observation_operator,
            observations=observations_with_noise,
        )
        parameter_ensemble = parameter_ensemble.reshape((NUM_PARTICLES, 2, 64, 64))

        '''
        if WITH_PARAMETER_MODEL:

            parameter_model.eval()

            batch_size = 25
            parameter_ensemble = torch.zeros((NUM_PARTICLES, 2, 64, 64))
            for i in range(0, NUM_PARTICLES, batch_size):
                sample = parameter_model.latent_to_sample(_parameter_ensemble[i:i+batch_size].to(DEVICE))
                parameter_ensemble[i:i+batch_size] = sample.detach().cpu()
        '''
        parameter_ensemble = parameter_ensemble.numpy()
        parameter_ensemble = np.mean(parameter_ensemble, axis=0)

        output_observations = output_ensemble[:, 0, :, OBS_X, OBS_Y]
        output_observations = output_observations.numpy()
        output_obs_mean = np.mean(output_observations, axis=0)
        output_obs_std = np.std(output_observations, axis=0)

        true_obs = output_variables[0, :, OBS_X, OBS_Y].numpy() 
        true_obs = observations_with_noise.view(true_obs.shape).numpy()

        plt.figure(figsize=(20, 10))

        plt.subplot(1, 3, 1)
        plt.imshow(true_static_spatial_parameters[0])
        plt.scatter(OBS_X, OBS_Y, color='r')
        plt.title('True')
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.imshow(parameter_ensemble[0])
        plt.title('Inversion')
        plt.colorbar()

        plt.subplot(1, 3, 3)
        plt.plot(true_obs, 'k', label='True', linewidth=3)
        plt.plot(output_obs_mean, 'tab:blue', label='Inversion')

        for i in range(len(OBS_X)):
            plt.fill_between(
                np.arange(output_obs_mean.shape[0]),
                output_obs_mean[:,i] - output_obs_std[:,i],
                output_obs_mean[:,i] + output_obs_std[:,i],
                color='tab:blue',
                alpha=0.25,
            )
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == '__main__':
    #ray.init(num_cpus=25)
    main()
    #ray.shutdown()

        