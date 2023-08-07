import os
import pickle
import xarray as xr
import numpy as np
import torch
import yaml
import pdb
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from subsurface_DA_with_generative_models.model_testing import ForwardModelTester
from subsurface_DA_with_generative_models.models.forward_models.FNO3D import FNO3d


# use matplotlib agg backend so figures can be saved in the background
plt.switch_backend('Qt5Agg')

from subsurface_DA_with_generative_models.models.forward_models import u_net_GAN
from subsurface_DA_with_generative_models.models.forward_models import u_net
from subsurface_DA_with_generative_models.data_handling.xarray_data import XarrayDataset

torch.set_default_dtype(torch.float32)

torch.backends.cuda.enable_flash_sdp(enabled=True)
torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True

MODELS_TO_TEST = ['UNet', 'UNetGAN']
DEVICE = 'cuda'

CONTINUE_TRAINING = False

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


        model_tester = ForwardModelTester(
            forward_model=model,
            preprocessor=preprocessor,
            device=DEVICE,
            metrics=['RMSE', 'RRMSE'],
        )

        # Compute metrics
        metrics = model_tester.test_model(
            dataloader=dataloader,
            print_metrics=True,
            return_metrics=True,
        )

        # Plot
        model_tester.plot(
            dataloader=dataloader,
            save_path=None, 
            plot_time=40,
            plot_x_y=(20, 20),
            sample_index=0,
            show_plot=True,
        )


if __name__ == "__main__":
    main()