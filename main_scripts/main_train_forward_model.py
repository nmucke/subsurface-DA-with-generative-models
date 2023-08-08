import os
import pickle
import xarray as xr
import numpy as np
import torch
import yaml
import pdb
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from subsurface_DA_with_generative_models.models.forward_models.FNO3D import FNO3d

from subsurface_DA_with_generative_models.optimizers.forward_model_optimizer import ForwardModelOptimizer
from subsurface_DA_with_generative_models.train_steppers.forward_FNO3D_train_stepper import FNO3DTrainStepper
from subsurface_DA_with_generative_models.train_steppers.forward_train_stepper import ForwardTrainStepper

# use matplotlib agg backend so figures can be saved in the background
plt.switch_backend('Qt5Agg')

from subsurface_DA_with_generative_models.models.forward_models import u_net_GAN
from subsurface_DA_with_generative_models.models.forward_models import u_net
from subsurface_DA_with_generative_models.optimizers.GAN_optimizer import GANOptimizer
from subsurface_DA_with_generative_models.train_steppers.forward_GAN_train_stepper import ForwardGANTrainStepper
from subsurface_DA_with_generative_models.data_handling.xarray_data import XarrayDataset
from subsurface_DA_with_generative_models.trainers.train_forward_model import train_forward_model

torch.set_default_dtype(torch.float32)

torch.backends.cuda.enable_flash_sdp(enabled=True)
torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True

MODEL_TYPE = 'UNet'
DEVICE = 'cuda'
SAVE_PATH = f'trained_models/{MODEL_TYPE}'

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

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

if MODEL_TYPE == 'UNetGAN' or MODEL_TYPE == 'UNet':
    PREPROCESSOR_LOAD_PATH = 'trained_preprocessors/preprocessor_64.pkl'
elif MODEL_TYPE == 'FNO3D':
    PREPROCESSOR_LOAD_PATH = 'trained_preprocessors/preprocessor_64_FNO.pkl'
    STATIC_SPATIAL_VARS += ['x_encoding', 'y_encoding']

config_path = f"configs/{MODEL_TYPE}.yml"
with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

# Save config to save path
with open(f'{SAVE_PATH}/config.yml', 'w') as f:
    yaml.dump(config, f)
    
def main():
    
    # Load up preprocessor
    with open(PREPROCESSOR_LOAD_PATH, 'rb') as f:
        preprocessor = pickle.load(f)
    
    # Load data
    dataset = XarrayDataset(
        data_folder=FOLDER,
        parameter_vars=parameter_vars,
        output_vars=OUTPUT_VARS,
        preprocessor=preprocessor,
        num_samples=500,
    )
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

    train_dataloader = DataLoader(
        train_dataset,
        **config['dataloader_args'],
    )
    val_dataloader = DataLoader(
        val_dataset,
        **config['dataloader_args'],
    )

    # Set up model
    if MODEL_TYPE == 'UNetGAN':
        model = u_net_GAN.UNetGAN(
            generator_args=config['model_args']['generator_args'],
            critic_args=config['model_args']['critic_args'],
        )
    elif MODEL_TYPE == 'FNO3D':
        model = FNO3d(
            **config['model_args'],
        )
    elif MODEL_TYPE == 'UNet':
        model = u_net.UNet(
            **config['model_args'],
        )
    model.to(DEVICE)
    
    # Set up optimizer
    if MODEL_TYPE == 'UNet':
        optimizer = ForwardModelOptimizer(
            model=model,
            args=config['optimizer_args']
        )
    elif MODEL_TYPE == 'UNetGAN':
        optimizer = GANOptimizer(
            model=model,
            args=config['optimizer_args']
        )
    elif MODEL_TYPE == 'FNO3D':
        optimizer = ForwardModelOptimizer(
            model=model,
            args=config['optimizer_args']
        )

    # Load model and optimizer weights if continuing training
    if CONTINUE_TRAINING:
        state_dict = torch.load(f'{SAVE_PATH}/model.pt', map_location=DEVICE)
        model.load_state_dict(state_dict['model_state_dict'])
        optimizer.load_state_dict(state_dict)
    
    if MODEL_TYPE == 'UNetGAN':
        train_stepper = ForwardGANTrainStepper(
            model=model,
            optimizer=optimizer,
            model_save_path=SAVE_PATH,
            **config['train_stepper_args'],
        )
    elif MODEL_TYPE == 'FNO3D':
        train_stepper = FNO3DTrainStepper(
            model=model,
            optimizer=optimizer,
            device=DEVICE,
            model_save_path=SAVE_PATH,
            #**config['train_stepper_args'],
        )
    elif MODEL_TYPE == 'UNet':
        train_stepper = ForwardTrainStepper(
            model=model,
            optimizer=optimizer,
            model_save_path=SAVE_PATH,
            #**config['train_stepper_args'],
        )

    # Set up trainer
    train_forward_model(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        train_stepper=train_stepper,
        plot_path='gan_output',
        **config['trainer_args'],
    )

if __name__ == "__main__":
    main()