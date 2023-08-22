import os
import xarray as xr
import numpy as np
import torch
import yaml
import pdb
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from subsurface_DA_with_generative_models import routine 
from subsurface_DA_with_generative_models.models.forward_models import u_net_GAN
from subsurface_DA_with_generative_models.optimizers.GAN_optimizer import GANOptimizer
from subsurface_DA_with_generative_models.train_steppers.old_train_steppers.GAN_train_stepper import GANTrainStepper
from subsurface_DA_with_generative_models.trainers.train_GAN import train_GAN
from subsurface_DA_with_generative_models.data_handling.xarray_data import XarrayDataset

torch.set_default_dtype(torch.float32)

torch.backends.cuda.enable_flash_sdp(enabled=True)
torch.set_float32_matmul_precision('medium')s
torch.backends.cuda.matmul.allow_tf32 = True


MODEL_TYPE = 'GAN'
DEVICE = 'cuda'
SAVE_PATH = 'trained_models/GAN.pt'

CONTINUE_TRAINING = True

FOLDER = "data/results64"
INPUT_VARS = ['Por', 'Perm'] # Porosity, Permeability, Pressure + x, y, time encodings 
DYNAMIC_INPUT_VARS = ['gas_rate',]
OUTPUT_VARS =  ['Pressure', 'CO_2']

INPUT_PREPROCESSOR_LOAD_PATH = 'trained_preprocessors/input_preprocessor_64.pkl'
DYNAMIC_INPUT_PREPROCESSOR_LOAD_PATH = 'trained_preprocessors/dynamic_input_preprocessor_64.pkl'
OUTPUT_PREPROCESSOR_LOAD_PATH = 'trained_preprocessors/output_preprocessor_64.pkl'

config_path = f"configs/{MODEL_TYPE}.yml"
with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

def main():
    
    # Load data
    train_dataset = XarrayDataset(
        folder=FOLDER,
        input_vars=INPUT_VARS,
        output_vars=OUTPUT_VARS,
        dynamic_input_vars=DYNAMIC_INPUT_VARS,
        include_spatial_coords=False,
        include_time=True,
        input_preprocessor_load_path=INPUT_PREPROCESSOR_LOAD_PATH,
        output_preprocessor_load_path=OUTPUT_PREPROCESSOR_LOAD_PATH,
        dynamic_input_preprocessor_load_path=DYNAMIC_INPUT_PREPROCESSOR_LOAD_PATH,
    )
    train_dataloader = DataLoader(
        train_dataset,
        **config['dataloader_args'],
    )
   

    # Set up model
    model = u_net_GAN.GAN(
        generator_args=config['model_args']['generator_args'],
        critic_args=config['model_args']['critic_args'],
    )

    if CONTINUE_TRAINING:
        state_dict = torch.load('trained_models/GAN.pt', map_location=DEVICE)
        model.load_state_dict(state_dict['model_state_dict'])
    
    model.to(DEVICE)
    
    # Set up optimizer
    optimizer = GANOptimizer(
        model=model,
        args=config['optimizer_args']
    )
    
    if CONTINUE_TRAINING:
        optimizer.generator.load_state_dict(state_dict['generator_optimizer_state_dict'])
        optimizer.critic.load_state_dict(state_dict['critic_optimizer_state_dict'])
        optimizer.generator_scheduler.load_state_dict(state_dict['generator_scheduler_state_dict'])
        optimizer.critic_scheduler.load_state_dict(state_dict['critic_scheduler_state_dict'])
    
    # Set up train stepper
    train_stepper = GANTrainStepper(
        model=model,
        optimizer=optimizer,
        **config['train_stepper_args'],
    )

    # Set up trainer
    train_GAN(
        train_dataloader=train_dataloader,
        train_stepper=train_stepper,
        **config['trainer_args'],
        model_save_path=SAVE_PATH,
        save_output=True,
    )

if __name__ == "__main__":
    main()