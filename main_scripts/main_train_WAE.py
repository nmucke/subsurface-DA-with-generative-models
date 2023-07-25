import os
import xarray as xr
import numpy as np
import torch
import yaml
import pdb
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from subsurface_DA_with_generative_models import routine 
from subsurface_DA_with_generative_models.models.parameter_models import  WAE
from subsurface_DA_with_generative_models.optimizers.WAE_optimizer import WAEOptimizer
from subsurface_DA_with_generative_models.train_steppers.WAE_train_stepper import WAETrainStepper
from subsurface_DA_with_generative_models.data_handling.xarray_data import XarrayDataset
from subsurface_DA_with_generative_models.trainers.train_WAE import train_WAE

torch.set_default_dtype(torch.float32)

torch.backends.cuda.enable_flash_sdp(enabled=True)
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True

MODEL_TYPE = 'WAE'
DEVICE = 'cuda'
SAVE_PATH = 'trained_models/WAE.pt'

CONTINUE_TRAINING = True

FOLDER = "data/results64"
INPUT_VARS = ['Por', 'Perm'] # Porosity, Permeability, Pressure + x, y, time encodings 
DYNAMIC_INPUT_VARS = ['gas_rate']
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
        include_time=False,
        input_preprocessor_load_path=INPUT_PREPROCESSOR_LOAD_PATH,
        output_preprocessor_load_path=OUTPUT_PREPROCESSOR_LOAD_PATH,
        dynamic_input_preprocessor_load_path=DYNAMIC_INPUT_PREPROCESSOR_LOAD_PATH,
    )
    train_dataloader = DataLoader(
        train_dataset,
        **config['dataloader_args'],
    )

    # Set up model
    model = WAE.WassersteinAutoencoder(
        decoder_args=config['model_args']['decoder_args'],
        encoder_args=config['model_args']['encoder_args'],
    )
    model.to(DEVICE)
    
    if CONTINUE_TRAINING:
        state_dict = torch.load('trained_models/WAE.pt', map_location=DEVICE)
        model.load_state_dict(state_dict['model_state_dict'])
    
    # Set up optimizer
    optimizer = WAEOptimizer(
        model=model,
        args=config['optimizer_args']
    )

    if CONTINUE_TRAINING:
        optimizer.decoder.load_state_dict(state_dict['decoder_optimizer_state_dict'])
        optimizer.encoder.load_state_dict(state_dict['encoder_optimizer_state_dict'])
        optimizer.decoder_scheduler.load_state_dict(state_dict['decoder_scheduler_state_dict'])
        optimizer.encoder_scheduler.load_state_dict(state_dict['encoder_scheduler_state_dict'])
    

    # Set up train stepper
    train_stepper = WAETrainStepper(
        model=model,
        optimizer=optimizer,
        **config['train_stepper_args'],
    )

    # Set up trainer
    train_WAE(
        train_dataloader=train_dataloader,
        train_stepper=train_stepper,
        **config['trainer_args'],
        model_save_path=SAVE_PATH,
        save_output=True,
        only_input=True,
    )

if __name__ == "__main__":
    main()