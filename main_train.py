import os
import xarray as xr
import numpy as np
import torch
import yaml
import pdb
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from subsurface_DA_with_generative_models import routine 
from subsurface_DA_with_generative_models.models import GAN
from subsurface_DA_with_generative_models.optimizers.GAN_optimizer import GANOptimizer
from subsurface_DA_with_generative_models.train_steppers.GAN_train_stepper import GANTrainStepper
from subsurface_DA_with_generative_models.trainers.train_GAN import train_GAN
from subsurface_DA_with_generative_models.data_handling.xarray_data import XarrayDataset


MODEL_TYPE = 'GAN'
DEVICE = 'cpu'
SAVE_PATH = 'trained_models/GAN.pt'

FOLDER = "data/results32"
INPUT_VARS = ['Por', 'Perm', 'Pressure'] # Porosity, Permeability, Pressure + x, y, time encodings 
OUTPUT_VARS = ['Pressure']

config_path = f"configs/{MODEL_TYPE}.yml"
with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)


def main():
    # Load data
    train_dataset = XarrayDataset(
        folder=FOLDER,
        input_vars=INPUT_VARS,
        output_vars=OUTPUT_VARS
    )
    train_dataloader = DataLoader(
        train_dataset,
        **config['dataloader_args'],
    )

    '''
    # Test the dataloader by loading a few batches
    num_batches_to_check = 1
    for i, (input_data, output_data) in enumerate(train_dataloader):
        if i >= num_batches_to_check:
            break

        print(f"Batch {i+1}:")
        print(f"Input data shape: {input_data.shape}")
        print(f"Output data shape: {output_data.shape}")
    ''' 

    # Set up model
    model = GAN.GAN(
        generator_args=config['model_args']['generator_args'],
        critic_args=config['model_args']['critic_args'],
    )
    model.to(DEVICE)
    
    # Set up optimizer
    optimizer = GANOptimizer(
        model=model,
        args=config['optimizer_args']
    )
    
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
        model_save_path=SAVE_PATH
    )



if __name__ == "__main__":
    main()