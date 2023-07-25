import os
import pickle
import xarray as xr
import numpy as np
import torch
import yaml
import pdb
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from subsurface_DA_with_generative_models.models.parameter_models.WAE import WassersteinAutoencoder

from subsurface_DA_with_generative_models.models.parameter_models.parameter_GAN import ParameterGAN
from subsurface_DA_with_generative_models.optimizers.GAN_optimizer import GANOptimizer
from subsurface_DA_with_generative_models.optimizers.WAE_optimizer import WAEOptimizer
from subsurface_DA_with_generative_models.train_steppers.old_train_steppers.GAN_train_stepper import GANTrainStepper
from subsurface_DA_with_generative_models.train_steppers.old_train_steppers.WAE_train_stepper import WAETrainStepper
from subsurface_DA_with_generative_models.train_steppers.parameter_GAN_train_stepper import ParameterGANTrainStepper
from subsurface_DA_with_generative_models.trainers.train_parameter_generator import train_parameter_generator
from subsurface_DA_with_generative_models.data_handling.xarray_data import ParameterDataset, XarrayDataset


torch.set_default_dtype(torch.float32)

torch.backends.cuda.enable_flash_sdp(enabled=True)
torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True


MODEL_TYPE = 'parameter_GAN'
DEVICE = 'cuda'
SAVE_PATH = 'trained_models/parameter_GAN'

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)


CONTINUE_TRAINING = True

PREPROCESSOR_LOAD_PATH = 'trained_preprocessors/preprocessor_64.pkl'

FOLDER = "data/parameters_processed"
STATIC_SPATIAL_VARS = ['Por', 'Perm']


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
    dataset = ParameterDataset(
        preprocessor=preprocessor,
        path = 'data/parameters_processed',
        )

    train_dataloader = DataLoader(
        dataset,
        **config['dataloader_args'],
    )

    # Set up model
    if MODEL_TYPE == 'parameter_GAN':
        model = ParameterGAN(
            generator_args=config['model_args']['generator_args'],
            critic_args=config['model_args']['critic_args'],
        )
        # Set up optimizer
        optimizer = GANOptimizer(
            model=model,
            args=config['optimizer_args']
        )
    elif MODEL_TYPE == 'parameter_WAE':
        model = WassersteinAutoencoder(
            encoder_args=config['model_args']['encoder_args'],
            decoder_args=config['model_args']['decoder_args'],
        )
        # Set up optimizer
        optimizer = WAEOptimizer(
            model=model,
            args=config['optimizer_args']
        )


    # Load model and optimizer weights if continuing training
    if CONTINUE_TRAINING:
        state_dict = torch.load(f'{SAVE_PATH}/model.pt', map_location=DEVICE)
        model.load_state_dict(state_dict['model_state_dict'])
        optimizer.load_state_dict(state_dict)
    
    # Set up train stepper
    if MODEL_TYPE == 'parameter_GAN':
        train_stepper = ParameterGANTrainStepper(
            model=model,
            optimizer=optimizer,
            model_save_path=SAVE_PATH,
            **config['train_stepper_args'],
        )
        
    # Set up train stepper
    train_stepper = ParameterGANTrainStepper(
        model=model,
        optimizer=optimizer,
        model_save_path=SAVE_PATH,
        **config['train_stepper_args'],
    )

    # Set up trainer
    train_parameter_generator(
        train_dataloader=train_dataloader,
        val_dataloader=None,
        train_stepper=train_stepper,
        plot_path='gan_output',
        **config['trainer_args'],
    )

if __name__ == "__main__":
    main()