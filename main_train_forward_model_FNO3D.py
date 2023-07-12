#%%
import os
import pickle
import xarray as xr
import numpy as np
import torch
import yaml
import pdb
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
%matplotlib inline

# use matplotlib agg backend so figures can be saved in the background
#plt.switch_backend('Agg')

from subsurface_DA_with_generative_models import routine 
from subsurface_DA_with_generative_models.models.forward_models import FNO3D
from subsurface_DA_with_generative_models.optimizers.FNO3D_optimizer import FNO3dOptimizer
from subsurface_DA_with_generative_models.preprocessor import Preprocessor
from subsurface_DA_with_generative_models.train_steppers.forward_FNO3D_train_stepper import FNO3DTrainStepper
from subsurface_DA_with_generative_models.trainers.train_FNO3D_forward_model import train_forward_model
from subsurface_DA_with_generative_models.data_handling.xarray_data import XarrayDataset


torch.set_default_dtype(torch.float32)
torch.backends.cuda.enable_flash_sdp(enabled=True)
torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True

#%%

MODEL_TYPE = 'FNO3D'
DEVICE = 'cuda'
SAVE_PATH = 'trained_models/FNO3D'

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

CONTINUE_TRAINING = False

PREPROCESSOR_LOAD_PATH = '/samoa/data/smrserraoseabr/subsurface-DA-with-generative-models/trained_preprocessors/preprocessor_32x32.pkl'

FOLDER = "/samoa/data/smrserraoseabr/NO-DA/dataset/mixedcontext32x32"
STATIC_POINT_VARS = None
STATIC_SPATIAL_VARS = ['Por', 'Perm']
DYNAMIC_SPATIAL_VARS = ['time_encoding']
DYNAMIC_POINT_VARS = ['gas_rate']
OUTPUT_VARS = ['Pressure']



parameter_vars = {
    'static_point': STATIC_POINT_VARS,
    'static_spatial': STATIC_SPATIAL_VARS,
    'dynamic_point': DYNAMIC_POINT_VARS,
    'dynamic_spatial': DYNAMIC_SPATIAL_VARS,
}



config_path = f"configs/{MODEL_TYPE}.yml"
with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)


# Save config to save path
with open(f'{SAVE_PATH}/config.yml', 'w') as f:
    yaml.dump(config, f)
    
 
    # Load up preprocessor
    with open(PREPROCESSOR_LOAD_PATH, 'rb') as f:
        preprocessor = pickle.load(f)
    
# Load data
dataset = XarrayDataset(
    data_folder=FOLDER,
    parameter_vars=parameter_vars,
    output_vars=OUTPUT_VARS,
    preprocessor=preprocessor,

)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])

#%%
train_dataloader = DataLoader(
    train_dataset,
    **config['dataloader_args'],
)

val_dataloader = DataLoader(
    val_dataset,
    **config['dataloader_args'],
)

# Set up model
model = FNO3D.FNO3d(
    **config['model_args']['fno3d_args'],
)

model.to(DEVICE)

# Set up optimizer
optimizer = FNO3dOptimizer(
    model=model,
    args=config['optimizer_args']
)
#%%

# Load model and optimizer weights if continuing training
if CONTINUE_TRAINING:
    state_dict = torch.load(f'{SAVE_PATH}.pt', map_location=DEVICE)
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])

# Set up train stepper
if MODEL_TYPE == 'FNO3D':
    train_stepper = FNO3DTrainStepper(
        model=model,
        optimizer=optimizer,
        device=DEVICE,
        ##model_save_path=SAVE_PATH,
        **config['train_stepper_args'],
    )
#%%
# Set up trainer
train_forward_model(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    train_stepper=train_stepper,
    plot_path='fno3d_output',
    **config['trainer_args'],

)

# %%
