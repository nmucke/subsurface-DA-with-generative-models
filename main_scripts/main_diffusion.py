import pdb
import pickle
import numpy as np
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from tqdm import tqdm
import netCDF4
import xarray as xr

from subsurface_DA_with_generative_models.data_handling.xarray_data import XarrayDataset
from subsurface_DA_with_generative_models.models.parameter_models.parameter_diffusion import ParameterDiffusion
from subsurface_DA_with_generative_models.models.parameter_models.parameter_SMLD import ParameterSMLD


torch.set_default_dtype(torch.float32)

import matplotlib.pyplot as plt

DEVICE = 'cpu'

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

PREPROCESSOR_LOAD_PATH = 'trained_preprocessors/preprocessor_64.pkl'

# Load up preprocessor
with open(PREPROCESSOR_LOAD_PATH, 'rb') as f:
    preprocessor = pickle.load(f)

MODEL_SAVE_PATH = 'trained_models/diffusion.pt'

CONTINUE_TRAINING = False

def main():

    fp='data/fluvialgeologicalmodels.nc'

    # dataset= xr.open_dataset(fp)
    # print(dataset)

    data = netCDF4.Dataset(fp)
    data = data.variables['facies code'][:]
    data = torch.tensor(data, dtype=torch.float32)
    data = data[0:8]
    num_samples = data.shape[0]

    batch_size = 8

    # Normalize data
    data = (data - data.min()) / (data.max() - data.min())


    # Load data
    # dataset = XarrayDataset(
    #     data_folder=FOLDER,
    #     parameter_vars=parameter_vars,
    #     output_vars=OUTPUT_VARS,
    #     preprocessor=preprocessor,
    #     num_samples=1000,
    # )

    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=16,
    #     shuffle=True,
    #     num_workers=8,
    # )

    
    # model = Unet(
    #     dim=64,
    #     dim_mults=(1, 2, 4, 8),
    #     flash_attn=True,
    #     channels=2
    # )
    # model = model.to(DEVICE)

    # diffusion = GaussianDiffusion(
    #     model,
    #     image_size = 64,
    #     timesteps = 100    # number of steps
    # )
    

    diffusion_model = ParameterSMLD(DEVICE)

    diffusion_model = diffusion_model.to(DEVICE)
    diffusion_model.train()

    if CONTINUE_TRAINING:
        diffusion_model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=1e-4)

    pbar = tqdm(range(2000))
    for epoch in pbar:
        ids = torch.randperm(num_samples)
        data = data[ids]
        for i, batch in enumerate(range(0, num_samples, batch_size)):

            data_batch = data[batch:batch+batch_size]
            data_batch = data_batch.to(DEVICE)

            #training_images = batch['static_spatial_parameters'].to(DEVICE)

            optimizer.zero_grad()


            loss = diffusion_model.diffusion(data_batch)

            loss.backward()

            optimizer.step()
        
        print(f'Epoch {epoch}, loss: {loss.item()}')

        if epoch % 500 == 0 and epoch > 0:
            diffusion_model.eval()
            sampled_images = diffusion_model.sample(batch_size = 16)
            plt.figure()
            for i in range(16):
                plt.subplot(4, 4, i+1)
                plt.imshow(sampled_images[i, 0, :, :].cpu().detach().numpy())
            plt.savefig('diffusion_samples.png')
            plt.close()

            diffusion_model.train()

            # Save model
            torch.save(diffusion_model.state_dict(), MODEL_SAVE_PATH)

    # after a lot of training

    diffusion_model.eval()
    sampled_images = diffusion_model.diffusion.sample(batch_size = 16)
    plt.figure()
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(sampled_images[i, 0, :, :].cpu().detach().numpy())
    plt.show()


if __name__ == '__main__':
    main() 