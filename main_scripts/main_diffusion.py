import pdb
import pickle
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from tqdm import tqdm

from subsurface_DA_with_generative_models.data_handling.xarray_data import XarrayDataset

torch.set_default_dtype(torch.float32)

import matplotlib.pyplot as plt

DEVICE = 'cuda'

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

def main():

    # Load data
    dataset = XarrayDataset(
        data_folder=FOLDER,
        parameter_vars=parameter_vars,
        output_vars=OUTPUT_VARS,
        preprocessor=preprocessor,
        num_samples=1000,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
    )

    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=True,
        channels=2
    )
    model = model.to(DEVICE)

    diffusion = GaussianDiffusion(
        model,
        image_size = 64,
        timesteps = 100    # number of steps
    )
    diffusion = diffusion.to(DEVICE)
    diffusion.train()

    optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-4)

    pbar = tqdm(range(2000))
    for epoch in pbar:
        for i, batch in enumerate(dataloader):

            optimizer.zero_grad()

            training_images = batch['static_spatial_parameters'].to(DEVICE)

            loss = diffusion(training_images)
            loss.backward()

            optimizer.step()
        
        print(f'Epoch {epoch}, loss: {loss.item()}')

    # after a lot of training
    diffusion.eval()

    sampled_images = diffusion.sample(batch_size = 16)
    plt.figure()
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(sampled_images[i, 0, :, :].cpu().detach().numpy())
    plt.show()


if __name__ == '__main__':
    main()