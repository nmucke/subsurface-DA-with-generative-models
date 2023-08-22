
import pdb
import pickle
import numpy as np
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

import hamiltorch

import torch

from scipy.sparse.linalg import svds

import torch.nn as nn


import matplotlib.pyplot as plt
import yaml
from subsurface_DA_with_generative_models.data_handling.data_utils import prepare_batch
from subsurface_DA_with_generative_models.data_handling.xarray_data import XarrayDataset

from subsurface_DA_with_generative_models.models.forward_models import u_net


DEVICE = 'cuda'
SPACE_DIM = 32 
LOAD_MODEL = True

class PCAReconstructor(nn.Module):
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim

    def fit(self, data):

        self.V = torch.randn(
            (self.latent_dim, data.shape[1]),
            device=DEVICE,
        )

        data = data.to(DEVICE)

        self.V.requires_grad = True

        optim = torch.optim.Adam([self.V], lr=1e-2)

        reconstruct_loss_fn = torch.nn.L1Loss()
        #reconstruct_loss_fn = torch.nn.MSELoss()
        ortho_loss_fn = torch.nn.MSELoss()
        
        identity = torch.eye(self.latent_dim, device=DEVICE)

        num_iterations = 50000
        for i in range(num_iterations):
                
                optim.zero_grad()

                ortho_loss = ortho_loss_fn(torch.matmul(self.V, self.V.T), identity)
                #latent = self.reduce(data)

                latent = torch.matmul(self.V, data.T).T

                #reconstruct = self.reconstruct(latent) 
                reconstruct = torch.matmul(self.V.T, latent.T).T
                reconstruct_loss = reconstruct_loss_fn(reconstruct, data)

                loss = reconstruct_loss + 1e-3*ortho_loss

                #loss += 1e-6*torch.norm(self.V, p=1)

                loss.backward()
                optim.step()
    
                if i % 100 == 0:
                    print(f"Loss: {loss.item()}, Iteration: {i}")

        '''
        self.U, self.S, self.V = torch.linalg.svd(data)

        self.U = self.U[:, :self.latent_dim]
        self.S = self.S[:self.latent_dim]
        self.V = self.V[:self.latent_dim]

        # require grad
        self.U.requires_grad = True
        self.S.requires_grad = True
        self.V.requires_grad = True
        '''

    def reduce(self, data):
        return torch.matmul(self.V, data)

    def reconstruct(self, data):
        return torch.matmul(self.V.T, data)
    
    def forward(self, data):
        return self.reconstruct(data)
    


def log_posterior(
    latent_vec, 
    observed_data, 
    parameter_model, 
    likelihood_dist, 
    latent_dist, 
    forward_model,
    true_porosity,
    ):

    generated_params = parameter_model.reconstruct(latent_vec)
    generated_params = generated_params.reshape(1, 1, SPACE_DIM, SPACE_DIM)
    generated_params = torch.cat([true_porosity, generated_params], dim=1)
    
    generated_observations = forward_model(generated_params)

    log_likelihood = likelihood_dist.log_prob(
        generated_observations[0, 0, :, 0::1, 0::1] - observed_data[0, :, 0::1, 0::1]
    ).mean()

    log_prior = latent_dist.log_prob(latent_vec).mean()

    return log_likelihood# + log_prior


FOLDER = "data/results32"
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

# Load up preprocessor
preprocessor_load_path = 'trained_preprocessors/preprocessor_32.pkl'
with open(preprocessor_load_path, 'rb') as f:
    preprocessor = pickle.load(f)


config_path = f"configs/UNet.yml"
with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)
model = u_net.UNet(
    **config['model_args'],
)
model.to(DEVICE)



def main():

    # Load data
    dataset = XarrayDataset(
        data_folder=FOLDER,
        parameter_vars=parameter_vars,
        output_vars=OUTPUT_VARS,
        preprocessor=preprocessor,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    permeability = []
    num_train_samples = 700
    for i, batch in enumerate(dataloader):

        perm = batch['static_spatial_parameters'][0, 1]
        permeability.append(perm)

        if i == num_train_samples:
            break
        
    permeability = torch.stack(permeability)
    permeability = permeability.reshape(num_train_samples+1, SPACE_DIM * SPACE_DIM)

    #true_perm = permeability[10]

    batch = dataset.__getitem__(900)
    true_perm = batch['static_spatial_parameters'][1].flatten()
    true_perm = true_perm.to(DEVICE)


    # unpack batch
    static_point_parameters, static_spatial_parameters, \
    dynamic_point_parameters, dynamic_spatial_parameters, \
    output_variables = prepare_batch(
        batch=batch,
        device=DEVICE,
    )

    true_porosity = batch['static_spatial_parameters'][0].unsqueeze(0).unsqueeze(0).to(DEVICE)

    fixed_input = {
        'static_point_parameters': static_point_parameters,
        'dynamic_point_parameters': dynamic_point_parameters.unsqueeze(0).to(DEVICE),
        'dynamic_spatial_parameters': dynamic_spatial_parameters.unsqueeze(0).to(DEVICE),
    }

    forward_model = lambda x: model(
        static_spatial_parameters=x, 
        **fixed_input
        )


    latent_dim = 150

    pca = PCAReconstructor(latent_dim=latent_dim)
    if not LOAD_MODEL:
        pca.fit(permeability)
        torch.save(pca, 'pca.pkl')

    else:
        pca = torch.load('pca.pkl')



    latent_list = []
    for i in range(num_train_samples+1):
        latent_list.append(pca.reduce(permeability[i].to(DEVICE)))

    latent_samples = torch.stack(latent_list)
    latent_samples = latent_samples.flatten()
    
    noise_std = 0.001

    # set up distributions
    likelihood_dist = torch.distributions.Normal(
        loc=torch.zeros_like(output_variables[0, :, 0::1, 0::1], device=DEVICE),
        scale=10*noise_std,
    )
    latent_dist = torch.distributions.Normal(
        loc=latent_samples.mean()*torch.ones((1, latent_dim), device=DEVICE),
        scale=latent_samples.std()*torch.ones((1, latent_dim), device=DEVICE),
    )

    # Run MCMC
    num_MCMC_samples = 5000
    num_burn = 4000
    step_size = 1.0
    L = 5

    init_latent = torch.randn(latent_dim, device=DEVICE, requires_grad=True)


    batch = dataset.__getitem__(950)
    init_perm = batch['static_spatial_parameters'][1].flatten()
    init_perm = init_perm.to(DEVICE)
    _init_latent = pca.reduce(init_perm)

    init_latent = _init_latent.clone().detach().requires_grad_(True)

    optim = torch.optim.Adam([init_latent], lr=1e-1)

    obs = output_variables# + noise_std*torch.randn_like(output_variables, device=DEVICE)
    num_iterations = 2000
    for i in range(num_iterations):

        optim.zero_grad()
        loss = -log_posterior(
            latent_vec=init_latent, 
            observed_data=obs,
            parameter_model=pca, 
            likelihood_dist=likelihood_dist, 
            latent_dist=latent_dist, 
            forward_model=forward_model,
            true_porosity=true_porosity,
        )
        loss.backward(retain_graph=True)
        optim.step()

        if i % 100 == 0:
            print(f"Loss: {loss.item()}, Iteration: {i}")

    max_posterior_latent = init_latent.clone()

    max_post_reconstruct = pca.reconstruct(max_posterior_latent).cpu().detach().numpy()
    
    latent_samples = hamiltorch.sample(
        log_prob_func = lambda latent_vec: log_posterior(
            latent_vec=latent_vec,
            observed_data=obs, 
            parameter_model=pca, 
            likelihood_dist=likelihood_dist, 
            latent_dist=latent_dist, 
            forward_model=forward_model,
            true_porosity=true_porosity,
        ), 
        params_init=init_latent, 
        num_samples=num_MCMC_samples,
        step_size=step_size, 
        num_steps_per_sample=L,
        sampler=hamiltorch.Sampler.HMC_NUTS,
        desired_accept_rate=0.3,
        burn=num_burn,
        integrator=hamiltorch.Integrator.IMPLICIT,
    )


    latent_samples = torch.stack(latent_samples)
    num_samples = num_MCMC_samples-num_burn

    latent_samples = latent_samples.reshape(num_samples, latent_dim)

    reconstruct_samples = []
    for i in range(num_samples):
        reconstruct_samples.append(pca.reconstruct(latent_samples[i]))

    reconstruct_samples = torch.stack(reconstruct_samples)
    reconstruct_samples = reconstruct_samples.detach().cpu().numpy()

    reconstruct_mean = reconstruct_samples.mean(axis=0)

    best_latent = pca.reduce(true_perm)
    best_reconstruct = pca.reconstruct(best_latent).cpu().detach().numpy()

    true_perm = true_perm.cpu().detach().numpy()

    plt.figure()
    plt.subplot(1, 4, 1)
    plt.title("Reconstructed MCMC mean")
    plt.imshow(reconstruct_mean.reshape(SPACE_DIM, SPACE_DIM), vmin=true_perm.min(), vmax=true_perm.max())
    plt.colorbar()

    plt.subplot(1, 4, 2)
    plt.title("True")
    plt.imshow(true_perm.reshape(SPACE_DIM, SPACE_DIM), vmin=true_perm.min(), vmax=true_perm.max())
    plt.colorbar()

    plt.subplot(1, 4, 3)
    plt.title("Best")
    plt.imshow(best_reconstruct.reshape(SPACE_DIM, SPACE_DIM), vmin=true_perm.min(), vmax=true_perm.max())
    plt.colorbar()

    plt.subplot(1, 4, 4)
    plt.title("Max posterior")
    plt.imshow(max_post_reconstruct.reshape(SPACE_DIM, SPACE_DIM), vmin=true_perm.min(), vmax=true_perm.max())
    plt.colorbar()  


    plt.show()






if __name__ == "__main__":
    main()