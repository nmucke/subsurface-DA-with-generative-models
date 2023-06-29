import os
import xarray as xr
import numpy as np
import torch
import yaml
import pdb
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
plt.switch_backend('agg')

# set matplotlib font size globally
plt.rcParams.update({'font.size': 5})

from tqdm import tqdm
import hamiltorch

from subsurface_DA_with_generative_models import routine 
from subsurface_DA_with_generative_models.models import GAN, WAE, parameter_GAN
from subsurface_DA_with_generative_models.optimizers.GAN_optimizer import GANOptimizer
from subsurface_DA_with_generative_models.train_steppers.GAN_train_stepper import GANTrainStepper
from subsurface_DA_with_generative_models.trainers.train_GAN import train_GAN
from subsurface_DA_with_generative_models.data_handling.xarray_data import XarrayDataset

torch.set_default_dtype(torch.float32)

torch.backends.cuda.enable_flash_sdp(enabled=True)
torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True

FORWARD_MODEL_TYPE = 'GAN'
PARAMETER_MODEL_TYPE = 'WAE'
#PARAMETER_MODEL_TYPE = 'WAE'

DEVICE = 'cuda'

FOLDER = "data/results64"
INPUT_VARS = ['Por', 'Perm'] # Porosity, Permeability, Pressure + x, y, time encodings 
DYNAMIC_INPUT_VARS = ['gas_rate',]
OUTPUT_VARS =  ['Pressure', 'CO_2']

INPUT_PREPROCESSOR_LOAD_PATH = 'trained_preprocessors/input_preprocessor_64.pkl'
DYNAMIC_INPUT_PREPROCESSOR_LOAD_PATH = 'trained_preprocessors/dynamic_input_preprocessor_64.pkl'
OUTPUT_PREPROCESSOR_LOAD_PATH = 'trained_preprocessors/output_preprocessor_64.pkl'

forward_config_path = f"configs/{FORWARD_MODEL_TYPE}.yml"
with open(forward_config_path) as f:
    forward_model_config = yaml.load(f, Loader=yaml.SafeLoader)

parameter_config_path = f"configs/{PARAMETER_MODEL_TYPE}.yml"
with open(parameter_config_path) as f:
    paramater_model_config = yaml.load(f, Loader=yaml.SafeLoader)

def observation_operator(
    input_data,
    x_obs_indices,
    y_obs_indices,
    ):

    return input_data#[:, :, x_obs_indices, y_obs_indices]

def log_posterior(
    latent_vec, 
    observations,
    input_data, 
    dynamic_input_data, 
    output_data, 
    forward_model, 
    parameter_model, 
    likelihood_dist, 
    latent_dist, 
    observation_operator,
    ):

    latent_vec = latent_vec.unsqueeze(0)

    if PARAMETER_MODEL_TYPE == 'WAE':
        generated_params = parameter_model.decoder(latent_vec)
    elif PARAMETER_MODEL_TYPE == 'parameter_GAN':
        generated_params = parameter_model.generator(latent_vec)

    generated_params = torch.tile(generated_params, (output_data.shape[0], 1, 1, 1))
    generated_params = torch.cat([generated_params, input_data[:, 2:]], dim=1)

    generated_output_data = forward_model.generator(
        input_data=generated_params,
        dynamic_input_data=dynamic_input_data
        )
    generated_observations = observation_operator(generated_output_data)
    
    log_likelihood = likelihood_dist.log_prob(generated_observations - observations).mean() 
    log_prior = latent_dist.log_prob(latent_vec).mean()

    return log_likelihood + log_prior

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
    
    input_data, dynamic_input_data, output_data = train_dataset.__getitem__(0)

    true_input_data = input_data.clone()

    input_data = input_data.to(DEVICE)
    dynamic_input_data = dynamic_input_data.to(DEVICE)
    output_data = output_data.to(DEVICE)


    # Set up models
    if PARAMETER_MODEL_TYPE == 'WAE':
        parameter_model = WAE.WassersteinAutoencoder(
            decoder_args=paramater_model_config['model_args']['decoder_args'],
            encoder_args=paramater_model_config['model_args']['encoder_args'],
        )
        state_dict = torch.load('trained_models/WAE.pt', map_location=DEVICE)
        parameter_model.load_state_dict(state_dict['model_state_dict'])
        parameter_model.to(DEVICE)
        parameter_model.eval()
    elif PARAMETER_MODEL_TYPE == 'parameter_GAN':
        parameter_model = parameter_GAN.ParameterGAN(
            generator_args=paramater_model_config['model_args']['generator_args'],
            critic_args=paramater_model_config['model_args']['critic_args'],
        )
        state_dict = torch.load('trained_models/parameter_GAN.pt', map_location=DEVICE)
        parameter_model.load_state_dict(state_dict['model_state_dict'])
        parameter_model.to(DEVICE)
        parameter_model.eval()


    forward_model = GAN.GAN(
        generator_args=forward_model_config['model_args']['generator_args'],
        critic_args=forward_model_config['model_args']['critic_args'],
    )
    state_dict = torch.load('trained_models/GAN.pt', map_location=DEVICE)
    forward_model.load_state_dict(state_dict['model_state_dict'])
    forward_model.to(DEVICE)
    forward_model.eval()

    latent_dim = parameter_model.latent_dim

    # get observations
    obs_operator = lambda input_data: observation_operator(
        input_data=input_data,
        x_obs_indices=[20, 40],
        y_obs_indices=[20, 40],
        )
    
    noise_std = 0.01
    observations = obs_operator(output_data) 
    observations = noise_std*torch.randn_like(observations)

    num_iterations = 2000

    # set up distributions
    likelihood_dist = torch.distributions.Normal(
        loc=torch.zeros_like(observations, device=DEVICE),
        scale=10*noise_std,
    )
    latent_dist = torch.distributions.Normal(
        loc=torch.zeros((1, latent_dim), device=DEVICE),
        scale=torch.ones((1, latent_dim), device=DEVICE),
    )

    # optimize latent vector
    best_loss = 1e12
    for iter in range(3):

        latent_vec = torch.randn((1, latent_dim), device=DEVICE, requires_grad=True)

        latent_optimizer = torch.optim.Adam(
            [latent_vec],
            lr=1e-2,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            latent_optimizer,
            mode='min',
            factor=0.9,
            patience=25,
            verbose=False,
        )

        pbar = tqdm(
            range(num_iterations),
            total=num_iterations,
            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'
        )
        for i in pbar:
            latent_optimizer.zero_grad()

            if PARAMETER_MODEL_TYPE == 'WAE':
                generated_params = parameter_model.decoder(latent_vec)
            elif PARAMETER_MODEL_TYPE == 'parameter_GAN':
                generated_params = parameter_model.generator(latent_vec)
            generated_params = torch.tile(generated_params, (observations.shape[0], 1, 1, 1))

            generated_params = torch.cat([generated_params, input_data[:, 2:]], dim=1)
            generated_output_data = forward_model.generator(
                input_data=generated_params,
                dynamic_input_data=dynamic_input_data
                )
            generated_observations = obs_operator(generated_output_data)
            #loss = torch.nn.MSELoss()(generated_observations, observations)# + torch.norm(latent_vec, p=2)
            #loss = torch.nn.MSELoss()(generated_params[:, 0:2], input_data[:, 0:2]) #+ 1e-4*torch.norm(latent_vec, p=2)
            loss = - likelihood_dist.log_prob(generated_observations - observations).mean() - latent_dist.log_prob(latent_vec).mean()

            loss.backward(retain_graph=True)
            latent_optimizer.step()

            scheduler.step(loss)

            pbar.set_postfix({
                'loss': loss.detach().item(),
            })
        if loss.detach().item() < best_loss or iter == 0:
            best_latent_vec = latent_vec.clone()
            best_generated_params = generated_params.clone()
            best_generated_output_data = generated_output_data.clone()

            best_loss = loss.detach().item()

    # Run MCMC
    num_MCMC_samples = 2
    num_burn = 1
    step_size = .1
    L = 5

    latent_samples = hamiltorch.sample(
        log_prob_func = lambda latent_vec: log_posterior(
            latent_vec=latent_vec,
            observations=observations,
            input_data=input_data,
            dynamic_input_data=dynamic_input_data,
            output_data=output_data,
            forward_model=forward_model,
            parameter_model=parameter_model,
            likelihood_dist=likelihood_dist,
            latent_dist=latent_dist,
            observation_operator=obs_operator,
        ), 
        params_init=best_latent_vec.squeeze(0), 
        num_samples=num_MCMC_samples,
        step_size=step_size, 
        num_steps_per_sample=L,
        sampler=hamiltorch.Sampler.HMC_NUTS,
        desired_accept_rate=0.3,
        burn=num_burn,
        integrator=hamiltorch.Integrator.IMPLICIT,
        )

    latent_samples = torch.stack(latent_samples)

    if PARAMETER_MODEL_TYPE == 'WAE':
        generated_params = parameter_model.decoder(latent_samples)
    elif PARAMETER_MODEL_TYPE == 'parameter_GAN':
        generated_params = parameter_model.generator(latent_samples)

    generated_params_list = []
    generated_output_list = []
    for i in range(num_MCMC_samples-num_burn):

        _generated_params = torch.tile(generated_params[i:i+1], (observations.shape[0], 1, 1, 1))
        _generated_params = torch.cat([_generated_params, input_data[:, 2:]], dim=1)

        generated_output_data = forward_model.generator(
            input_data=_generated_params,
            dynamic_input_data=dynamic_input_data
            )

        generated_params_list.append(_generated_params.cpu().detach())
        generated_output_list.append(generated_output_data.cpu().detach())
    
    generated_params = torch.stack(generated_params_list)
    generated_output_data = torch.stack(generated_output_list)

    generated_params = generated_params.mean(dim=0)
    generated_output_data = generated_output_data.mean(dim=0)
    

    generated_params = generated_params.cpu().detach().numpy()
    generated_output_data = generated_output_data.cpu().detach().numpy()

    
    pred_output_data = forward_model.generator(
        input_data=true_input_data.cuda(),
        dynamic_input_data=dynamic_input_data
        )
    pred_output_data = pred_output_data.cpu().detach().numpy()

    input_data = input_data.cpu().detach().numpy()
    dynamic_input_data = dynamic_input_data.cpu().detach().numpy()
    output_data = output_data.cpu().detach().numpy()

    plot_time = 50
    plt.figure()

    plt.subplot(4, 3, 1)
    plt.imshow(output_data[plot_time, 0, :, :])
    plt.title('True pressure')                           
    plt.colorbar()

    plt.subplot(4, 3, 2)
    plt.imshow(generated_output_data[plot_time, 0, :, :])
    plt.title('Inverse approx pressure')
    plt.colorbar()


    plt.subplot(4, 3, 3)
    plt.imshow(pred_output_data[plot_time, 0, :, :])
    plt.title('Forward approx pressure')
    plt.colorbar()

    plt.subplot(4, 3, 4)
    plt.imshow(output_data[plot_time, 1, :, :])
    plt.title('True CO2')                           
    plt.colorbar()

    plt.subplot(4, 3, 5)
    plt.imshow(generated_output_data[plot_time, 1, :, :])
    plt.title('Inverse approx CO2')
    plt.colorbar()

    plt.subplot(4, 3, 6)
    plt.imshow(pred_output_data[plot_time, 1, :, :])
    plt.title('Forward approx CO2')
    plt.colorbar()

    plt.subplot(4, 3, 7)
    plt.imshow(true_input_data[plot_time, 0, :, :])
    plt.title('True porosity')
    plt.colorbar()

    plt.subplot(4, 3, 8)
    plt.imshow(generated_params[plot_time, 0, :, :])
    plt.title('Inverse approx porosity')
    plt.colorbar()

    plt.subplot(4, 3, 10)
    plt.imshow(true_input_data[plot_time, 1, :, :])
    plt.title('True permeability')
    plt.colorbar()

    plt.subplot(4, 3, 11)
    plt.imshow(generated_params[plot_time, 1, :, :])
    plt.title('Inverse approx permeability')
    plt.colorbar()

    '''
    plt.subplot(3, 2, 5)
    plt.plot(output_data[:, 0, 20, 20], '-', label='True', color='tab:blue')
    plt.plot(output_data[:, 0, 40, 40], '-', color='tab:blue')
    plt.plot(generated_output_data[:, 0, 20, 20], label='Approximated', color='tab:orange')
    plt.plot(generated_output_data[:, 0, 40, 40], color='tab:orange')
    plt.title('Pressure at (20,20)')
    plt.legend()
    '''

    plt.tight_layout()
    plt.savefig('inverse_problem.pdf')
    plt.show()



if __name__ == "__main__":
    main()
   