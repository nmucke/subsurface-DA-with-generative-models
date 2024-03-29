import pdb
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from subsurface_DA_with_generative_models.preprocessor import Preprocessor
import ray


#@ray.remote
def compute_parameter_posterior(prior, C_up, C_pp, r, R, h):
    return prior + torch.matmul(C_up, torch.linalg.solve(C_pp + 1 / h * R, r))

class EnsembleKalmanInversion():
    def __init__(
        self,
        forward_model: nn.Module,
        fixed_input: dict,
        preprocessor: Preprocessor = None,
        num_particles: int = 100,
        num_iterations: int = 100,
        parameter_dim: int = (2, 64, 64),
        device: str = 'cuda',
    ) -> None:
        
        self.device = device
        
        self.forward_model = forward_model
        self.fixed_input = fixed_input
        for key in fixed_input:
            if fixed_input[key] is not None:
                self.fixed_input[key] = self.fixed_input[key].unsqueeze(0)
                _dim = torch.ones(len(fixed_input[key].shape), dtype=torch.int)
                _dim[0] = num_particles
                self.fixed_input[key] = self.fixed_input[key].repeat(_dim.tolist())

        self.preprocessor = preprocessor
        
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.h = 1/self.num_iterations

        self.parameter_dim = parameter_dim
        self.num_parameter_dofs = parameter_dim[0]*parameter_dim[1]*parameter_dim[2]

        self.batch_size = 25

        self.output_dim = (self.num_particles, 2, 61, 64, 64)


    def _compute_ensemble(
        self, 
        parameters: torch.Tensor
    ) -> torch.Tensor:
        
        parameters = parameters.reshape((self.num_particles, *self.parameter_dim))

        # Compute model output in batch
        model_output = torch.zeros(self.output_dim)

        with torch.no_grad():
            for i in range(0, self.num_particles, self.batch_size):
                
                batch_fixed_input = {}
                for key in self.fixed_input:
                    if self.fixed_input[key] is not None:
                        batch_fixed_input[key] = self.fixed_input[key][i:i+self.batch_size].to(self.device)

                parameters_batch = parameters[i:i+self.batch_size].to(self.device) 

                #with torch.autocast(device_type='cuda', dtype=torch.float16):
                model_output[i:i+self.batch_size] = self.forward_model.forward_model(
                    static_spatial_parameters=parameters_batch,
                    **batch_fixed_input
                ).cpu()

        if self.preprocessor is not None:
            for i in range(model_output.shape[0]):
                model_output[i] = self.preprocessor.output.inverse_transform(model_output[i])

        return model_output
    
    def solve(
        self, 
        observation_operator: callable,
        observations: torch.Tensor,
    ):

        self.forward_model.eval()

        num_observations = observations.shape[0]

        R = 0.005*torch.eye(observations.shape[0])

        _parameter_ensemble = torch.rand((self.num_particles, 2, 64, 64))

        batch_size = 25
        parameter_ensemble = torch.zeros((self.num_particles, 2, 64, 64))
        for i in range(0, self.num_particles, batch_size):
            sample = self.forward_model.parameter_model(_parameter_ensemble[i:i+batch_size].to(self.device))
            parameter_ensemble[i:i+batch_size] = sample.detach().cpu()
        
        parameter_ensemble = parameter_ensemble.reshape((self.num_particles, self.num_parameter_dofs))
        
        pbar = tqdm(
            enumerate(range(self.num_iterations)),
            total=self.num_iterations,
            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'
        )
        for i in pbar:
            
            # Compute prior output
            output_prior = self._compute_ensemble(parameter_ensemble)

            # Compute prior observations
            t1 = time.time()
            obs_prior = [observation_operator(output_prior[j]) for j in range(self.num_particles)]
            obs_prior = torch.stack(obs_prior)

            # Compute prior mean of output and observations
            parameter_prior_mean = parameter_ensemble.mean(dim=0)
            obs_prior_mean = obs_prior.mean(dim=0)
            
            # Compute prior covariance of output and observations
            C_pp = torch.zeros((num_observations, num_observations))
            C_up = torch.zeros((self.num_parameter_dofs, num_observations))
            for j in range(self.num_particles):
                C_pp += torch.outer(obs_prior[j, :]-obs_prior_mean,obs_prior[j, :]-obs_prior_mean)
                C_up += torch.outer(parameter_ensemble[j, :]-parameter_prior_mean, obs_prior[j, :] - obs_prior_mean)
            C_pp /= self.num_particles
            C_up /= self.num_particles

            # Perturb observations
            obs_perturbed = torch.zeros((self.num_particles, num_observations))
            for j in range(self.num_particles):
                obs_perturbed[j] = observations + torch.normal(torch.zeros(num_observations), 1/self.h*torch.diag(R))
            
            # Compute residuals
            r = obs_perturbed - obs_prior

            LU, pivots = torch.linalg.lu_factor(C_pp + 1 / self.h * R)
            LU = LU.unsqueeze(0).repeat(self.num_particles, 1, 1)
            pivots = pivots.unsqueeze(0).repeat(self.num_particles, 1)
            mat_r_solve = torch.linalg.lu_solve(LU, pivots, r.unsqueeze(-1)).squeeze(-1)

            parameter_posterior = torch.zeros((self.num_particles, self.num_parameter_dofs))
            for j in range(self.num_particles):
                parameter_posterior[j] = parameter_ensemble[j] + torch.matmul(C_up, mat_r_solve[j])

            '''
            # Compute parameter posterior
            parameter_posterior = torch.zeros((self.num_particles, self.num_parameter_dofs))
            for j in range(self.num_particles):
                
                parameter_posterior[j] = parameter_ensemble[j] + torch.matmul(C_up, mat_r_solve)

                #post = compute_parameter_posterior.remote(
                post = compute_parameter_posterior(
                    prior=parameter_ensemble[j], 
                    C_up=C_up, 
                    C_pp=C_pp, 
                    r=r[j], 
                    R=R, 
                    h=self.h
                )
                #parameter_posterior.append(post)
            '''
            
            #parameter_posterior = ray.get(parameter_posterior)
            #parameter_posterior = torch.stack(parameter_posterior)

            parameter_ensemble = parameter_posterior
        
        output_posterior = self._compute_ensemble(parameter_ensemble)

        return parameter_ensemble.detach().cpu(), output_posterior.detach().cpu()