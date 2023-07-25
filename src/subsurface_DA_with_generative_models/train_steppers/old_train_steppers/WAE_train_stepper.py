import pdb
import torch
import torch.nn as nn
import matplotlib.pyplot as plt 

from subsurface_DA_with_generative_models.optimizers.GAN_optimizer import GANOptimizer
from subsurface_DA_with_generative_models.train_steppers.base_train_stepper import BaseTrainStepper

def MMD_loss(
    x: torch.Tensor, 
    y: torch.Tensor,
    kernel: str = "multiscale",
    device: str = "cpu",
    ) -> torch.Tensor:
    """
    Emprical maximum mean discrepancy. The lower the result, 
    the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
        '''
        C = 2*x.shape[-1]*1
        XX += C * (C + dxx)**-1
        YY += C * (C + dyy)**-1
        XY += C * (C + dxy)**-1
        '''
            
    if kernel == "rbf":
      
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)

    return torch.mean(XX + YY - 2. * XY)



class WAETrainStepper(BaseTrainStepper):

    def __init__(
        self,
        model: nn.Module,
        optimizer: GANOptimizer,
        MMD_regu: str,
    ) -> None:

        self.model = model
        self.optimizer = optimizer
        self.MMD_regu = MMD_regu

        self.device = model.device

        self.MSE_loss = nn.MSELoss()

    def _sample_latent(self, shape: torch.Tensor) -> torch.Tensor:
        return torch.randn(shape, device=self.device)
    
    def step_scheduler(self) -> None:
        self.optimizer.step_scheduler()


    def _train_step(
        self, 
        input_data: torch.Tensor = None,
    ):
        
        self.model.train()

        self.optimizer.zero_grad()

        # compute critic loss for fake data
        true_latent_samples = self._sample_latent(
            shape=(input_data.shape[0], self.model.latent_dim)
        )

        latent_samples = self.model.encoder(
            input_data=input_data
        )

        recosntructed_input_data = self.model.decoder(
            latent_samples=latent_samples,
        )

        recon_loss = self.MSE_loss(
            recosntructed_input_data,
            input_data
        )

        latent_loss = MMD_loss(
            true_latent_samples,
            latent_samples,
            kernel="multiscale",
            device=self.device,
        )

        loss = recon_loss + self.MMD_regu*latent_loss

        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return recon_loss.detach().item(), latent_loss.detach().item()

    def train_step(
        self,
        input_data: torch.Tensor,
        ) -> None:


        # train critic
        recon_loss, latent_loss = self._train_step(
            input_data=input_data,
            )

        return {
            'recon_loss': recon_loss,
            'latent_loss': latent_loss,
        }

    def val_step(
        self,
        output_data: torch.Tensor,
        input_data: torch.Tensor,
    ) -> None:
        
        return {
            'gen_loss': 0,
            'critic_loss': 0
        }

    def save_model(self, path: str) -> None:
        torch.save(self.model, path)