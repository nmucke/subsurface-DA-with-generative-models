import torch
import torch.nn as nn
import torch.nn.functional as F
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

class ParameterDiffusion(nn.Module):

    def __init__(self, device,):
        super().__init__()
        

        self.model = Unet(
            dim=64,
            dim_mults=(1, 2, 4, 8),
            flash_attn=True,
            channels=2
        )

        self.diffusion = GaussianDiffusion(
            self.model,
            image_size = 64,
            timesteps = 50 
        )
    
        self.device = device
    

    def _p_sample_loop(self,latent):

        img = latent
        imgs = [latent]

        x_start = None

        for t in reversed(range(0, self.diffusion.num_timesteps)):
            self_cond = x_start if self.diffusion.self_condition else None
            img, x_start = self.diffusion.p_sample(img, t, self_cond)
            imgs.append(img)

        ret = img

        ret = self.diffusion.unnormalize(ret)

        return ret
    

    def latent_to_sample(self, latent):

        return self._p_sample_loop(latent=latent)
        
