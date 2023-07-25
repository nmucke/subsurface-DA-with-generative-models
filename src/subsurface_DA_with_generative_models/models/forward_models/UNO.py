import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from subsurface_DA_with_generative_models.models.model_utils.UNO_utils import OperatorBlock_2D


class UNO(nn.Module):
    def __init__(
        self,
        in_width = 2,
        width = 64,
        pad = 0, 
        factor = 3/4
        ):
        super(UNO, self).__init__()


        self.in_width = in_width # input channel
        self.width = width 
        self.factor = factor
        self.padding = pad  

        self.fc = nn.Linear(self.in_width, self.width//2)

        self.fc0 = nn.Linear(self.width//2, self.width) # input channel is 3: (a(x, y), x, y)

        self.L0 = OperatorBlock_2D(self.width, 2*factor*self.width,48, 48, 22, 22)

        self.L1 = OperatorBlock_2D(2*factor*self.width, 4*factor*self.width, 32, 32, 14,14)

        self.L2 = OperatorBlock_2D(4*factor*self.width, 8*factor*self.width, 16, 16,6,6)
        
        self.L3 = OperatorBlock_2D(8*factor*self.width, 8*factor*self.width, 16, 16,6,6)
        
        self.L4 = OperatorBlock_2D(8*factor*self.width, 4*factor*self.width, 32, 32,6,6)

        self.L5 = OperatorBlock_2D(8*factor*self.width, 2*factor*self.width, 48, 48,14,14)

        self.L6 = OperatorBlock_2D(4*factor*self.width, self.width, 64, 64,22,22) # will be reshaped

        self.fc1 = nn.Linear(2*self.width, 4*self.width)
        self.fc2 = nn.Linear(4*self.width, 1)
    
    @property
    def device(self):
        return next(self.parameters()).device.type

    def prepare_spatial_data(
        self,
        static_point_parameters: torch.Tensor = None,
        static_spatial_parameters: torch.Tensor = None,
        dynamic_point_parameters: torch.Tensor = None,
        dynamic_spatial_parameters: torch.Tensor = None
    ):
        
        batch_size = static_spatial_parameters.shape[0]
        num_input_channels = static_spatial_parameters.shape[1]
        x_dim = static_spatial_parameters.shape[2]
        y_dim = static_spatial_parameters.shape[3]
        num_time_steps = dynamic_point_parameters.shape[-1]
        
        static_spatial_parameters = static_spatial_parameters.unsqueeze(dim=2)
        static_spatial_parameters = static_spatial_parameters.repeat(1, 1, num_time_steps, 1, 1)
        static_spatial_parameters = static_spatial_parameters.transpose(1, 2)
        static_spatial_parameters = static_spatial_parameters.reshape(
            batch_size*num_time_steps, num_input_channels, x_dim, y_dim
            )
        dynamic_spatial_parameters = dynamic_spatial_parameters.transpose(1, 2)
        dynamic_spatial_parameters = dynamic_spatial_parameters.reshape(
            batch_size*num_time_steps, dynamic_spatial_parameters.shape[2], x_dim, y_dim
            )

        spatial_parameters = torch.cat(
            (static_spatial_parameters, dynamic_spatial_parameters), 
            dim=1
            )

        spatial_parameters = static_spatial_parameters

        return spatial_parameters

    def forward(
        self, 
        static_point_parameters: torch.Tensor = None,
        static_spatial_parameters: torch.Tensor = None,
        dynamic_point_parameters: torch.Tensor = None,
        dynamic_spatial_parameters: torch.Tensor = None
        ):

        spatial_parameters = self.prepare_spatial_data(
            static_point_parameters=static_point_parameters,
            static_spatial_parameters=static_spatial_parameters,
            dynamic_point_parameters=dynamic_point_parameters,
            dynamic_spatial_parameters=dynamic_spatial_parameters
        )


        #grid = self.get_grid(x.shape, x.device)
        #x = torch.cat((x, grid), dim=-1)

        x = spatial_parameters
         
        x_fc = self.fc(x)
        x_fc = F.gelu(x_fc)

        x_fc0 = self.fc0(x_fc)
        x_fc0 = F.gelu(x_fc0)
        
        x_fc0 = x_fc0.permute(0, 3, 1, 2)
        
        
        x_fc0 = F.pad(x_fc0, [self.padding,self.padding, self.padding,self.padding])
        
        D1,D2 = x_fc0.shape[-2],x_fc0.shape[-1]
        
        x_c0 = self.L0(x_fc0,int(D1*self.factor),int(D2*self.factor))
        x_c1 = self.L1(x_c0 ,D1//2,D2//2)

        x_c2 = self.L2(x_c1 ,D1//4,D2//4)        
        x_c3 = self.L3(x_c2,D1//4,D2//4)
        x_c4 = self.L4(x_c3,D1//2,D2//2)
        x_c4 = torch.cat([x_c4, x_c1], dim=1)
        x_c5 = self.L5(x_c4,int(D1*self.factor),int(D2*self.factor))
        x_c5 = torch.cat([x_c5, x_c0], dim=1)
        x_c6 = self.L6(x_c5,D1,D2)
        x_c6 = torch.cat([x_c6, x_fc0], dim=1)

        if self.padding!=0:
            x_c6 = x_c6[..., :-self.padding, :-self.padding]

        x_c6 = x_c6.permute(0, 2, 3, 1)
        
        x_fc1 = self.fc1(x_c6)
        x_fc1 = F.gelu(x_fc1)
        
        x_out = self.fc2(x_fc1)
        
        return x_out
    
