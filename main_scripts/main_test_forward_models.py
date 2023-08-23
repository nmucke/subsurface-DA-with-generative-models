import os
import pickle
import xarray as xr
import numpy as np
import torch
import yaml
import pdb
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from subsurface_DA_with_generative_models.model_testing import ForwardModelTester
from subsurface_DA_with_generative_models.models.forward_models.FNO3D import FNO3d


# use matplotlib agg backend so figures can be saved in the background
plt.switch_backend('Agg')
#
from subsurface_DA_with_generative_models.models.forward_models import u_net_GAN
from subsurface_DA_with_generative_models.models.forward_models import u_net
from subsurface_DA_with_generative_models.data_handling.xarray_data import XarrayDataset

torch.set_default_dtype(torch.float32)

torch.backends.cuda.enable_flash_sdp(enabled=True)
torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True



def main():
    all_metrics = {}
     #MODEL_PATH = 'UNet_32x32_Pressure_200samples_minmax'
    MODEL_PATHS = {
    'UNet': [
        'UNet_32x32_Pressure_1000samples_normGaussian',
        'UNet_32x32_Pressure_100samples_normGaussian',
        'UNet_32x32_Pressure_200samples_normGaussian',
        'UNet_32x32_Pressure_500samples_normGaussian',
        
    ],
        
    'FNO3D': [
        'FNO3D_32x32_Pressure_1000samples_12modes_128width_normGaussian',
        'FNO3D_32x32_Pressure_1000samples_12modes_64width_normGaussian',
        'FNO3D_32x32_Pressure_1000samples_18modes_128width_normGaussian',
        'FNO3D_32x32_Pressure_1000samples_18modes_64width_normGaussian',
        'FNO3D_32x32_Pressure_1000samples_6modes_128width_normGaussian',
        'FNO3D_32x32_Pressure_1000samples_6modes_64width_normGaussian',
        'FNO3D_32x32_Pressure_100samples_12modes_128width_normGaussian',
        'FNO3D_32x32_Pressure_100samples_12modes_64width_normGaussian',
        'FNO3D_32x32_Pressure_100samples_18modes_128width_normGaussian', 
        'FNO3D_32x32_Pressure_100samples_18modes_64width_normGaussian',
        'FNO3D_32x32_Pressure_100samples_6modes_128width_normGaussian',
        'FNO3D_32x32_Pressure_100samples_6modes_64width_normGaussian',
        'FNO3D_32x32_Pressure_200samples_12modes_128width_normGaussian',
        'FNO3D_32x32_Pressure_200samples_12modes_64width_normGaussian',
        'FNO3D_32x32_Pressure_200samples_18modes_128width_normGaussian', 
        'FNO3D_32x32_Pressure_200samples_18modes_64width_normGaussian',
        'FNO3D_32x32_Pressure_200samples_6modes_128width_normGaussian',
        'FNO3D_32x32_Pressure_200samples_6modes_64width_normGaussian',
        'FNO3D_32x32_Pressure_500samples_12modes_128width_normGaussian',
        'FNO3D_32x32_Pressure_500samples_12modes_64width_normGaussian',
        'FNO3D_32x32_Pressure_500samples_18modes_128width_normGaussian',
        'FNO3D_32x32_Pressure_500samples_18modes_64width_normGaussian',
        'FNO3D_32x32_Pressure_500samples_6modes_128width_normGaussian',
        'FNO3D_32x32_Pressure_500samples_6modes_64width_normGaussian'
    ]

    }
    
    MODELS_TO_TEST = ['UNet', 'FNO3D'] #['FNO3D'] #, 'UNetGAN']
    DEVICE =  'cpu' #'cuda'

    

    FOLDER = "data/32x32_test"
    STATIC_POINT_VARS = None
    STATIC_SPATIAL_VARS = ['Por', 'Perm']
    DYNAMIC_SPATIAL_VARS = ['time_encoding']
    DYNAMIC_POINT_VARS = ['gas_rate']
    OUTPUT_VARS = ['Pressure'] #, 'CO_2']
    
    parameter_vars = {
        'static_point': STATIC_POINT_VARS,
        'static_spatial': STATIC_SPATIAL_VARS,
        'dynamic_point': DYNAMIC_POINT_VARS,
        'dynamic_spatial': DYNAMIC_SPATIAL_VARS,
    }
    
    all_metrics = {model_type: {} for model_type in MODEL_PATHS.keys()}
    
    for model_type, model_paths in MODEL_PATHS.items():
        for PATH in model_paths:
            
            load_path = f'trained_models/{PATH}'
            model_file_path = f'{load_path}/model.pt'

            # Check if the model file exists
            if not os.path.exists(model_file_path):
                print(f"Model file not found at {model_file_path}. Skipping...")
                continue
            
            try:
                if model_type == 'UNetGAN' or model_type == 'UNet':
                    preprocessor_load_path = 'trained_preprocessors/preprocessor_gaussian_32x32_Pressure.pkl'
                elif model_type == 'FNO3D' and 'x_encoding' not in STATIC_SPATIAL_VARS:
                    preprocessor_load_path = 'trained_preprocessors/preprocessor_gaussian_32x32_Pressure.pkl'
                    STATIC_SPATIAL_VARS += ['x_encoding', 'y_encoding']


                # Load up preprocessor
                with open(preprocessor_load_path, 'rb') as f:
                    preprocessor = pickle.load(f)

                
                # Load data
                dataset = XarrayDataset(
                    data_folder=FOLDER,
                    parameter_vars=parameter_vars,
                    output_vars=OUTPUT_VARS,
                    preprocessor=preprocessor,
                    num_samples=100,
                )

                dataloader = DataLoader(
                    dataset=dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                )

                config_path = f"{load_path}/config.yml"
                print(f'Loading config from {config_path}')
                with open(config_path) as f:
                    config = yaml.load(f, Loader=yaml.SafeLoader)

                # Set up model
                if model_type == 'UNetGAN':
                    model = u_net_GAN.UNetGAN(
                        generator_args=config['model_args']['generator_args'],
                        critic_args=config['model_args']['critic_args'],
                    )
                elif model_type == 'FNO3D':
                    model = FNO3d(
                        **config['model_args'],
                    )
                elif model_type == 'UNet':
                    model = u_net.UNet(
                        **config['model_args'],
                    )
                model.to(DEVICE)

                model.eval()
                
                # Load tained model weights
                state_dict = torch.load(f'{load_path}/model.pt', map_location=DEVICE)
                model.load_state_dict(state_dict['model_state_dict'])


                model_tester = ForwardModelTester(
                    forward_model=model,
                    preprocessor=preprocessor,
                    device=DEVICE,
                    metrics=['RMSE', 'RRMSE'],
                )

                # Compute metrics
                print(f'Computing metrics for {PATH}')
                metrics = model_tester.test_model(
                    dataloader=dataloader,
                    print_metrics=True,
                    return_metrics=True,
                )
                
                # Compute metrics
                all_metrics[model_type][PATH] = {metric: logger.average for metric, logger in metrics.items()}           


                # Plot
                model_tester.plot(
                    dataloader=dataloader,
                    save_path=load_path, 
                    plot_time=-1,
                    plot_x_y=(16, 16),
                    sample_index=0,
                    show_plot=True,
                )
                
            except RuntimeError as e:
            # Handle the specific error related to loading the model
                print(f"Error loading model from {model_file_path}: {str(e)}. Skipping...")
                continue
            del model
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()
            
    with open('model_metrics.pkl', 'wb') as f:
        pickle.dump(all_metrics, f)

    print("Metrics saved to 'model_metrics.pkl'")


if __name__ == "__main__":
    main()