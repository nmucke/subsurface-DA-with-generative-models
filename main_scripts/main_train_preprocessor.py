import pdb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle

from subsurface_DA_with_generative_models.data_handling.xarray_data import ParameterDataset, XarrayDataset
from subsurface_DA_with_generative_models.preprocessor import MinMaxTransformer, Preprocessor

FOLDER = "data/results32"
STATIC_POINT_VARS = None
STATIC_SPATIAL_VARS = ['Por', 'Perm']
DYNAMIC_SPATIAL_VARS = ['time_encoding']
DYNAMIC_POINT_VARS = ['gas_rate']
OUTPUT_VARS = ['Pressure']

PREPROCESSOR_SAVE_PATH = 'trained_preprocessors/preprocessor_32.pkl'

parameter_vars = {
    'static_point': STATIC_POINT_VARS,
    'static_spatial': STATIC_SPATIAL_VARS,
    'dynamic_point': DYNAMIC_POINT_VARS,
    'dynamic_spatial': DYNAMIC_SPATIAL_VARS,
}

def main():

    # Load data
    train_dataset = XarrayDataset(
        data_folder=FOLDER,
        parameter_vars=parameter_vars,
        output_vars=OUTPUT_VARS,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
    )

    # Set up preprocessor
    preprocessor = Preprocessor(
        type='MinMaxTransformer',
        static_point=False,
        static_spatial=True,
        dynamic_point=True,
        dynamic_spatial=True,
        output=True
    )
    
    for i, batch in enumerate(train_dataloader):
        preprocessor.partial_fit(batch) 

    print("Preprocessor fitted!")


    # Load data
    parameter_dataset = ParameterDataset(
        path = 'data/parameters_processed',
    )

    parameter_dataset_dataloader = DataLoader(
        parameter_dataset,
    )

    for i, batch in enumerate(parameter_dataset_dataloader):
        preprocessor.partial_fit(batch, variable_names='static_spatial') 
    print("Preprocessor fitted!")

    # Save the preprocessor
    with open(PREPROCESSOR_SAVE_PATH, 'wb') as f:
        pickle.dump(preprocessor, f)
    
    print("Preprocessor saved!")
    
if __name__ == "__main__":
    main()



