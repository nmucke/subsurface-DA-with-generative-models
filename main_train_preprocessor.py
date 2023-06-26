import pdb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle

from subsurface_DA_with_generative_models.data_handling.xarray_data import XarrayDataset
from subsurface_DA_with_generative_models.preprocessor import MinMaxTransformer

FOLDER = "data/results64"
INPUT_VARS = ['Por', 'Perm'] # Porosity, Permeability, Pressure + x, y, time encodings 
DYNAMIC_INPUT_VARS = ['p_0_reservoir_P', 'BHP', 'p_0_c_2_rate', 'gas_rate', 'p_0_c_0_rate', 'c_0_rate', 'wat_rate', 'p_0_c_1_rate', 'c_2_rate', 'c_1_rate']
OUTPUT_VARS = ['Pressure', 'CO_2']

INPUT_PREPROCESSOR_SAVE_PATH = 'trained_preprocessors/input_preprocessor_64.pkl'
DYNAMIC_INPUT_PREPROCESSOR_SAVE_PATH = 'trained_preprocessors/dynamic_input_preprocessor_64.pkl'
OUTPUT_PREPROCESSOR_SAVE_PATH = 'trained_preprocessors/output_preprocessor_64.pkl'

def main():

    # Load data
    train_dataset = XarrayDataset(
        folder=FOLDER,
        input_vars=INPUT_VARS,
        output_vars=OUTPUT_VARS,
        dynamic_input_vars=DYNAMIC_INPUT_VARS,
        include_spatial_coords=False,
        include_time=True,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
    )

    # Set up preprocessor
    input_min_max_transformer = MinMaxTransformer(
        num_channels=3
    )
    dynamic_min_max_transformer = MinMaxTransformer(
        num_channels=10
    )
    output_min_max_transformer = MinMaxTransformer(
        num_channels=2
    )

    for i, (input_data, dynamic_input_data, output_data) in enumerate(train_dataloader):
        
        input_data = input_data.view(-1, input_data.shape[2], input_data.shape[3], input_data.shape[4])

        dynamic_input_data = dynamic_input_data.view(-1, dynamic_input_data.shape[2], dynamic_input_data.shape[3])

        output_data = output_data.view(-1, output_data.shape[2], output_data.shape[3], output_data.shape[4])

        # Fit the preprocessor
        input_min_max_transformer.fit_in_batches(input_data)
        dynamic_min_max_transformer.fit_in_batches(dynamic_input_data)
        output_min_max_transformer.fit_in_batches(output_data)
    
    print("Preprocessor fitted!")
    

    # Save the preprocessor
    with open(INPUT_PREPROCESSOR_SAVE_PATH, 'wb') as f:
        pickle.dump(input_min_max_transformer, f)

    with open(DYNAMIC_INPUT_PREPROCESSOR_SAVE_PATH, 'wb') as f:
        pickle.dump(dynamic_min_max_transformer, f)
    
    with open(OUTPUT_PREPROCESSOR_SAVE_PATH, 'wb') as f:
        pickle.dump(output_min_max_transformer, f)
    
    print("Preprocessor saved!")

if __name__ == "__main__":
    main()



