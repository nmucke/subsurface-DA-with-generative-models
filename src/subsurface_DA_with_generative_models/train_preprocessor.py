import pdb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle

from subsurface_DA_with_generative_models.data_handling.xarray_data import XarrayDataset
from subsurface_DA_with_generative_models.preprocessor import MinMaxTransformer

FOLDER = "data/results32"
INPUT_VARS = ['Por', 'Perm', 'Pressure'] # Porosity, Permeability, Pressure + x, y, time encodings 
OUTPUT_VARS = ['Pressure']

INPUT_PREPROCESSOR_SAVE_PATH = 'trained_preprocessors/input_preprocessor_32.pkl'
OUTPUT_PREPROCESSOR_SAVE_PATH = 'trained_preprocessors/output_preprocessor_32.pkl'

def main():

    # Load data
    train_dataset = XarrayDataset(
        folder=FOLDER,
        input_vars=INPUT_VARS,
        output_vars=OUTPUT_VARS
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
    )

    # Set up preprocessor
    input_min_max_transformer = MinMaxTransformer(
        num_channels=6
    )
    output_min_max_transformer = MinMaxTransformer(
        num_channels=1
    )

    for i, (input_data, output_data) in enumerate(train_dataloader):
        
        input_data = input_data.view(-1, input_data.shape[2], input_data.shape[3], input_data.shape[4])
        output_data = output_data.view(-1, output_data.shape[2], output_data.shape[3], output_data.shape[4])

        # Fit the preprocessor
        input_min_max_transformer.fit_transform_in_batches(input_data)
        output_min_max_transformer.fit_transform_in_batches(output_data)
    
    print("Preprocessor fitted!")
    

    # Save the preprocessor
    with open(INPUT_PREPROCESSOR_SAVE_PATH, 'wb') as f:
        pickle.dump(input_min_max_transformer, f)
    
    with open(OUTPUT_PREPROCESSOR_SAVE_PATH, 'wb') as f:
        pickle.dump(output_min_max_transformer, f)
    
    print("Preprocessor saved!")

if __name__ == "__main__":
    main()



