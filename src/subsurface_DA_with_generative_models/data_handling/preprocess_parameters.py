
import xarray as xr
import pandas as pd
import os
import pdb

def process_realization_data(realization):
    #realization = xr.open_dataset(f'.nc')
    #Dataset treatment
    # Multiply the Permeability values by the factor to have mD
    factor = 1.01324997e12
    realization['Permeability'] = realization['Permeability'] * factor
    # Set the minimum value of Permeability to 0.001 mD
    realization['Permeability'] = realization['Permeability'].where(realization['Permeability'] >= 0.001, 0.001)

    #Set the minimum value of Porosity to 0.01
    realization['Porosity'] = realization['Porosity'].where(realization['Porosity'] >= 0.01, 0.01)


    #realization = realization.assign_coords(Realization=(["Realization"], [job_number]))
    
    return realization

def main():

    sizes = ['8x8x1', '16x16x1', '32x32x1', '64x64x1', '128x128x1', '256x256x1']

    for size in sizes:
        for i in range(1,5936):
            data = xr.open_dataset(f'data/parameters/2Dresampled_{size}_realization_{i}.nc') 

            processed_data = process_realization_data(data)

            # Save the preprocessor
            processed_data.to_netcdf(f'data/parameters_processed/{size}_{i-1}.nc')

        print(f"data processsed for {size}!")


if __name__ == "__main__":
    main()

