import xarray as xr
import os
import glob
import numpy as np
import pandas as pd
import cftime
import pop_tools 
import dask.array # Explicitly import dask for clarity

# --- Configuration Variables ---
output_dir = '/Data/skd/scratch/innag3580/comp/'
output_filename = 'var_ensemble_mean.nc'
output_path = os.path.join(output_dir, output_filename)

path = '/Data/skd/scratch/innag3580/comp/averages/'
temp_files_pattern = 'TEMP_*.nc'
salt_files_pattern = 'SALT_*.nc'

# --- 1. Utility Functions ---

def standardise_time(ds):
    """Converts cftime.DatetimeNoLeap to numpy datetime64."""
    decoded = xr.decode_cf(ds, use_cftime=True)
    if isinstance(decoded.time.values[0], cftime._cftime.DatetimeNoLeap):
        time_as_datetime64 = np.array([pd.Timestamp(str(dt)).to_datetime64() for dt in decoded.time.values])
        ds['time'] = xr.DataArray(time_as_datetime64, dims='time')
    return ds

# --- 2. Mask Definition ---

grid_name = 'POP_gx1v7'
region_defs = {
    'SubpolarAtlantic': [
        {'match': {'REGION_MASK': [6]}, 'bounds': {'TLAT': [50.0, 65.0], 'TLONG': [200.0, 360.0]}}
    ],
    'LabradorSea': [
        {'match': {'REGION_MASK': [8]}, 'bounds': {'TLAT': [50.0, 65.0], 'TLONG': [260.0, 360.0]}}
    ]
}

mask = pop_tools.region_mask_3d(grid_name, region_defs=region_defs, mask_name='North Atlantic').sum('region')


# --- 3. Processing Function (Memory Optimized) ---

def calculate_ensemble_mean():
    
    # 1. Load and process Temperature (TEMP)
    temp_file_list = sorted(glob.glob(os.path.join(path, temp_files_pattern)))
    
    # Load with chunking, time standardization, and apply time slice filter first
    ds_temp = xr.open_mfdataset(
        temp_file_list, 
        chunks={'time': 120, 'z_t': 10}, # Added depth chunking for better parallelization
        preprocess=standardise_time
    ).sel(time=slice(None, '2025'))
    
    # Chain operations for efficiency: Convert, Mask, Resample Mean, Select Data Variable
    ds_temp_mean = ds_temp['TEMP'].astype('float32').where(mask == 1).resample(time='1Y').mean()
    
    # 2. Load and process Salinity (SALT)
    salt_file_list = sorted(glob.glob(os.path.join(path, salt_files_pattern)))
    
    ds_salt = xr.open_mfdataset(
        salt_file_list, 
        chunks={'time': 120, 'z_t': 10}, 
        preprocess=standardise_time
    ).sel(time=slice(None, '2025'))
    
    # Chain operations for efficiency: Convert, Mask, Resample Mean, Select Data Variable
    ds_salt_mean = ds_salt['SALT'].astype('float32').where(mask == 1).resample(time='1Y').mean()

    # 3. Merge and Compute
    # Merge the two DataArrays
    ds_ensemble = xr.Dataset({'TEMP': ds_temp_mean, 'SALT': ds_salt_mean})

    # Save Output
    os.makedirs(output_dir, exist_ok=True)
    
    # Use .compute() only when writing to disk to force Dask execution
    print(f"Saving ensemble mean to {output_path}...")
    ds_ensemble.to_netcdf(output_path, engine='netcdf4')
    
    print("Process complete.")

# --- Execute Script ---
if __name__ == "__main__":
    calculate_ensemble_mean()