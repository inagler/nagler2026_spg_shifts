#!/usr/bin/env python3
# inagler 28/11/24

import time
import os
import sys
import glob
import numpy as np
import pandas as pd
import cftime
import xarray as xr
import dask
from dask import delayed, compute

def standardise_time(ds):
    ds['time'] = xr.decode_cf(ds, use_cftime=True).time
    if isinstance(ds.time.values[0], cftime._cftime.DatetimeNoLeap):
        time_as_datetime64 = np.array([pd.Timestamp(str(dt)).to_datetime64() for dt in ds.time.values])
        ds['time'] = xr.DataArray(time_as_datetime64, dims='time')
    return ds

def process_time_period(time_period, files, var, output_dir):
    try:
        mean_ds = xr.open_mfdataset(files, preprocess=standardise_time, concat_dim='new_dim', combine='nested').mean(dim='new_dim')
        output_file = os.path.join(output_dir, f'{var}_mean_{time_period}.nc')
        mean_ds.to_netcdf(output_file)
        print(f'mean for {time_period} saved')
    except Exception as e:
        print(f"Error processing: {e}")
        

def main():

    start_time = time.time()
    
    if len(sys.argv) < 2:
        print("Usage: python your_script.py <var>")
        sys.exit(1)

    var = sys.argv[1]
    
    global base_dir, output_dir

    # Base directory containing the variable folders
    base_dir = '/Data/gfi/share/ModData/CESM2_LENS2/ocean/monthly'
    output_dir = '/Data/skd/scratch/innag3580/comp/averages'

    var_dir = os.path.join(base_dir, var)
    file_paths = sorted(glob.glob(os.path.join(var_dir, '*.nc')))

    # Group files by their time periods
    time_periods = {}
    for file_path in file_paths:
        time_period = file_path.split('.')[-2]
        time_periods.setdefault(time_period, []).append(file_path)

    tasks = [delayed(process_time_period)(time_period, files, var, output_dir) for time_period, files in time_periods.items()]
    dask.compute(*tasks)

    print('process complete')
    end_time = time.time()
    duration = end_time - start_time
    print(f"Execution time: {duration} seconds")

if __name__ == "__main__":
    main()