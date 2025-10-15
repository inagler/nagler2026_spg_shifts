#!/usr/bin/env python3
# inagler 21/11/24

import psutil
import time

import os
import gc
import glob

import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
import dask

import cftime
import pop_tools  
import gsw

start_time = time.time()

'''

# compute density bins
numbers = np.array([29.70, 31.50, 33.15, 34.75, 35.80, 36.38, 36.70, 36.89, 37.06, 37.13, 37.30])
intervals = np.diff(numbers) / 4
result = np.concatenate([np.arange(numbers[i], numbers[i+1], intervals[i]) for i in range(len(intervals))])
density_bins = np.append(result, numbers[-1])

# define paths
vvel_dir = '/Data/gfi/share/ModData/CESM2_LENS2/ocean/monthly/VVEL'
temp_dir = '/Data/gfi/share/ModData/CESM2_LENS2/ocean/monthly/TEMP'
salt_dir = '/Data/gfi/share/ModData/CESM2_LENS2/ocean/monthly/SALT'
base_path = '/Data/gfi/share/ModData/CESM2_LENS2/ocean/monthly/'
output_dir = '/Data/skd/scratch/innag3580/comp/smoc/'

# define ensemble members
ensemble_members = [
    "1001.001", "1011.001", "1021.002", "1031.002", "1041.003", "1051.003", "1061.004", "1071.004", "1081.005", "1091.005",
    "1101.006", "1111.006", "1121.007", "1131.007", "1141.008", "1151.008", "1161.009", "1171.009", "1181.010", "1191.010",
    "1231.001", "1231.002", "1231.003", "1231.004", "1231.005", "1231.006", "1231.007", "1231.008", "1231.009", "1231.010",
    "1231.011", "1231.012", "1231.013", "1231.014", "1231.015", "1231.016", "1231.017", "1231.018", "1231.019", "1231.020",
    "1251.001", "1251.002", "1251.003", "1251.004", "1251.005", "1251.006", "1251.007", "1251.008", "1251.009", "1251.010",
    "1251.011", "1251.012", "1251.013", "1251.014", "1251.015", "1251.016", "1251.017", "1251.018", "1251.019", "1251.020",
    "1281.001", "1281.002", "1281.003", "1281.004", "1281.005", "1281.006", "1281.007", "1281.008", "1281.009", "1281.010",
    "1281.011", "1281.012", "1281.013", "1281.014", "1281.015", "1281.016", "1281.017", "1281.018", "1281.019", "1281.020",
    "1301.001", "1301.002", "1301.003", "1301.004", "1301.005", "1301.006", "1301.007", "1301.008", "1301.009", "1301.010",
    "1301.011", "1301.012", "1301.013", "1301.014", "1301.015", "1301.016", "1301.017", "1301.018", "1301.019", "1301.020"
]

time_periods = ['185001-185912', '186001-186912', '187001-187912', '188001-188912', '189001-189912', '190001-190912', '191001-191912', '192001-192912', '193001-193912', '194001-194912', '195001-195912', '196001-196912', '197001-197912', '198001-198912', '199001-199912', '200001-200912', '201001-201412', '201501-202412', '202501-203412', '203501-204412', '204501-205412', '205501-206412', '206501-207412', '207501-208412', '208501-209412', '209501-210012']

# select latitude
sel_nlat = 340
sel_nlon = slice(198, 254)

def standardise_time(ds):
    ds['time'] = xr.decode_cf(ds, use_cftime=True).time
    if isinstance(ds.time.values[0], cftime._cftime.DatetimeNoLeap):
        time_as_datetime64 = np.array([pd.Timestamp(str(dt)).to_datetime64() for dt in ds.time.values])
        ds['time'] = xr.DataArray(time_as_datetime64, dims='time')
    return ds

def load_dataset(file):
    try:
        #ds = xr.open_dataset(file, chunks='auto') 
        ds = xr.open_dataset(file) 
        ds = ds.roll(nlon=-100)
        ds = ds.isel(nlat=sel_nlat, nlon=sel_nlon)
    except (RuntimeError, IndexError) as e:
        print(f"Loading failed for {file}: {str(e)}")
        return None
    return ds

def calculate_dens2(temp_ds, salt_ds):
    CT = gsw.conversions.CT_from_pt(salt_ds['SALT'], temp_ds['TEMP'])
    sigma2 = gsw.density.sigma2(salt_ds['SALT'], CT)
    sigma2 = sigma2.rename('SIGMA_2')
    return sigma2

def calculate_smoc55(ds_vvel, da_dens):
    
    max_overturning_series = []
    cell_thickness = ds_vvel['dz']
    cell_width = ds_vvel['DXU']
    
    for time_step in range(len(ds_vvel.time)):
        try:
            # compute meridional flow rate for the specified latitude
            velocity = ds_vvel['VVEL'].isel(time=time_step)
            flow_rate = velocity * cell_thickness * cell_width

            # compute meridional flow rate and for each density bin and integrate zonally
            density_at_time = da_dens.isel(time=time_step)
            flow_rate_by_density = np.zeros(len(density_bins))
            for bin_index in range(len(density_bins) - 1):
                in_bin = (density_at_time >= density_bins[bin_index]) & (density_at_time < density_bins[bin_index + 1])
                flow_rate_by_density[bin_index] = flow_rate.where(in_bin).sum()

            # compute density overturning, reverse to integrate from high to low density
            density_overturning = np.cumsum(flow_rate_by_density)[::-1]
            max_overturning = np.max(density_overturning)
            max_overturning_series.append(max_overturning)
            
        except IndexError as e:
            print(f"Error occurred at time step: {time_step}")
            raise e
    max_overturning_dataarray = xr.DataArray(max_overturning_series, dims=["time"], coords={"time": ds_vvel['time']})
    return max_overturning_dataarray  * 1e-12

def log_error(member_id, time_period, log_file='smoc55_error_log.txt'):
    with open(log_file, 'a') as f:
        f.write(f"{member_id} and {time_period}\n")
        
def log_memory_usage(log_file='memory_log.txt'):
    memory_info = psutil.virtual_memory()
    used_memory = memory_info.used / (1024 ** 2)  # Convert bytes to megabytes
    with open(log_file, 'a') as f:
        f.write(f"Memory Used: {used_memory:.2f} MB\n")
    return used_memory
        
def process_files(member_id, time_period):
    print(f'Starting processing for member: {member_id}, time period: {time_period}')

    try:
        vvel_pattern = os.path.join(vvel_dir, f'*{member_id}*{time_period}*.nc')
        temp_pattern = os.path.join(temp_dir, f'*{member_id}*{time_period}*.nc')
        salt_pattern = os.path.join(salt_dir, f'*{member_id}*{time_period}*.nc')

        temp_files = glob.glob(temp_pattern)
        salt_files = glob.glob(salt_pattern)
        vvel_files = glob.glob(vvel_pattern)

        if not temp_files or not salt_files or not vvel_files:
            print(f'Files not found for member {member_id} and period {time_period}.')
            return

        ds_temp = load_dataset(temp_files[0])
        if ds_temp is None: return
        print('ds_temp loaded')
        log_memory_usage()
        
        ds_salt = load_dataset(salt_files[0])
        if ds_salt is None: return
        print('ds_salt loaded')
        log_memory_usage()
        
        ds_vvel = load_dataset(vvel_files[0])
        if ds_vvel is None: return
        print('ds_vvel loaded')
        log_memory_usage()

        da_dens = calculate_dens2(ds_temp, ds_salt)
        da_dens.astype('float32')
        print('da_dens computed')
        log_memory_usage()

        da_smoc55 = calculate_smoc55(ds_vvel, da_dens)
        print('smoc55 computed')
        da_smoc55.to_netcdf(os.path.join(output_dir, f'smoc55_member_{member_id}_{time_period}.nc'))
        log_memory_usage()
        print(f'{member_id} - {time_period} saved')

        del ds_temp, ds_salt, ds_vvel, da_dens, da_smoc55
        gc.collect()

    except IndexError as e:
        print(f"Error occurred for member: {member_id}, time period: {time_period}")
        log_error(member_id, time_period)  # Assuming log_error_member is defined

# Using Dask to parallelize the processing
tasks = []
for member_id in ensemble_members:
    for time_period in time_periods:
        task = dask.delayed(process_files)(member_id, time_period)
        tasks.append(task)



# Compute all the delayed tasks in parallel
dask.compute(*tasks)

print('')
print('single file computation complete')
print('')
print('starting concatenating')
print('')


for member_id in ensemble_members:
    
    pattern = f'*{member_id}_*.nc'
    file_list = glob.glob(output_dir + pattern)
    file_list.sort()
    
    datasets = []

    # Load, standardize, and store each dataset
    for file_path in file_list:
        ds = xr.open_dataset(file_path)
        ds = standardise_time(ds)  # Standardise the dataset
        datasets.append(ds)
    
    ds_concat =xr.concat(datasets, dim='time')
    ds_concat.to_netcdf(os.path.join(output_dir, f'smoc55_member_{member_id}.nc'))
    
    for file_path in file_list:
        os.remove(file_path)

    print(member_id, ' saved')
    
files = 'smoc55_member_*.nc'
ensemble_mean = xr.open_mfdataset(output_dir+files, concat_dim='new_dim', combine='nested').mean(dim='new_dim')
ensemble_mean.to_netcdf(os.path.join(output_dir, f'smoc55_ensemble_mean.nc'))
print('ensemble mean saved')

'''
# messy now because i keep adding to this code - can be cleaner
output_dir = '/Data/skd/scratch/innag3580/comp/smoc/'
files = [f for f in os.listdir(output_dir) if f.endswith('.nc') and f != 'smoc55_ensemble_mean.nc']
datasets = [xr.open_dataarray(os.path.join(output_dir, f)) for f in files]
######################

stacked = xr.concat(datasets, dim='member')
std = stacked.std(dim='member')
std.to_netcdf(os.path.join(output_dir, f'smoc55_ensemble_std.nc'))
print('ensemble std saved')

end_time = time.time()
duration = end_time - start_time
print(f"Execution time: {duration} seconds")