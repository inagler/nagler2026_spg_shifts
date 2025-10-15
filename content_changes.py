#!/usr/bin/env python3
# inagler 22/09/25

import os
import glob
import numpy as np
import xarray as xr
import pandas as pd
import cftime
import pop_tools

# --- Helper to standardise time ---
def standardise_time(ds):
    decoded = xr.decode_cf(ds, use_cftime=True)
    if isinstance(decoded.time.values[0], cftime._cftime.DatetimeNoLeap):
        time_as_datetime64 = np.array([pd.Timestamp(str(dt)).to_datetime64()
                                       for dt in decoded.time.values])
        ds['time'] = xr.DataArray(time_as_datetime64, dims='time')
    return ds

# --- Region mask ---
grid_name = 'POP_gx1v7'
region_defs = {'SubpolarAtlantic': [{'match': {'REGION_MASK': [6]}, 'bounds': {'TLAT': [50.0, 65.0], 'TLONG': [200.0, 360.0]}}],
               'LabradorSea': [{'match': {'REGION_MASK': [8]}, 'bounds': {'TLAT': [50.0, 65.0], 'TLONG': [260.0, 360.0]}}]}
mask = pop_tools.region_mask_3d(grid_name, region_defs=region_defs, mask_name='North Atlantic')
mask = mask.sum('region')

# --- Paths ---
path = '/Data/skd/scratch/innag3580/comp/averages/'
temp_files_pattern = 'TEMP_*.nc'
salt_files_pattern = 'SALT_*.nc'

# --- Load ---
temp_file_list = sorted(glob.glob(os.path.join(path, temp_files_pattern)))
ds_temp = xr.open_mfdataset(temp_file_list, chunks={'time': 120}, preprocess=standardise_time)
ds_temp['TEMP'] = ds_temp['TEMP'].astype('float32')
ds_temp = ds_temp.where(mask == 1).resample(time='1Y').mean()

salt_file_list = sorted(glob.glob(os.path.join(path, salt_files_pattern)))
ds_salt = xr.open_mfdataset(salt_file_list, chunks={'time': 120}, preprocess=standardise_time)
ds_salt['SALT'] = ds_salt['SALT'].astype('float32')
ds_salt = ds_salt.where(mask == 1).resample(time='1Y').mean()
ds_temp = ds_temp.sel(time=slice(None, '2025'))
ds_salt = ds_salt.sel(time=slice(None, '2025'))

# --- Unit conversions ---
ds_salt['dz'] = ds_temp.dz * 1e-2
ds_temp['UAREA'] = ds_temp.UAREA * 1e-4

# --- Constants ---
rho_sw, cp_sw, S_ref = 1026, 3990, 35

# --- Heat & freshwater content ---
# Compute heat content
heat_content = rho_sw * cp_sw * (ds_temp.dz * ds_temp.UAREA * ds_temp.TEMP).sum(dim=['nlat', 'nlon', 'z_t'])

# Freshwater content computation (add this line)
freshwater_content = ((S_ref - ds_salt.SALT) / S_ref * ds_salt.dz * ds_salt.UAREA).sum(dim=['nlat', 'nlon', 'z_t'])

# --- SMOC ---
output_dir = '/Data/skd/scratch/innag3580/comp/smoc/'
output = os.path.join(output_dir, 'smoc55_ensemble_mean.nc')
smoc55 = xr.open_dataarray(output).resample(time='1Y').mean().sel(time=slice(None, '2025'))

# --- Fractional anomalies ---
delta_heat_content = (heat_content - heat_content.sel(time=slice(None, '2025')).mean('time')) / heat_content.sel(time=slice(None, '2025')).mean('time')
delta_freshwater_content = (freshwater_content - freshwater_content.sel(time=slice(None, '2025')).mean('time')) / freshwater_content.sel(time=slice(None, '2025')).mean('time')
delta_smoc55 = (smoc55 - smoc55.sel(time=slice(None, '2025')).mean('time')) / smoc55.sel(time=slice(None, '2025')).mean('time')

# --- Save ---
xr.Dataset({
    'delta_heat_content': delta_heat_content,
    'delta_freshwater_content': delta_freshwater_content,
    'delta_smoc55': delta_smoc55
}).to_netcdf('/Data/skd/scratch/innag3580/comp/content_changes.nc')