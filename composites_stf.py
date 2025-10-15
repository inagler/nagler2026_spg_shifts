#!/usr/bin/env python3
# inagler 16/02/25

import os
import glob
import numpy as np
import xarray as xr
import csv
import pop_tools
import gsw

## INITIALISATION ###

# 'aa_hist', 'ghg_hist', 'early_hist', 'late_hist', 'above'
period = 'aa_hist'
path = '/Data/skd/scratch/innag3580/comp/composites/'

grid_name = 'POP_gx1v7'
region_defs = {
    'SubpolarAtlantic':[
        {'match': {'REGION_MASK': [6]}, 'bounds': {'TLAT': [10.0, 65.0], 'TLONG': [260.0, 360.0]}}   
    ],
    'LabradorSea': [
        {'match': {'REGION_MASK': [8]}, 'bounds': {'TLAT': [10.0, 65.0], 'TLONG': [260.0, 360.0]}}]}
mask = pop_tools.region_mask_3d(grid_name, region_defs=region_defs, mask_name='North Atlantic')
mask = mask.sum('region').roll(nlon=-100)   

if period == 'ghg_hist':
    #ds_temp = xr.open_dataset(path + 'Below_combined_TEMP_3.0_40_20.nc').where(mask == 1)
    #ds_salt = xr.open_dataset(path + 'Below_combined_SALT_3.0_40_20.nc').where(mask == 1)
    ds_vvel = xr.open_dataset(path + 'Below_combined_VVEL_3.0_40_20.nc').where(mask == 1)
elif period == 'aa_hist':
    #ds_temp = xr.open_dataset(path + 'Below_combined_TEMP_2.0_40_20_aa_hist.nc').where(mask == 1)
    #ds_salt = xr.open_dataset(path + 'Below_combined_SALT_2.0_40_20_aa_hist.nc').where(mask == 1)
    ds_vvel = xr.open_dataset(path + 'Below_combined_VVEL_2.0_40_20_aa_hist.nc').where(mask == 1)
elif period == 'above':
    #ds_temp = xr.open_dataset(path + 'Above_combined_TEMP_3.0_40_20.nc').where(mask == 1)
    #ds_salt = xr.open_dataset(path + 'Above_combined_SALT_3.0_40_20.nc').where(mask == 1)
    ds_vvel = xr.open_dataset(path + 'Above_combined_VVEL_3.0_40_20.nc').where(mask == 1)
    
# for deprecated dz and DXU    
ds_vvel_original = xr.open_dataset('/Data/gfi/share/ModData/CESM2_LENS2/ocean/monthly/VVEL/b.e21.BHISTcmip6.f09_g17.LE2-1001.001.pop.h.VVEL.185001-185912.nc').roll(nlon=-100).isel(time=slice(0,3)).where(mask == 1) 

cell_thickness = ds_vvel_original.dz
cell_width = ds_vvel_original.DXU
    
replacement_value = 0
ds_vvel['ULAT'] = xr.where(
    np.logical_or(np.isnan(ds_vvel['ULAT']), np.isinf(ds_vvel['ULAT']) | np.ma.getmask(ds_vvel['ULAT'])),
    replacement_value,
    ds_vvel['ULAT'])
    
CT = gsw.conversions.CT_from_pt(ds_salt['SALT'], ds_temp['TEMP'])
sigma2 = gsw.density.sigma2(ds_salt['SALT'], CT)
sigma2 = xr.DataArray(sigma2, name='DENS2', dims=ds_temp['TEMP'].dims, coords=ds_temp['TEMP'].coords)

ds_dens = ds_temp
ds_dens = ds_dens.drop_vars('TEMP')
ds_dens['DENS2'] = sigma2

sorted_densities = np.array([34.93614813, 34.9578122 , 35.00964641, 35.1183501 , 35.2305248 ,
                           35.3365047 , 35.42915185, 35.50959735, 35.57460779, 35.62762557,
                           35.67111872, 35.70908475, 35.74091352, 35.77153398, 35.79984782,
                           35.8268007 , 35.85105131, 35.8752147 , 35.89779526, 35.9154073 ,
                           35.93675417, 35.95600783, 35.97367325, 35.99304331, 36.00893165,
                           36.02535853, 36.04645767, 36.06327999, 36.08619115, 36.11248339,
                           36.13816436, 36.16728558, 36.20238324, 36.24309049, 36.28594763,
                           36.33856534, 36.39389841, 36.45772171, 36.52552381, 36.59612625,
                           36.66504614, 36.7238107 , 36.77042751, 36.8070972 , 36.83360747,
                           36.85462751, 36.87082049, 36.87123488, 36.87847967, 36.88012872,
                           36.88032291, 36.88046028, 36.8835508 , 36.89367036, 36.90081019,
                           36.90642509, 36.90981405, 36.91219296, 36.9138063 , 36.91554088])

print('initialised')

### COMPUTATION ###

def density_MOC(ds_vvel, ds_dens, sorted_densities):
    
    cell_thickness = ds_vvel_original.dz
    cell_width = ds_vvel_original.DXU
    density_bins = sorted_densities
    
     # initialise  3D array for density overturning: [density_bins, latitudes, time_steps]
    density_overturning = np.zeros((len(density_bins)-1,  len(ds_vvel.nlat), len(ds_vvel.time)))

    # compute meridional flow rate
    velocity = ds_vvel['VVEL']
    flow_rate = velocity * cell_thickness * cell_width

    # prepare computation for overturning per time step per latitude
    for time_step in range(len(ds_vvel.time)):
        for lat in range(len(ds_vvel.nlat)):
            flow_rate_field = flow_rate.isel(time=time_step, nlat=lat)
            density_field = ds_dens.DENS2.isel(time=time_step, nlat=lat)
            flow_rate_by_density = np.zeros(len(density_bins) - 1)
            
            # find meridional flow rate for each density bin and integrate zonally
            for bin_index in range(len(density_bins) - 1):
                in_bin = (density_field >= density_bins[bin_index]) & (density_field < density_bins[bin_index + 1])
                flow_rate_by_density[bin_index] = flow_rate_field.where(in_bin).sum()
            
            # compute density overturning, integrate from high to low density
            density_overturning[:, lat, time_step] = np.cumsum(flow_rate_by_density)

    # create xarray dataset
    ds_smoc = xr.Dataset(
        {'sMOC': (['dens2', 'nlat', 'time'], density_overturning)},
        coords={
            'dens2': (['dens2'], 0.5 * (sorted_densities[:-1] + sorted_densities[1:])),
            'nlat': (['nlat'], ds_vvel.nlat.values),
            'time': (['time'], ds_vvel['time'].values)})
    ds_smoc = ds_smoc.assign_coords(TLAT=ds_vvel_original.TLAT.isel(nlon=0).squeeze())
    replacement_value = 0
    ds_smoc['TLAT'] = xr.where(
        np.logical_or(np.isnan(ds_smoc['TLAT']), np.isinf(ds_smoc['TLAT']) | np.ma.getmask(ds_smoc['TLAT'])),
        replacement_value,
        ds_smoc['TLAT'])
    
    return ds_smoc

ds_smoc = density_MOC(ds_vvel, ds_dens, sorted_densities)

da_dmoc = (ds_vvel.VVEL * ds_vvel_original.dz * ds_vvel_original.DXU).sum(dim='nlon').cumsum(dim='z_t')
ds_dmoc = xr.Dataset(
    {'dMOC': (['time', 'z_t', 'nlat'], da_dmoc.data)  },
    coords={'time': (['time'], ds_vvel['time'].values),  
            'z_t': (['z_t'], ds_vvel_original.z_t.values),   
            'nlat': (['nlat'], ds_vvel.nlat.values), 
            'TLAT': (['TLAT'], ds_smoc.TLAT.values),
    })
ds_dmoc['TLAT'] = xr.where(
        np.logical_or(np.isnan(ds_dmoc['TLAT']), np.isinf(ds_dmoc['TLAT']) | np.ma.getmask(ds_dmoc['TLAT'])),
        replacement_value,
        ds_dmoc['TLAT'])

da_bsf = (ds_vvel.VVEL * ds_vvel_original.dz * ds_vvel_original.DXU).sum(dim='z_t').cumsum(dim='nlon')
ds_bsf = xr.Dataset(
    {'BSF': (['time', 'nlat', 'nlon'], da_bsf.data)  },
    coords={'time': (['time'], ds_vvel['time'].values),      
            'nlat': (['nlat'], ds_vvel.nlat.values), 
            'nlon': (['nlon'], ds_vvel.nlon.values), 
            'ULAT': (['ULAT'], ds_smoc.ULAT.values),
            'ULONG': (['ULONG'], ds_smoc.ULONG.values)
    })

print('stream functions computed')

if period == 'ghg_hist':
    ds_smoc.to_netcdf(os.path.join(path, f'Below_combined_smoc_3.0_40_20_ghg_hist.nc'))
    ds_dmoc.to_netcdf(os.path.join(path, f'Below_combined_dmoc_3.0_40_20_ghg_hist.nc'))
    ds_bsf.to_netcdf(os.path.join(path, f'Below_combined_bsf_3.0_40_20_ghg_hist.nc'))
if period == 'aa_hist':
    ds_smoc.to_netcdf(os.path.join(path, f'Below_combined_smoc_2.0_40_20_aa_hist.nc'))
    ds_dmoc.to_netcdf(os.path.join(path, f'Below_combined_dmoc_2.0_40_20_aa_hist.nc'))
    ds_bsf.to_netcdf(os.path.join(path, f'Below_combined_bsf_2.0_40_20_aa_hist.nc'))
elif period == 'above':
    ds_smoc.to_netcdf(os.path.join(path, f'Above_combined_smoc_3.0_40_20.nc'))
    ds_dmoc.to_netcdf(os.path.join(path, f'Above_combined_dmoc_3.0_40_20.nc'))
    ds_bsf.to_netcdf(os.path.join(path, f'Above_combined_bsf_3.0_40_20.nc'))

print('das saved')


