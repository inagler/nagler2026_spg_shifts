#!/home/innag3580/.conda/envs/movie/bin/python
"""
compute_bsf_composites.py
Compute BSF composite from VVEL composites for each of four shift categories.

BSF = (VVEL * dz * DXU).sum('z_t').cumsum('nlon')
then roll(nlon=-100) and mask to TLAT 10–65°N.

dz and DXU are read once from the reference CESM2-LE VVEL file.

Output: one NetCDF per category at
    /Data/skd/scratch/innag3580/comp/revision/composites/raw/{category}_BSF.nc
with dimensions (shift, time, nlat, nlon).

Usage:
    ~/.conda/envs/movie/bin/python compute_bsf_composites.py
"""
import os
import numpy as np
import sys
import xarray as xr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
COMP_DIR = '/Data/skd/scratch/innag3580/comp/revision/composites/raw/'
REF_FILE = (
    '/Data/gfi/share/ModData/CESM2_LENS2/ocean/monthly/VVEL/'
    'b.e21.BHISTcmip6.f09_g17.LE2-1001.001.pop.h.VVEL.185001-185912.nc'
)

CATEGORIES = [
    'strengthening_early',
    'strengthening_late',
    'weakening_early',
    'weakening_late',
]


def main():
    sys.stdout.reconfigure(line_buffering=True)

    # --- Startup banner ---
    print('=' * 60)
    print('compute_bsf_composites.py  —  BSF composites from VVEL')
    print('=' * 60)
    print(f'  comp dir    : {COMP_DIR}')
    print(f'  ref file    : {REF_FILE}')
    print()
    print('  categories  :')
    for _cat in CATEGORIES:
        print(f'    {_cat}')
    print()
    print(f'  first load  : {CATEGORIES[0]}')
    print('-' * 60)
    print()

    # ---------------------------------------------------------------------------
    # Read dz and DXU once from reference file
    # dz: (z_t,) in cm
    # DXU: (nlat, nlon) in cm
    # ---------------------------------------------------------------------------
    print('Loading dz and DXU from reference file ...')
    ref = xr.open_dataset(REF_FILE)
    dz  = ref['dz']    # (z_t,)
    DXU = ref['DXU']   # (nlat, nlon)
    print(f'  dz shape: {dz.shape},  DXU shape: {DXU.shape}')

    # ---------------------------------------------------------------------------
    # Process each category
    # ---------------------------------------------------------------------------
    for cat in CATEGORIES:
        in_path  = os.path.join(COMP_DIR, f'{cat}_VVEL.nc')
        out_path = os.path.join(COMP_DIR, f'{cat}_BSF.nc')

        print(f'\n[{cat}]')
        print(f'  Reading {in_path}')
        ds_vvel = xr.open_dataset(in_path)

        n_shifts = ds_vvel.dims['shift']
        n_time   = ds_vvel.dims['time']
        TLAT     = ds_vvel['TLAT']   # (nlat, nlon)
        lat_mask = (TLAT >= 10) & (TLAT <= 65)

        print(f'  n_shifts = {n_shifts},  n_time = {n_time}')
        print(f'  Computing BSF per shift ...')

        bsf_list = []

        for s in range(n_shifts):
            if s % 10 == 0:
                print(f'    shift {s}/{n_shifts}')

            # Load one shift window: (time, z_t, nlat, nlon)
            vvel_s = ds_vvel['VVEL'].isel(shift=s).load()

            # BSF = integral of v * dz * dx  (cm³/s)
            # dz broadcast over (time, nlat, nlon), DXU broadcast over (time, z_t)
            bsf_s = (vvel_s * dz * DXU).sum('z_t').cumsum('nlon')

            # Roll -100 to close the streamfunction correctly, then roll back
            # so the output is on the original unrolled grid
            bsf_s = bsf_s.roll(nlon=-100, roll_coords=False)
            bsf_s = bsf_s.where(lat_mask)
            bsf_s = bsf_s.roll(nlon=+100, roll_coords=False)

            bsf_list.append(bsf_s)

        # Concatenate along shift dimension
        print(f'  Concatenating {n_shifts} shifts ...')
        bsf_out = xr.concat(bsf_list, dim=ds_vvel['shift'])   # (shift, time, nlat, nlon)
        bsf_out.name = 'BSF'
        bsf_out.attrs['long_name'] = 'Barotropic streamfunction'
        bsf_out.attrs['units']     = 'cm3 s-1'
        bsf_out.attrs['note']      = 'cumsum computed after roll(-100), rolled back to original grid; mask TLAT 10-65N'

        ds_out = bsf_out.to_dataset()

        # Carry over ULAT/ULONG from input for plotting reference
        for coord in ['ULAT', 'ULONG', 'TLAT', 'TLONG']:
            if coord in ds_vvel.coords:
                ds_out[coord] = ds_vvel[coord]

        print(f'  Writing {out_path}')
        ds_out.to_netcdf(out_path)
        print(f'  Done.')

        ds_vvel.close()

    print('\nAll categories complete.')


if __name__ == '__main__':
    main()
