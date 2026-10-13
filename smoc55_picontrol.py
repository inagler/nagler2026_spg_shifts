#!/usr/bin/env python3
# inagler 2026

"""
Compute the density-space Subpolar MOC index at 55°N (SMOC55) for the
CESM2 piControl run.

Adapted from smoc55_timeseries.py (LENS2 ensemble).

Key differences vs. smoc55_timeseries.py
-----------------------------------------
Variable names  : VVEL/TEMP/SALT (POP native)  →  vo/thetao/so (CMIP6)
vo units        : cm s⁻¹ (POP native)          →  m s⁻¹ (CMIP6)
                  → Sv conversion: × 1e-12 cm³/s  →  × 1e-6 m³/s
Grid metrics    : dz/DXU embedded in POP files  →  loaded from pop_tools
                  (pop_tools stores both in cm; converted to m here)
Ensemble        : 100 members                   →  single 1200-year run
lev coordinate  : cm in CESM2 CMIP6 output (same as POP native)

Outputs
-------
  /Data/skd/scratch/innag3580/comp/smoc55_picontrol/smoc55_picontrol.nc
"""

import gc
import glob
import os
import time as time_mod

import gsw
import numpy as np
import pop_tools
import xarray as xr

start_time = time_mod.time()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

VO_DIR     = '/Data/gfi/share/ModData/CMIP6/raw/ocean/piControl/CESM2/VO'
THETAO_DIR = '/Data/gfi/share/ModData/CMIP6/raw/ocean/piControl/CESM2/THETAO'
SO_DIR     = '/Data/gfi/share/ModData/CMIP6/raw/ocean/piControl/CESM2/SO'

OUTPUT_DIR = '/Data/skd/scratch/innag3580/comp/smoc55_picontrol'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Grid selection — same indices as smoc55_timeseries.py
# nlon roll of -100 moves the dateline discontinuity away from the Atlantic
# ---------------------------------------------------------------------------

SEL_NLAT = 340             # row nearest 55°N on POP_gx1v7
SEL_NLON = slice(198, 254) # North Atlantic nlon range after roll(-100)

# ---------------------------------------------------------------------------
# Sigma-2 density bins — identical to smoc55_timeseries.py
# ---------------------------------------------------------------------------

_anchors   = np.array([29.70, 31.50, 33.15, 34.75, 35.80,
                        36.38, 36.70, 36.89, 37.06, 37.13, 37.30])
_intervals = np.diff(_anchors) / 4
density_bins = np.append(
    np.concatenate([
        np.arange(_anchors[i], _anchors[i + 1], _intervals[i])
        for i in range(len(_intervals))
    ]),
    _anchors[-1],
)

# ---------------------------------------------------------------------------
# Grid metrics from pop_tools
# ---------------------------------------------------------------------------

print('Loading POP_gx1v7 grid metrics ...', flush=True)
grid = pop_tools.get_grid('POP_gx1v7')

# DXU: zonal U-point face width, cm → m.
# Roll by -100 to match the data roll applied below.
DXU_m   = np.roll(grid['DXU'].values * 1e-2, -100, axis=1)  # [nlat, nlon], m
dxu_sec = DXU_m[SEL_NLAT, SEL_NLON]                          # [nlon_sec], m

# dz: layer thickness, cm → m.
dz_m = grid['dz'].values * 1e-2   # [lev], m

del grid
gc.collect()
print(f'  dxu_sec shape: {dxu_sec.shape}   dz_m shape: {dz_m.shape}', flush=True)

# ---------------------------------------------------------------------------
# Load 55°N section data from CMIP6 piControl files
# ---------------------------------------------------------------------------

print('\nOpening piControl section datasets ...', flush=True)

vo_files     = sorted(glob.glob(os.path.join(VO_DIR,     'vo_Omon_CESM2_piControl_*.nc')))
thetao_files = sorted(glob.glob(os.path.join(THETAO_DIR, 'thetao_Omon_CESM2_piControl_*.nc')))
so_files     = sorted(glob.glob(os.path.join(SO_DIR,     'so_Omon_CESM2_piControl_*.nc')))

print(f'  vo files:     {len(vo_files)}')
print(f'  thetao files: {len(thetao_files)}')
print(f'  so files:     {len(so_files)}')

def open_section(files, var):
    ds = xr.open_mfdataset(files, combine='by_coords', use_cftime=True)
    ds = ds.roll(nlon=-100, roll_coords=True)
    ds = ds.isel(nlat=SEL_NLAT, nlon=SEL_NLON)
    return ds

ds_vo = open_section(vo_files,     'vo')
ds_t  = open_section(thetao_files, 'thetao')
ds_s  = open_section(so_files,     'so')

nt = len(ds_vo.time)
print(f'  Total timesteps: {nt}', flush=True)

# Load section arrays into memory — each is [nt, lev, nlon_sec] ≈ 190 MB
print('\nLoading section data into memory ...', flush=True)
vo_data     = ds_vo['vo'].values.astype(np.float64)      # m s⁻¹
thetao_data = ds_t['thetao'].values.astype(np.float64)   # °C
so_data     = ds_s['so'].values.astype(np.float64)       # PSU
time_coord  = ds_vo['time'].values

ds_vo.close(); ds_t.close(); ds_s.close()
gc.collect()
print(f'  vo shape: {vo_data.shape}', flush=True)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def calculate_dens2(thetao_ts, so_ts):
    """
    Compute sigma-2 at one timestep.
    Uses the same gsw approximation as smoc55_timeseries.py: practical
    salinity is passed where absolute salinity is expected.
    thetao_ts, so_ts : [lev, nlon_sec]
    """
    CT     = gsw.CT_from_pt(so_ts, thetao_ts)
    sigma2 = gsw.sigma2(so_ts, CT)
    return sigma2


def calculate_smoc55(vo_ts, sigma2_ts):
    """
    Compute density-space overturning streamfunction maximum at one timestep.

    vo_ts     : [lev, nlon_sec], m s⁻¹
    sigma2_ts : [lev, nlon_sec], kg m⁻³

    Returns SMOC55 in Sv.
    """
    # Flow rate per cell [lev, nlon_sec], m³ s⁻¹
    vo_clean  = np.where(np.isfinite(vo_ts), vo_ts, 0.0)
    flow_rate = (vo_clean
                 * dz_m    [:, np.newaxis]
                 * dxu_sec [np.newaxis, :])

    # Bin flow into sigma-2 density bins
    flow_by_density = np.zeros(len(density_bins))
    for k in range(len(density_bins) - 1):
        in_bin = (
            (sigma2_ts >= density_bins[k]) &
            (sigma2_ts <  density_bins[k + 1]) &
            np.isfinite(sigma2_ts)
        )
        flow_by_density[k] = flow_rate[in_bin].sum()

    # Cumulative integral from light→dense, reversed → dense→light
    overturning = np.cumsum(flow_by_density)[::-1]

    # m³ s⁻¹ → Sv  (1 Sv = 1e6 m³ s⁻¹)
    return float(np.max(overturning)) * 1e-6

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

print(f'\nComputing SMOC55 for {nt} timesteps ...', flush=True)
smoc55_values = np.full(nt, np.nan)

for i in range(nt):
    if i % 120 == 0:
        elapsed = time_mod.time() - start_time
        print(f'  step {i:5d}/{nt}  ({elapsed:.0f}s elapsed)', flush=True)

    vo_ts     = vo_data    [i].copy()
    thetao_ts = thetao_data[i].copy()
    so_ts     = so_data    [i].copy()

    # Mask CMIP6 fill values (1e+20)
    vo_ts    [~np.isfinite(vo_ts)]     = np.nan
    thetao_ts[~np.isfinite(thetao_ts)] = np.nan
    so_ts    [~np.isfinite(so_ts)]     = np.nan

    sigma2_ts        = calculate_dens2(thetao_ts, so_ts)
    smoc55_values[i] = calculate_smoc55(vo_ts, sigma2_ts)

del vo_data, thetao_data, so_data
gc.collect()

print(f'\nSMOC55 range: {np.nanmin(smoc55_values):.2f} – '
      f'{np.nanmax(smoc55_values):.2f} Sv', flush=True)

# ---------------------------------------------------------------------------
# Save output
# ---------------------------------------------------------------------------

out_file = os.path.join(OUTPUT_DIR, 'smoc55_picontrol.nc')
print(f'\nSaving → {out_file}', flush=True)

ds_out = xr.Dataset(
    {
        'smoc55_index': xr.DataArray(
            smoc55_values,
            dims=['time'],
            coords={'time': time_coord},
            attrs={
                'long_name'  : 'Subpolar MOC index at 55°N (density-space)',
                'units'      : 'Sv',
                'description': (
                    'Maximum of the density-space overturning streamfunction '
                    'at nlat=340 (≈55°N), binned into sigma-2 density classes '
                    'and integrated zonally (nlon 198–253 after −100 roll). '
                    'Computed from CESM2 piControl CMIP6 vo/thetao/so with '
                    'pop_tools POP_gx1v7 grid metrics (DXU, dz in m).'
                ),
            },
        )
    }
)
ds_out.to_netcdf(out_file)
ds_out.close()
print('Saved.', flush=True)

end_time = time_mod.time()
print(f'\nTotal time: {end_time - start_time:.0f}s '
      f'({(end_time - start_time) / 60:.1f} min)', flush=True)
