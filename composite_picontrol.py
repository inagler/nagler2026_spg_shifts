#!/usr/bin/env python3
# inagler 2026

"""
Compute ±WINDOW_YR window composites of CESM2 piControl ocean fields around
strong and weak SMOC55 events detected by smoc55_picontrol.py.

Usage
-----
    python composite_picontrol.py <variable> <event_type>

    variable   : thetao | so | vo | uo | mlotst | zos
    event_type : strong | weak

Examples
--------
    python composite_picontrol.py thetao strong
    python composite_picontrol.py so weak
    python composite_picontrol.py mlotst strong

What this script does
---------------------
1. Reads event time indices from the CSV written by smoc55_picontrol.py.
2. Opens all piControl CMIP6 files for the requested variable lazily.
3. Computes the full-record long-term mean field (used for anomalies).
4. For each event, loads the ±WINDOW_YR monthly window from disk,
   subtracts the long-term mean, and accumulates into a running sum.
   Events too close to the record boundary are silently skipped.
5. Divides the accumulated sum by the number of valid events.
6. Saves the mean anomaly composite as a NetCDF file.

Outputs
-------
  /Data/skd/scratch/innag3580/comp/smoc55_picontrol/composites/
    smoc55_{event_type}_{variable}_composite.nc
    smoc55_{event_type}_{variable}_mean.nc       (long-term mean field)

Variables supported
-------------------
  thetao  : potential temperature, °C           [time, lev, nlat, nlon]
  so      : practical salinity, PSU              [time, lev, nlat, nlon]
  vo      : meridional velocity, m s⁻¹          [time, lev, nlat, nlon]
  uo      : zonal velocity, m s⁻¹               [time, lev, nlat, nlon]
  mlotst  : mixed layer depth (MLD), m          [time, nlat, nlon]
  zos     : sea surface height, m               [time, nlat, nlon]

Memory notes
------------
  3-D variables : composite accumulation array is [n_lag, lev, nlat, nlon]
                  = [121, 60, 384, 320] float32 ≈ 3.5 GB.
  2-D variables : [121, 384, 320] float32 ≈ 58 MB.
  Long-term mean: same as one time slice (e.g. [60, 384, 320] ≈ 57 MB).
  Only one variable and one event_type are held in memory at a time.
  The full 3-D record (≈42 GB) is never loaded — xarray reads only the
  time slices needed per event window and for the mean.
"""

import csv
import gc
import glob
import os
import sys
import time as time_mod

import numpy as np
import xarray as xr

start_time = time_mod.time()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PICONTROL_ROOT = '/Data/gfi/share/ModData/CMIP6/raw/ocean/piControl/CESM2'
SMOC55_DIR     = '/Data/skd/scratch/innag3580/comp/smoc55_picontrol'
OUTPUT_DIR     = os.path.join(SMOC55_DIR, 'composites')

WINDOW_YR   = 5          # half-width of the composite window in years
HALF_WINDOW = WINDOW_YR * 12   # 60 months

# CMIP6 directory and variable name for each supported variable.
# Directories use UPPERCASE on disk; filenames use lowercase.
VAR_INFO = {
    'thetao': {'dir': 'THETAO', 'units': 'degrees_C'},
    'so'    : {'dir': 'SO',     'units': 'PSU'},
    'vo'    : {'dir': 'VO',     'units': 'm s-1'},
    'uo'    : {'dir': 'UO',     'units': 'm s-1'},
    'mlotst': {'dir': 'MLOTST', 'units': 'm'},
    'zos'   : {'dir': 'ZOS',    'units': 'm'},
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

if len(sys.argv) != 3:
    print('Usage: python composite_picontrol.py <variable> <event_type>')
    print('  variable   : thetao | so | vo | uo | mlotst | zos')
    print('  event_type : strong | weak')
    sys.exit(1)

var        = sys.argv[1]
event_type = sys.argv[2]

if var not in VAR_INFO:
    print(f'Unknown variable "{var}". Choose from: {list(VAR_INFO)}')
    sys.exit(1)
if event_type not in ('strong', 'weak'):
    print(f'Unknown event_type "{event_type}". Choose: strong | weak')
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f'variable   : {var}', flush=True)
print(f'event_type : {event_type}', flush=True)
print(f'window     : ±{WINDOW_YR} yr ({2 * HALF_WINDOW + 1} monthly steps)', flush=True)

# ---------------------------------------------------------------------------
# Load event indices
# ---------------------------------------------------------------------------

event_csv = os.path.join(SMOC55_DIR, f'events_smoc55_{event_type}.csv')
if not os.path.isfile(event_csv):
    print(f'Event CSV not found: {event_csv}')
    print('Run smoc55_picontrol.py first.')
    sys.exit(1)

event_indices = []
with open(event_csv, newline='') as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        event_indices.append(int(row['time_index']))
event_indices = np.array(event_indices, dtype=int)

print(f'\nLoaded {len(event_indices)} {event_type} events from {event_csv}', flush=True)

# ---------------------------------------------------------------------------
# Open CMIP6 data files
# ---------------------------------------------------------------------------

data_dir = os.path.join(PICONTROL_ROOT, VAR_INFO[var]['dir'])
files    = sorted(glob.glob(os.path.join(data_dir, f'{var}_Omon_CESM2_piControl_*.nc')))

if not files:
    print(f'No files found in {data_dir}')
    sys.exit(1)

print(f'\nOpening {len(files)} files for {var} ...', flush=True)
ds = xr.open_mfdataset(files, combine='by_coords', use_cftime=True)
nt = len(ds.time)
print(f'  Total timesteps: {nt}', flush=True)

# ---------------------------------------------------------------------------
# Compute long-term mean (needed for anomaly composites)
# ---------------------------------------------------------------------------

print('\nComputing long-term mean (reads full record) ...', flush=True)
lt_mean = ds[var].mean('time').compute().values   # [lev, nlat, nlon] or [nlat, nlon]
print(f'  Long-term mean shape: {lt_mean.shape}', flush=True)

# Save the long-term mean field for reference / further analysis
mean_file = os.path.join(OUTPUT_DIR, f'smoc55_{event_type}_{var}_mean.nc')
ds[var].mean('time').compute().to_netcdf(mean_file)
print(f'  Long-term mean saved → {mean_file}', flush=True)

# ---------------------------------------------------------------------------
# Accumulate composite
# ---------------------------------------------------------------------------

print(f'\nAccumulating composite over events ...', flush=True)

composite_sum = None
n_valid       = 0
n_skip        = 0

for idx_event, t_idx in enumerate(event_indices):
    t_start = t_idx - HALF_WINDOW
    t_end   = t_idx + HALF_WINDOW + 1   # exclusive

    # Skip events whose window extends outside the record.
    if t_start < 0 or t_end > nt:
        n_skip += 1
        continue

    # Load this window from disk — xarray reads only these time slices.
    window = ds[var].isel(time=slice(t_start, t_end)).load().values  # [n_lag, ...]

    # Subtract long-term mean to get anomaly (broadcast over the lag axis).
    window_anom = window - lt_mean[np.newaxis, ...]

    # Accumulate
    if composite_sum is None:
        composite_sum = window_anom.astype(np.float32)
    else:
        composite_sum += window_anom.astype(np.float32)

    n_valid += 1

    if n_valid % 10 == 0 or n_valid == 1:
        elapsed = time_mod.time() - start_time
        print(f'  event {idx_event + 1:4d}/{len(event_indices)}  '
              f'valid: {n_valid}  skipped: {n_skip}  '
              f'elapsed: {elapsed:.0f}s', flush=True)

if n_valid == 0:
    print('ERROR: no valid events (all within WINDOW_YR of record boundary).')
    sys.exit(1)

print(f'\n  Done. valid={n_valid}, skipped={n_skip}', flush=True)

composite = composite_sum / n_valid   # mean anomaly composite

del composite_sum, lt_mean
gc.collect()

# ---------------------------------------------------------------------------
# Build lag coordinate and save output
# ---------------------------------------------------------------------------

lags_mo = np.arange(-HALF_WINDOW, HALF_WINDOW + 1)   # integer months
lags_yr = lags_mo / 12.0                              # fractional years

# Retrieve lat/lon 2D coordinates from the opened dataset (CMIP6 convention).
lat2d = ds['lat'].values   # [nlat, nlon], °N
lon2d = ds['lon'].values   # [nlat, nlon], °E
ds.close()

# Build output DataArray dimensions.
# For 3-D variables the composite has dims [lag, lev, nlat, nlon];
# for 2-D variables (mlotst, zos) it is [lag, nlat, nlon].
comp_dims  = ['lag'] + [d for d in (['lev', 'nlat', 'nlon'] if composite.ndim == 4 else ['nlat', 'nlon'])]
comp_coords = {
    'lag'      : ('lag', lags_yr,
                  {'long_name': 'lag relative to event', 'units': 'years'}),
    'latitude' : (['nlat', 'nlon'], lat2d,
                  {'long_name': 'latitude',  'units': 'degrees_north'}),
    'longitude': (['nlat', 'nlon'], lon2d,
                  {'long_name': 'longitude', 'units': 'degrees_east'}),
}

out_var   = f'{var}_anom'
long_name = (
    f'Window composite anomaly of {var} — '
    f'{event_type} SMOC55 events (±{WINDOW_YR} yr window). '
    f'Anomaly relative to piControl long-term mean.'
)

ds_out = xr.Dataset(
    {
        out_var: xr.DataArray(
            composite,
            dims=comp_dims,
            coords=comp_coords,
            attrs={
                'long_name'           : long_name,
                'units'               : VAR_INFO[var]['units'],
                'source_variable'     : var,
                'event_type'          : event_type,
                'n_events'            : n_valid,
                'n_events_skipped'    : n_skip,
                'composite_window_yr' : WINDOW_YR,
                'note': (
                    'Anomaly = window field − piControl long-term mean. '
                    'lag=0 is the event month; negative lags precede the event. '
                    'Events with insufficient boundary padding are excluded.'
                ),
            },
        )
    }
)

out_file = os.path.join(OUTPUT_DIR, f'smoc55_{event_type}_{var}_composite.nc')
ds_out.to_netcdf(out_file)
ds_out.close()

elapsed_total = time_mod.time() - start_time
print(f'\nSaved → {out_file}', flush=True)
print(f'  composite shape : {composite.shape}', flush=True)
print(f'  lag range       : {lags_yr[0]:.1f} … {lags_yr[-1]:.1f} yr', flush=True)
print(f'  n_events used   : {n_valid}', flush=True)
print(f'  Total time      : {elapsed_total:.0f}s ({elapsed_total / 60:.1f} min)', flush=True)
