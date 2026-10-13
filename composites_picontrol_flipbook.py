#!/usr/bin/env python3
# inagler 22/08/26

"""
Compute piControl composites (BSF, thetao, so) around 2-sigma Below events
detected in smoc55_picontrol_changepoint.ipynb, then produce a flip-book
figure per config in the same format as Plot_decrease_aa.ipynb.

Event years are re-detected here from smoc55_picontrol.nc using the same
algorithm and parameters (THRESHOLD=2.0, GAP=5) as the notebook — no
external CSV is needed and the script is fully standalone.

Two configs:
  P1_len=40 / P2_len=20  ->  60-year composite window,  3 events
  P1_len=20 / P2_len=20  ->  40-year composite window, 13 events

Composite column windows follow the same proportional placement as the
historical (aa_hist) flip-book — 10-year slices centred +-10 yr on the
shift boundary (P1_len):
  Pre-Shift  :  years P1-15 to P1-5
  Shift      :  years P1-5  to P1+5
  Post-Shift :  years P1+5  to P1+15

For P1=40 this gives (25-35), (35-45), (45-55) — matching Plot_decrease_aa.ipynb.
For P1=20 this gives  (5-15), (15-25), (25-35).

Anomalies are computed relative to the first P1_len years (P1 baseline),
matching the normalisation in Plot_decrease_aa.ipynb.

BSF is computed from the composite annual-mean vo field following composites_stf.py:
  BSF = (vo * dz * DXU).sum(lev).cumsum(nlon)   [Sv, x1e-6]
with dz and DXU from pop_tools POP_gx1v7 (cm -> m), matching CMIP6 vo in m/s.
The cumsum starts at nlon=0 (Pacific). Plotting uses anomalies, so the
integration-constant offset cancels; no longitude roll is needed.

Outputs per config (written to OUTPUT_DIR):
  Below_combined_bsf_2.0_{P1}_{P2}_picontrol.nc
  Below_combined_thetao_2.0_{P1}_{P2}_picontrol.nc
  Below_combined_so_2.0_{P1}_{P2}_picontrol.nc
  Figure_Decrease_picontrol_{P1}_{P2}_Flipbook.pdf
"""

import os
import gc
import glob
import time as time_mod

import numpy as np
import xarray as xr
import pop_tools
import gsw

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cmocean.cm as cmo
import cartopy
import cartopy.crs as ccrs

start_time = time_mod.time()


### PATHS ###

PICONTROL_ROOT = '/Data/gfi/share/ModData/CMIP6/raw/ocean/piControl/CESM2'
SMOC55_FILE    = '/Data/skd/scratch/innag3580/comp/smoc55_picontrol/smoc55_picontrol.nc'
OUTPUT_DIR     = '/Data/skd/scratch/innag3580/comp/smoc55_picontrol/composites_flipbook'
os.makedirs(OUTPUT_DIR, exist_ok=True)


### FIGURE STYLE  (AMS column widths, 11 pt font — same as Plot_decrease_aa.ipynb) ###

one_col = 3.167   # 19 picas
two_col = 6.5     # 39 picas
plt.rcParams.update({
    'figure.figsize': (two_col, two_col * 0.6),
    'font.size': 11,
    'axes.titlesize': 11,
    'savefig.dpi': 300,
})
nature_coast = cartopy.feature.NaturalEarthFeature(
    'physical', 'land', '50m',
    linewidth=0.3, edgecolor='black', facecolor='whitesmoke')


### DETECTION PARAMETERS  (identical to smoc55_picontrol_changepoint.ipynb) ###

THRESHOLD = 2.0
GAP       = 5    # years — finalized gap value from LENS2 analysis (plotting CSVs: _5)

CONFIGS = [
    dict(P1_len=40, P2_len=20),
    dict(P1_len=20, P2_len=20),
]


### GRID METRICS AND MASKS ###

print('Loading POP_gx1v7 grid metrics and masks ...', flush=True)

grid  = pop_tools.get_grid('POP_gx1v7')

# Layer thickness and zonal U-face width (cm -> m) for BSF computation
dz_m  = grid['dz'].values  * 1e-2    # [lev], m
DXU_m = grid['DXU'].values * 1e-2    # [nlat, nlon], m

# 2-D lat / lon coordinates (native POP order, no roll) for plotting
TLAT  = grid['TLAT'].values           # T-grid [nlat, nlon]
TLONG = grid['TLONG'].values
ULAT  = grid['ULAT'].values           # U-grid [nlat, nlon]
ULONG = grid['ULONG'].values

del grid
gc.collect()

grid_name = 'POP_gx1v7'

# Broad North Atlantic mask — for climatological contours on T/S panels
region_defs_na = {
    'SubpolarAtlantic': [
        {'match': {'REGION_MASK': [6]},
         'bounds': {'TLAT': [10.0, 80.0], 'TLONG': [260.0, 360.0]}}
    ],
    'LabradorSea': [
        {'match': {'REGION_MASK': [8]}}
    ],
}
NA_mask = pop_tools.region_mask_3d(
    grid_name, region_defs=region_defs_na,
    mask_name='North Atlantic').sum('region')    # [nlat, nlon]

# Narrower North Atlantic mask — to restrict BSF display
region_defs_bsf = {
    'SubpolarAtlantic': [
        {'match': {'REGION_MASK': [6]},
         'bounds': {'TLAT': [10.0, 65.0], 'TLONG': [260.0, 360.0]}}
    ],
    'LabradorSea': [
        {'match': {'REGION_MASK': [8]},
         'bounds': {'TLAT': [10.0, 65.0], 'TLONG': [260.0, 360.0]}}
    ],
}
BSF_mask = pop_tools.region_mask_3d(
    grid_name, region_defs=region_defs_bsf,
    mask_name='North Atlantic BSF').sum('region')   # [nlat, nlon]

print('Grid and masks loaded.', flush=True)


### EVENT DETECTION  (same function and parameters as smoc55_picontrol_changepoint.ipynb) ###

def detect_below(annual_vals, annual_yrs, P1_len, P2_len,
                 threshold=THRESHOLD, gap=GAP):
    """Return array of model years where a 2-sigma Below event is detected."""
    events = []
    for j in range(P1_len, len(annual_vals) - P2_len):
        P1_avg = annual_vals[j - P1_len:j].mean()
        P1_std = annual_vals[j - P1_len:j].std()
        P2_avg = annual_vals[j:j + P2_len].mean()
        if P2_avg <= P1_avg - threshold * P1_std:
            if len(events) == 0 or annual_yrs[j] >= events[-1] + gap:
                events.append(annual_yrs[j])
    return np.array(events)


ds_smoc     = xr.open_dataset(SMOC55_FILE)
smoc        = ds_smoc['smoc55_index']
annual_smoc = smoc.resample(time='1Y').mean()
annual_vals = annual_smoc.values
annual_yrs  = np.array([t.year for t in annual_smoc.time.values])
ds_smoc.close()

print(f'piControl: {len(annual_yrs)} annual values.', flush=True)


### COMPOSITE COMPUTATION ###

def open_picontrol(var):
    """Open all monthly piControl CMIP6 files for `var` lazily."""
    var_dir = {'thetao': 'THETAO', 'so': 'SO', 'vo': 'VO'}
    pattern = os.path.join(
        PICONTROL_ROOT, var_dir[var], f'{var}_Omon_CESM2_piControl_*.nc')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f'No piControl files found: {pattern}')
    ds = xr.open_mfdataset(files, combine='by_coords', use_cftime=True,
                            chunks={'time': 120})
    return ds


def build_composite(var, event_years, P1_len, P2_len):
    """
    Compute mean composite of annual-mean `var` over all Below events.

    For each event year j the window is [j-P1_len, j+P2_len) annual steps,
    aligned so that composite time index 0 is the first year of P1 and
    index P1_len is the shift year — same convention as composite_analysis.py
    uses for LENS2.

    Returns an xr.DataArray with:
      dims  : ['time', 'lev', 'nlat', 'nlon']  (thetao/so/vo)
      'time': integer index 0 .. P1_len+P2_len-1
      'lev' : cm coordinate from the CMIP6 files
    """
    window_len = P1_len + P2_len
    ds_full    = open_picontrol(var)
    ds_annual  = ds_full.resample(time='1Y').mean()

    # Capture the lev coordinate (cm) before closing ds_full
    lev_vals = ds_annual['lev'].values if 'lev' in ds_annual.coords else None

    print(f'  {var}: {len(ds_annual.time)} annual steps, '
          f'{len(event_years)} events', flush=True)

    composite_sum = None
    n_valid = 0

    for yr in event_years:
        idx_arr = np.where(annual_yrs == yr)[0]
        if idx_arr.size == 0:
            print(f'  WARNING: event year {yr} not in annual_yrs - skipping.', flush=True)
            continue
        j       = int(idx_arr[0])
        i_start = j - P1_len
        i_end   = j + P2_len    # exclusive

        if i_start < 0 or i_end > len(ds_annual.time):
            print(f'  WARNING: event year {yr} (idx {j}) window out of bounds - skipping.',
                  flush=True)
            continue

        window = ds_annual[var].isel(time=slice(i_start, i_end)).load()

        if composite_sum is None:
            composite_sum = window.values.astype(np.float64)
        else:
            composite_sum += window.values.astype(np.float64)

        n_valid += 1
        elapsed = time_mod.time() - start_time
        print(f'  event year {yr}  (n_valid={n_valid}, {elapsed:.0f}s)', flush=True)

    ds_full.close()

    if n_valid == 0:
        raise RuntimeError(f'No valid events for {var} - aborting.')

    composite_mean = composite_sum / n_valid

    # Determine spatial dims (everything except time)
    spatial_dims = [d for d in ds_annual[var].dims if d != 'time']

    # Build coordinates: composite time index plus lev if present
    coords = {'time': np.arange(window_len)}
    if lev_vals is not None:
        coords['lev'] = lev_vals

    da_composite = xr.DataArray(
        composite_mean,
        dims=['time'] + spatial_dims,
        coords=coords,
    )
    return da_composite


def compute_bsf(da_vo):
    """
    Compute barotropic streamfunction from composite annual-mean vo field.

    Follows composites_stf.py:
      BSF = (vo * dz * DXU).sum(lev).cumsum(nlon)   [m3/s, *1e-6 -> Sv]

    vo is in m/s (CMIP6); dz and DXU are in m (pop_tools, cm -> m).
    Cumsum starts at nlon=0 (Pacific side, native POP). Because the flip-book
    plots anomalies (BSF - P1-mean BSF), the integration constant cancels and
    no longitude roll is required.

    da_vo : [time, lev, nlat, nlon]
    Returns da_bsf : [time, nlat, nlon] in Sv
    """
    lev_vals = da_vo.coords['lev'].values
    dz_xr    = xr.DataArray(dz_m,  dims=['lev'], coords={'lev': lev_vals})
    DXU_xr   = xr.DataArray(DXU_m, dims=['nlat', 'nlon'])

    vo_clean = da_vo.fillna(0.0)
    da_bsf   = (vo_clean * dz_xr * DXU_xr).sum('lev').cumsum('nlon')
    da_bsf   = da_bsf * 1e-6   # m3/s -> Sv
    return da_bsf


### FLIP-BOOK FIGURE  (format from Plot_decrease_aa.ipynb) ###

def plot_flipbook(da_bsf, da_thetao, da_so, P1_len, P2_len, save_path):
    """
    3-row x 3-col orthographic flip-book figure in the style of Plot_decrease_aa.ipynb.

    Rows   : BSF | thetao (depth-mean, slice_z) | so (depth-mean, slice_z)
    Columns: Pre-Shift | Shift | Post-Shift
             10-year averages centred -10, 0, +10 yr relative to shift boundary P1_len.

    For P1=40 the column labels and time slices match Plot_decrease_aa.ipynb exactly:
      Pre-Shift (25-35 years), Shift (35-45 years), Post-Shift (45-55 years).

    Anomaly baseline: first P1_len composite years (P1 mean).
    Black contour lines show the P1-mean climatology.
    """
    # --- Column time-index windows (0-indexed into composite 'time') ---
    shift = P1_len
    years_ranges = [
        (shift - 15, shift -  5),
        (shift -  5, shift +  5),
        (shift +  5, shift + 15),
    ]
    col_names = [
        f'Pre-Shift ({shift - 15}-{shift - 5} years)',
        f'Shift ({shift - 5}-{shift + 5} years)',
        f'Post-Shift ({shift + 5}-{shift + 15} years)',
    ]

    row_labels = [
        r'$\Delta \psi_{b}$ [Sv]',
        r'$\Delta T$ [$^{\circ}$C]',
        r'$\Delta S$ [g kg$^{-1}$]',
    ]
    cmaps_list = [cm.seismic, cmo.balance, cmo.tarn_r]
    vmins_list = [-8, -0.8, -0.3]

    # Depth range for T/S vertical mean — same as Plot_decrease_aa.ipynb
    slice_z = slice(5e2, 5.5e3)   # lev coordinate in cm (POP/CMIP6 native)

    # --- P1-mean baselines ---
    p1_slice       = slice(0, P1_len)
    baseline_bsf    = da_bsf   .isel(time=p1_slice).mean('time')
    baseline_thetao = da_thetao.isel(time=p1_slice).mean('time')
    baseline_so     = da_so    .isel(time=p1_slice).mean('time')

    # --- Anomalies ---
    anom_bsf    = da_bsf    - baseline_bsf
    anom_thetao = da_thetao - baseline_thetao
    anom_so     = da_so     - baseline_so

    # --- Vertical mean for T and S (dz-weighted, same as LENS2 flip-book) ---
    lev_dim  = 'lev' if 'lev' in da_thetao.dims else 'z_t'
    lev_coord = da_thetao[lev_dim].values
    dz_weights = xr.DataArray(dz_m, dims=[lev_dim], coords={lev_dim: lev_coord})

    def depth_mean(da):
        da_sel = da.sel({lev_dim: slice_z})
        w      = dz_weights.sel({lev_dim: slice_z}).fillna(0)
        return da_sel.weighted(w).mean(lev_dim)

    anom_thetao_z = depth_mean(anom_thetao)
    anom_so_z     = depth_mean(anom_so)

    # --- Climatological fields for contour lines ---
    clim_bsf    = baseline_bsf.where(BSF_mask == 1)
    clim_thetao = depth_mean(baseline_thetao).where(NA_mask == 1)
    clim_so     = depth_mean(baseline_so    ).where(NA_mask == 1)

    # --- BSF restricted to North Atlantic ---
    anom_bsf_na = anom_bsf.where(BSF_mask == 1)

    # --- Figure ---
    alphabet  = 'abcdefghi'
    label_idx = 0

    fig, axes = plt.subplots(
        nrows=3, ncols=3,
        subplot_kw={'projection': ccrs.Orthographic(
            central_latitude=35, central_longitude=-35)})

    for col, col_name in enumerate(col_names):
        axes[0, col].set_title(col_name, pad=14)

    # Row 0 — BSF (U-grid, ULAT/ULONG from pop_tools)
    for col, (t_start, t_end) in enumerate(years_ranges):
        ax = axes[0, col]
        da = anom_bsf_na.isel(time=slice(t_start, t_end)).mean('time')
        ax.pcolormesh(ULONG, ULAT, da.values,
                      transform=ccrs.PlateCarree(),
                      cmap=cmaps_list[0], vmin=vmins_list[0], vmax=-vmins_list[0])
        ax.contour(ULONG, ULAT, clim_bsf.values,
                   linewidths=0.3, colors='k', transform=ccrs.PlateCarree())
        ax.set_extent([-90, -5, 35, 65], crs=ccrs.PlateCarree())
        ax.add_feature(nature_coast)
        ax.text(0.03, 0.95, f'({alphabet[label_idx]})', transform=ax.transAxes,
                fontweight='bold', va='top', ha='left', zorder=15)
        label_idx += 1
        for spine in ax.spines.values():
            spine.set_linewidth(0.3)

    # Row 1 — thetao (T-grid, TLAT/TLONG from pop_tools)
    for col, (t_start, t_end) in enumerate(years_ranges):
        ax = axes[1, col]
        da = anom_thetao_z.isel(time=slice(t_start, t_end)).mean('time')
        ax.pcolormesh(TLONG, TLAT, da.values,
                      transform=ccrs.PlateCarree(),
                      cmap=cmaps_list[1], vmin=vmins_list[1], vmax=-vmins_list[1])
        ax.contour(TLONG, TLAT, clim_thetao.values,
                   linewidths=0.3, colors='k', transform=ccrs.PlateCarree())
        ax.set_extent([-90, -5, 35, 65], crs=ccrs.PlateCarree())
        ax.add_feature(nature_coast)
        ax.text(0.03, 0.95, f'({alphabet[label_idx]})', transform=ax.transAxes,
                fontweight='bold', va='top', ha='left', zorder=15)
        label_idx += 1
        for spine in ax.spines.values():
            spine.set_linewidth(0.3)

    # Row 2 — so (T-grid)
    for col, (t_start, t_end) in enumerate(years_ranges):
        ax = axes[2, col]
        da = anom_so_z.isel(time=slice(t_start, t_end)).mean('time')
        ax.pcolormesh(TLONG, TLAT, da.values,
                      transform=ccrs.PlateCarree(),
                      cmap=cmaps_list[2], vmin=vmins_list[2], vmax=-vmins_list[2])
        ax.contour(TLONG, TLAT, clim_so.values,
                   linewidths=0.3, colors='k', transform=ccrs.PlateCarree())
        ax.set_extent([-90, -5, 35, 65], crs=ccrs.PlateCarree())
        ax.add_feature(nature_coast)
        ax.text(0.03, 0.95, f'({alphabet[label_idx]})', transform=ax.transAxes,
                fontweight='bold', va='top', ha='left', zorder=15)
        label_idx += 1
        for spine in ax.spines.values():
            spine.set_linewidth(0.3)

    # --- Colorbars (one per row, right-hand side) ---
    cbar_axes = [fig.add_axes([0.91, 0.665 - i * 0.27, 0.01, 0.2]) for i in range(3)]
    for row_idx in range(3):
        norm = plt.Normalize(vmin=vmins_list[row_idx], vmax=-vmins_list[row_idx])
        sm   = plt.cm.ScalarMappable(cmap=cmaps_list[row_idx], norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_axes[row_idx])
        cbar.ax.set_ylabel('')
        if row_idx == 2:
            cbar.set_ticks([-0.3, 0, 0.3])
            cbar.ax.set_yticklabels(['-0.3', '0', '0.3'])
        for spine in cbar.ax.spines.values():
            spine.set_linewidth(0.3)

    # --- Row labels ---
    for i, label in enumerate(row_labels):
        fig.text(1.01, 0.77 - i * 0.275, label,
                 verticalalignment='center', horizontalalignment='center', rotation=90)

    plt.subplots_adjust(wspace=0.1, hspace=0.01)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'Saved -> {save_path}', flush=True)


### MAIN LOOP ###

for cfg in CONFIGS:
    P1_len = cfg['P1_len']
    P2_len = cfg['P2_len']

    event_years = detect_below(annual_vals, annual_yrs, P1_len, P2_len)
    print(f'\n=== Config P1={P1_len} / P2={P2_len} ===', flush=True)
    print(f'  {len(event_years)} Below events at model years: {event_years}', flush=True)

    tag = f'2.0_{P1_len}_{P2_len}_picontrol'

    # thetao composite
    print('\nComputing thetao composite ...', flush=True)
    da_thetao = build_composite('thetao', event_years, P1_len, P2_len)
    xr.Dataset({'thetao': da_thetao}).to_netcdf(
        os.path.join(OUTPUT_DIR, f'Below_combined_thetao_{tag}.nc'))
    print('  thetao composite saved.', flush=True)

    # so composite
    print('\nComputing so composite ...', flush=True)
    da_so = build_composite('so', event_years, P1_len, P2_len)
    xr.Dataset({'so': da_so}).to_netcdf(
        os.path.join(OUTPUT_DIR, f'Below_combined_so_{tag}.nc'))
    print('  so composite saved.', flush=True)

    # vo composite -> BSF
    print('\nComputing vo composite (for BSF) ...', flush=True)
    da_vo  = build_composite('vo', event_years, P1_len, P2_len)
    da_bsf = compute_bsf(da_vo)
    del da_vo
    gc.collect()
    xr.Dataset({'BSF': da_bsf}).to_netcdf(
        os.path.join(OUTPUT_DIR, f'Below_combined_bsf_{tag}.nc'))
    print('  BSF composite saved.', flush=True)

    # Flip-book figure
    fig_path = os.path.join(
        OUTPUT_DIR, f'Figure_Decrease_picontrol_{P1_len}_{P2_len}_Flipbook.pdf')
    print(f'\nPlotting flip-book ...', flush=True)
    plot_flipbook(da_bsf, da_thetao, da_so, P1_len, P2_len, fig_path)

    del da_bsf, da_thetao, da_so
    gc.collect()

elapsed = time_mod.time() - start_time
print(f'\nAll done. Total time: {elapsed:.0f}s ({elapsed / 60:.1f} min)', flush=True)
