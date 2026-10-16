#!/usr/bin/env python3
"""
compute_composites.py — Revision: raw composites for CESM2-LE SMOC55 shift events.

Usage:
    ~/.conda/envs/work/bin/python compute_composites.py <VAR>

    VAR: TEMP | SALT | VVEL | HMXL | SHF | AICE | N_HEAT | N_SALT

For each category, reads shifts_k2_hist.csv, extracts a 60 DJFM-year window
centred on yr_s (L_pre=40 pre, L_post=20 post), and writes one NetCDF per category:

    /Data/skd/scratch/innag3580/comp/revision/composites/raw/{category}_{VAR}.nc

Output dimensions: (shift, time[, z_t], nlat, nlon)
  shift : string labels 'mem_yr_s', e.g. '1001.001_1924'
  time  : relative integer years -40 … +19
"""

import glob
import os
import sys
import warnings

import cftime
import dask
import numpy as np
import pandas as pd
import xarray as xr
from dask.array import PerformanceWarning

warnings.filterwarnings("ignore", category=PerformanceWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detect_shifts_config as cfg

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_PATH       = '/Data/gfi/share/ModData/CESM2_LENS2/ocean/monthly'
SHIFTS_FILE     = os.path.join(cfg.REVISION_OUTPUT_DIR, 'shifts_k2_hist.csv')
OUTPUT_DIR      = '/Data/skd/scratch/innag3580/comp/revision/composites/raw'
TMP_DIR         = '/Data/skd/scratch/innag3580/comp/revision/composites/tmp'
EXCLUDE_MEMBERS = {'1231.002', '1231.01', '1281.012'}

WINDOW_PRE  = cfg.L_PRE    # 40
WINDOW_POST = cfg.L_POST   # 20
WINDOW_LEN  = WINDOW_PRE + WINDOW_POST  # 60

# Output time coordinate: relative years -40 … +19 (60 values)
REL_YEARS = np.arange(-WINDOW_PRE, WINDOW_POST, dtype=np.int32)

# Category split boundaries
STR_SPLIT = 1930                   # strengthening: < 1930 → early, ≥ 1930 → late
WK_SPLIT  = cfg.CATEGORY_BOUNDARY  # weakening:     < 1985 → early, ≥ 1985 → late

# ---------------------------------------------------------------------------
# DJFM helpers (same logic as composite_analysis.py)
# ---------------------------------------------------------------------------

def standardise_time(ds):
    """Ensure time coordinate is datetime64; convert cftime NoLeap if needed."""
    ds['time'] = xr.decode_cf(ds, use_cftime=True).time
    if len(ds.time) > 0 and isinstance(ds.time.values[0], cftime._cftime.DatetimeNoLeap):
        ts = np.array([pd.Timestamp(str(dt)).to_datetime64() for dt in ds.time.values])
        ds['time'] = xr.DataArray(ts, dims='time')
    return ds


def djfm_average(ds):
    """
    DJFM average of a dataset already filtered to months [12, 1, 2, 3].

    Matches the coarsen logic in composite_analysis.py:
      - First 2 timesteps (Jan + Feb of the start year): coarsen by 2
      - Remaining timesteps (Mar onwards, groups of Dec-Jan-Feb-Mar): coarsen by 4
    Produces exactly one DJFM value per calendar year.
    """
    numeric = {k: v for k, v in ds.data_vars.items()
               if np.issubdtype(v.dtype, np.number)}
    ds_num = xr.Dataset(numeric, coords=ds.coords)
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        first = ds_num.isel(time=slice(0, 2)).coarsen(time=2, boundary='trim').mean()
        rest  = ds_num.isel(time=slice(2, None)).coarsen(time=4, boundary='trim').mean()
    return xr.concat([first, rest], dim='time')


def make_preprocess(var_key):
    """Return a preprocess function that standardises time and drops unused data vars."""
    def preprocess(ds):
        ds = standardise_time(ds)
        drop = [v for v in ds.data_vars if v != var_key]
        return ds.drop_vars(drop, errors='ignore')
    return preprocess


# ---------------------------------------------------------------------------
# Member loading
# ---------------------------------------------------------------------------

def load_member_djfm(var, member_id, var_key):
    """
    Load all files for (var, member_id) across both BHIST and BSSP periods,
    apply DJFM month filtering and DJFM averaging.

    Returns
    -------
    ds_djfm   : xr.Dataset with one DJFM-averaged value per calendar year
    start_year: int, calendar year of the first DJFM timestep (= first year of record)
    """
    # Hard guard: excluded members must never be opened regardless of how this is called
    if member_id in EXCLUDE_MEMBERS:
        return None

    dir_name = 'AICE' if var == 'AICE' else var
    pattern  = os.path.join(BASE_PATH, dir_name, f'*LE2-{member_id}*.nc')
    paths    = sorted(glob.glob(pattern))

    # Belt-and-suspenders: drop any path that contains an excluded member string
    paths = [p for p in paths if not any(ex in p for ex in EXCLUDE_MEMBERS)]

    # Drop files whose date range starts after 2040 (filename ends …YYYYMM-YYYYMM.nc)
    def _file_start_year(path):
        stem = os.path.basename(path).rsplit('.', 1)[0]  # strip .nc
        date_part = stem.rsplit('-', 2)[-2]               # first YYYYMM token
        return int(date_part[:4])

    paths = [p for p in paths if _file_start_year(p) <= 2040]

    if not paths:
        raise FileNotFoundError(
            f'No files matched for member {member_id}, var {var}\n  pattern: {pattern}'
        )

    ds = xr.open_mfdataset(
        paths,
        chunks={'time': 120},
        preprocess=make_preprocess(var_key),
        drop_variables=['time_bound'],
    )
    ds[var_key] = ds[var_key].astype('float32')

    # Filter to DJFM months
    ds_djfm_months = ds.sel(time=ds['time.month'].isin([12, 1, 2, 3]))

    # Start year = calendar year of the first DJFM-filtered timestep
    start_year = int(ds_djfm_months.time.dt.year.values[0])

    # DJFM average
    ds_djfm = djfm_average(ds_djfm_months)
    ds.close()
    return ds_djfm, start_year


# ---------------------------------------------------------------------------
# Window extraction
# ---------------------------------------------------------------------------

def extract_window(ds_djfm, start_year, yr_s, member_id):
    """
    Extract a WINDOW_LEN-year DJFM window centred on yr_s.

    Window indices in the DJFM-averaged series:
        [yr_s - start_year - WINDOW_PRE,  yr_s - start_year + WINDOW_POST)

    Returns (ds_window, None) on success, (None, skip_message) if out of bounds.
    """
    idx = yr_s - start_year
    i0  = idx - WINDOW_PRE
    i1  = idx + WINDOW_POST
    n   = len(ds_djfm.time)

    if i0 < 0 or i1 > n:
        msg = (
            f'  SKIP {member_id} yr_s={yr_s}: '
            f'window [{yr_s - WINDOW_PRE}, {yr_s + WINDOW_POST - 1}] '
            f'outside record [{start_year}, {start_year + n - 1}]'
        )
        return None, msg

    return ds_djfm.isel(time=slice(i0, i1)), None


# ---------------------------------------------------------------------------
# Event categorisation
# ---------------------------------------------------------------------------

def build_categories(df):
    """Split events into the four composite categories."""
    s = df['sign'] == 'strengthening'
    w = df['sign'] == 'weakening'
    return {
        'strengthening_early': df[s & (df['yr_s'] <  STR_SPLIT)],
        'strengthening_late' : df[s & (df['yr_s'] >= STR_SPLIT)],
        'weakening_early'    : df[w & (df['yr_s'] <  WK_SPLIT)],
        'weakening_late'     : df[w & (df['yr_s'] >= WK_SPLIT)],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(line_buffering=True)

    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    var = sys.argv[1].upper()
    valid = {'TEMP', 'SALT', 'VVEL', 'HMXL', 'SHF', 'AICE', 'N_HEAT', 'N_SALT'}
    if var not in valid:
        print(f'ERROR: unknown variable {var!r}. Choose from: {sorted(valid)}')
        sys.exit(1)

    var_key = 'aice' if var == 'AICE' else var

    # --- Load and filter event list ---
    df = pd.read_csv(SHIFTS_FILE)
    df['mem'] = df['mem'].astype(str).str.strip()
    n_before = len(df)
    excluded = df[df['mem'].isin(EXCLUDE_MEMBERS)]
    df = df[~df['mem'].isin(EXCLUDE_MEMBERS)].copy()
    print(f'CSV filter: {n_before} events → {len(df)} after excluding '
          f'{sorted(EXCLUDE_MEMBERS)}')
    for ex_mem in sorted(EXCLUDE_MEMBERS):
        count = (excluded['mem'] == ex_mem).sum()
        print(f'  {ex_mem}: {count} events removed')

    categories = build_categories(df)

    # --- Startup banner (printed before any file loading begins) ---
    first_cat, first_mem = 'none', 'none'
    for _cname, _cdf in categories.items():
        if len(_cdf) > 0:
            first_cat = _cname
            first_mem = str(_cdf.iloc[0]['mem'])
            break

    print('=' * 60)
    print('compute_composites.py  —  SMOC55 shift composites (revision)')
    print('=' * 60)
    print(f'  variable    : {var}  (key: {var_key})')
    print(f'  window      : L_pre={WINDOW_PRE}, L_post={WINDOW_POST}  →  {WINDOW_LEN} DJFM years')
    print(f'  shifts file : {SHIFTS_FILE}')
    print(f'  output dir  : {OUTPUT_DIR}')
    print()
    print(f'  {n_before} events in CSV → {len(df)} after excluding {sorted(EXCLUDE_MEMBERS)}')
    print()
    print('  category counts:')
    for _cname, _cdf in categories.items():
        print(f'    {_cname:<24}: {len(_cdf):3d} events')
    print(f'    {"total":<24}: {sum(len(v) for v in categories.values()):3d} events')
    print()
    print(f'  first load  : member {first_mem}  [{first_cat}]')
    print('-' * 60)
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    # --- Process each category ---
    for cat_name, cat_df in categories.items():

        if len(cat_df) == 0:
            print(f'[{cat_name}] 0 events — skipping.\n')
            continue

        out_path = os.path.join(OUTPUT_DIR, f'{cat_name}_{var}.nc')
        if os.path.exists(out_path):
            print(f'[{cat_name}] Output already exists — skipping:\n  {out_path}\n')
            continue

        print(f'[{cat_name}] {len(cat_df)} events → {out_path}')

        member_cache = {}   # member_id → (ds_djfm, start_year)
        tmp_files    = []   # paths of per-window temp NetCDF files
        labels       = []   # shift coordinate labels (for reporting)

        for _, row in cat_df.iterrows():
            mem  = str(row['mem'])
            yr_s = int(row['yr_s'])

            print(f'  member {mem}  yr_s={yr_s}')

            # Load member (cached within this category)
            if mem not in member_cache:
                print(f'    loading {mem} ...')
                try:
                    result = load_member_djfm(var, mem, var_key)
                    if result is None:
                        print(f'    WARNING: {mem} is in EXCLUDE_MEMBERS — skipping.')
                        continue
                    ds_djfm, start_year = result
                    member_cache[mem] = (ds_djfm, start_year)
                    print(f'    loaded: {len(ds_djfm.time)} DJFM years '
                          f'[{start_year} – {start_year + len(ds_djfm.time) - 1}]')
                except FileNotFoundError as exc:
                    print(f'    ERROR: {exc}')
                    continue

            ds_djfm, start_year = member_cache[mem]

            # Extract window
            ds_win, skip_msg = extract_window(ds_djfm, start_year, yr_s, mem)
            if skip_msg is not None:
                print(skip_msg)
                continue

            # Write window to a temp file (avoids stacking all windows in memory)
            label    = f'{mem}_{yr_s}'
            tmp_path = os.path.join(TMP_DIR, f'{cat_name}_{var}_{label}.nc')
            da = ds_win[var_key].assign_coords(time=REL_YEARS).compute()
            da.expand_dims({'shift': [label]}).to_dataset(name=var_key).to_netcdf(tmp_path)
            tmp_files.append(tmp_path)
            labels.append(label)

        if not tmp_files:
            print(f'  No valid windows for [{cat_name}] — nothing written.\n')
            for ds_djfm, _ in member_cache.values():
                ds_djfm.close()
            member_cache.clear()
            continue

        print(f'  Concatenating {len(tmp_files)} windows via open_mfdataset ...')
        ds_cat = xr.open_mfdataset(tmp_files, concat_dim='shift', combine='nested')

        print(f'  Writing {out_path}')
        ds_cat.to_netcdf(out_path)
        ds_cat.close()

        # Delete temp files
        for f in tmp_files:
            os.remove(f)
        tmp_files.clear()
        print(f'  Done: {cat_name}_{var}.nc\n')

        # Release memory before next category
        for ds_djfm, _ in member_cache.values():
            ds_djfm.close()
        member_cache.clear()

    print('All categories processed.')


if __name__ == '__main__':
    main()
