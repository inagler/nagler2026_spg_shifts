#!/usr/bin/env python3
"""
detect_shifts.py  —  Shift detection for SMOC55 (revision of egusphere-2026-2588).

Implements the method defined in REVISION_PLAN.md Section 2.
New script; does not modify any existing analysis files.

Outputs (per k, to REVISION_OUTPUT_DIR):
  shifts_k{k}_hist.csv       — detected shifts, historical members
  shifts_k{k}_picontrol.csv  — detected shifts, piControl
  eta_hist.csv               — η(mem, yr) for all historical members and valid years
  eta_picontrol.csv          — η(yr) for piControl and all valid years

Run:
    conda activate nagler2026_spg_shifts
    python detect_shifts.py
"""

import csv
import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

import detect_shifts_config as cfg


# ---------------------------------------------------------------------------
# Core detection functions
# ---------------------------------------------------------------------------

def compute_eta_series(years, values, L_pre, L_post, end_year=None):
    """
    Compute normalised shift magnitude η(yr) for all valid windows.

    Pre-shift:  [yr − L_pre, yr − 1] (inclusive, L_pre years)
    Post-shift: [yr, yr + L_post − 1] (inclusive, L_post years)
    η(yr) = (mean_post − mean_pre) / std_pre

    Parameters
    ----------
    years    : 1-D int array of calendar (or model) years, length T
    values   : 1-D float array of annual SMOC55, length T
    L_pre, L_post : window lengths in years
    end_year : last year allowed in the post window (inclusive).
               None → max(years).

    Returns
    -------
    yr_arr  : 1-D int array of yr values for which η is defined
    eta_arr : 1-D float array of corresponding η values
    """
    if end_year is None:
        end_year = int(years[-1])

    yr_min = int(years[0]) + L_pre       # earliest yr: pre window fits from years[0]
    yr_max = end_year - L_post + 1       # latest  yr: post window ends at end_year

    # Fast year-to-index lookup
    yr2idx = {int(y): i for i, y in enumerate(years)}

    yr_list, eta_list = [], []

    for yr in range(yr_min, yr_max + 1):
        # Verify all required years are present (handles any data gaps)
        pre_yrs  = range(yr - L_pre, yr)
        post_yrs = range(yr, yr + L_post)
        if any(y not in yr2idx for y in pre_yrs):
            continue
        if any(y not in yr2idx for y in post_yrs):
            continue

        pre_vals  = np.array([values[yr2idx[y]] for y in pre_yrs],  dtype=float)
        post_vals = np.array([values[yr2idx[y]] for y in post_yrs], dtype=float)

        std_pre = float(np.nanstd(pre_vals, ddof=1))
        if std_pre == 0.0 or not np.isfinite(std_pre):
            continue

        eta = (np.nanmean(post_vals) - np.nanmean(pre_vals)) / std_pre
        yr_list.append(yr)
        eta_list.append(eta)

    return np.array(yr_list, dtype=int), np.array(eta_list, dtype=float)


def find_clusters(yr_arr, eta_arr, k, sign):
    """
    Identify shift candidates by clustering consecutive exceedance years.

    A cluster is a maximal run of consecutive years (no gap) in yr_arr where
    the exceedance condition holds.  The shift year yr_s within a cluster is
    the year with the largest |η|.

    Parameters
    ----------
    yr_arr, eta_arr : output of compute_eta_series (parallel arrays)
    k    : exceedance threshold
    sign : +1 (strengthening, η ≥ k) or −1 (weakening, η ≤ −k)

    Returns
    -------
    List of dicts, one per cluster:
        yr_s          – shift year (year of max |η| in cluster)
        eta           – η at yr_s
        cluster_start – first year of the exceedance run
        cluster_end   – last  year of the exceedance run
    """
    if sign == 1:
        exceed = eta_arr >= k
    else:
        exceed = eta_arr <= -k

    candidates = []
    in_cluster = False
    c_start = c_end = None
    c_yrs, c_etas = [], []

    for i, yr in enumerate(yr_arr):
        if exceed[i]:
            if not in_cluster:
                in_cluster = True
                c_start = yr
            c_end = yr
            c_yrs.append(yr)
            c_etas.append(eta_arr[i])
        else:
            if in_cluster:
                _close_cluster(candidates, c_yrs, c_etas, c_start, c_end)
                in_cluster = False
                c_yrs, c_etas = [], []

    if in_cluster:
        _close_cluster(candidates, c_yrs, c_etas, c_start, c_end)

    return candidates


def _close_cluster(candidates, c_yrs, c_etas, c_start, c_end):
    best = int(np.argmax(np.abs(c_etas)))
    candidates.append({
        'yr_s'         : c_yrs[best],
        'eta'          : c_etas[best],
        'cluster_start': c_start,
        'cluster_end'  : c_end,
    })


def apply_separation(candidates, D):
    """
    Greedy separation rule.

    Accept largest |η| first; discard any candidate whose yr_s is within D
    years of an already-accepted shift.  Strengthening and weakening are
    handled independently (separate calls), so they do not suppress each other.

    Parameters
    ----------
    candidates : list of dicts from find_clusters
    D          : minimum separation between accepted yr_s values (years)

    Returns
    -------
    List of accepted candidate dicts, sorted by yr_s.
    """
    sorted_cands = sorted(candidates, key=lambda c: abs(c['eta']), reverse=True)
    accepted = []
    accepted_yrs = []

    for cand in sorted_cands:
        yr_s = cand['yr_s']
        if all(abs(yr_s - ay) >= D for ay in accepted_yrs):
            accepted.append(cand)
            accepted_yrs.append(yr_s)

    return sorted(accepted, key=lambda c: c['yr_s'])


def assign_category(yr_s, sign, yr_split, category_boundary):
    """
    Return the category string for a detected shift.

    Categories (REVISION_PLAN.md Section 2):
      strengthening        — sign=+1, yr_split is None
      strengthening_early  — sign=+1, yr_s < yr_split
      strengthening_late   — sign=+1, yr_s >= yr_split
      weakening_early      — sign=−1, yr_s < category_boundary (1985)
      weakening_late       — sign=−1, yr_s >= category_boundary (1985)
    """
    if sign == 1:
        if yr_split is not None:
            return 'strengthening_early' if yr_s < yr_split else 'strengthening_late'
        return 'strengthening'
    else:
        return 'weakening_early' if yr_s < category_boundary else 'weakening_late'


def run_detection(series_dict, L_pre, L_post, k, D, end_year,
                  yr_split, category_boundary):
    """
    Run shift detection for a collection of members (or piControl) at threshold k.

    Parameters
    ----------
    series_dict : {mem_id: (years_arr, values_arr)}
    L_pre, L_post, k, D : detection parameters (see REVISION_PLAN.md Sec. 2)
    end_year    : last year allowed in post window for calendar-year series;
                  pass None to use all available years (piControl).
    yr_split    : strengthening sub-split year (int or None)
    category_boundary : weakening split year (1985)

    Returns
    -------
    shifts_rows : list of dicts (one per detected shift)
    eta_rows    : list of dicts (one η entry per valid (mem, yr) pair)
                  — returned once (independent of k)
    """
    shifts_rows = []
    eta_rows    = []

    for mem, (years, values) in sorted(series_dict.items()):
        yr_arr, eta_arr = compute_eta_series(
            years, values, L_pre, L_post, end_year=end_year
        )

        for yr, eta in zip(yr_arr, eta_arr):
            eta_rows.append({'mem': mem, 'yr': int(yr), 'eta': float(eta)})

        for sign in [+1, -1]:
            candidates = find_clusters(yr_arr, eta_arr, k, sign)
            accepted   = apply_separation(candidates, D)
            sign_str   = 'strengthening' if sign == 1 else 'weakening'

            for c in accepted:
                category = assign_category(
                    c['yr_s'], sign, yr_split, category_boundary
                )
                shifts_rows.append({
                    'mem'          : mem,
                    'yr_s'         : int(c['yr_s']),
                    'sign'         : sign_str,
                    'eta'          : float(c['eta']),
                    'cluster_start': int(c['cluster_start']),
                    'cluster_end'  : int(c['cluster_end']),
                    'category'     : category,
                })

    return shifts_rows, eta_rows


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_historical_members(smoc_dir, end_year):
    """
    Load all smoc55_member_*.nc files and return annual mean series.

    Returns
    -------
    series_dict : {mem_id: (years_arr, values_arr)}
        mem_id    : 8-char string extracted from filename (e.g. '1001.001')
        years_arr : 1-D int array of calendar years (1850 … end_year)
        values_arr: 1-D float array of annual mean SMOC55 (Sv)
    """
    files = sorted(glob.glob(os.path.join(smoc_dir, 'smoc55_member_*.nc')))
    if not files:
        raise FileNotFoundError(f'No smoc55_member_*.nc files found in {smoc_dir}')

    series_dict = {}
    end_month = (end_year - 1850 + 1) * 12  # months from 1850-01 to end_year-12

    for fpath in files:
        mem_id = os.path.basename(fpath)[-11:-3]  # 8 chars before '.nc'
        da = xr.load_dataarray(fpath).isel(time=slice(0, end_month))
        annual = da.resample(time='1Y').mean()
        years  = annual.time.dt.year.values.astype(int)
        values = annual.values.astype(float)
        series_dict[mem_id] = (years, values)

    print(f'Loaded {len(series_dict)} historical members '
          f'({min(len(v[0]) for v in series_dict.values())}–'
          f'{max(len(v[0]) for v in series_dict.values())} years each)')
    return series_dict


def load_picontrol(picontrol_file):
    """
    Load the piControl SMOC55 timeseries and return annual mean series.

    Returns
    -------
    series_dict : {'piControl': (years_arr, values_arr)}
        years_arr : 1-D int array of model years
        values_arr: 1-D float array of annual mean SMOC55 (Sv)
    """
    ds = xr.open_dataset(picontrol_file)
    da = ds['smoc55_index']
    annual = da.resample(time='1Y').mean()
    years  = annual.time.dt.year.values.astype(int)
    values = annual.values.astype(float)
    ds.close()
    print(f'Loaded piControl: {len(years)} years '
          f'(model years {years[0]}–{years[-1]})')
    return {'piControl': (years, values)}


# ---------------------------------------------------------------------------
# Comparison with existing v1.0 CSVs
# ---------------------------------------------------------------------------

def count_existing(csv_dir, threshold):
    """
    Count Above / Below detections from existing plotting-change_point_indices_*.csv.

    Parameters
    ----------
    csv_dir   : directory containing plotting-change_point_indices_*.csv
    threshold : float, e.g. 2.0 or 3.0

    Returns
    -------
    Dict {'Above': int, 'Below': int} or None if file not found.
    """
    fname = f'plotting-change_point_indices_{threshold:.1f}_40_20_5.csv'
    fpath = os.path.join(csv_dir, fname)
    if not os.path.isfile(fpath):
        return None
    counts = {'Above': 0, 'Below': 0}
    with open(fpath, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cond = row.get('Condition', '').strip()
            if cond in counts:
                counts[cond] += 1
    return counts


def print_comparison(shifts_rows, k, label, csv_dir):
    """Print shift counts per category and compare against existing v1.0 CSV."""
    from collections import Counter
    cats = Counter(r['category'] for r in shifts_rows)
    total_str = sum(v for c, v in cats.items() if 'strengthen' in c)
    total_wk  = sum(v for c, v in cats.items() if 'weaken'     in c)

    print(f'\n  k={k}  {label}')
    print(f'    {"category":<22}  count')
    for cat in sorted(cats):
        print(f'    {cat:<22}  {cats[cat]}')
    print(f'    {"TOTAL strengthening":<22}  {total_str}')
    print(f'    {"TOTAL weakening":<22}  {total_wk}')

    existing = count_existing(csv_dir, float(k))
    if existing is not None:
        print(f'\n    v1.0 reference (gap=5 method, k={k}):')
        print(f'      Above (strengthening): {existing["Above"]}')
        print(f'      Below (weakening):     {existing["Below"]}')
    else:
        print(f'\n    v1.0 reference not found for k={k}.')


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

SHIFTS_FIELDNAMES = ['mem', 'yr_s', 'sign', 'eta',
                     'cluster_start', 'cluster_end', 'category']
ETA_FIELDNAMES    = ['mem', 'yr', 'eta']


def write_csv(rows, fieldnames, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'  Wrote {len(rows):>6} rows → {filepath}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('=' * 60)
    print('detect_shifts.py  —  SMOC55 shift detection (revision)')
    print('=' * 60)
    print(f'  L_pre={cfg.L_PRE}, L_post={cfg.L_POST}, D={cfg.D}')
    print(f'  K_VALUES={cfg.K_VALUES}, END_YEAR={cfg.END_YEAR}')
    print(f'  CATEGORY_BOUNDARY={cfg.CATEGORY_BOUNDARY}, YR_SPLIT={cfg.YR_SPLIT}')
    print(f'  Output dir: {cfg.REVISION_OUTPUT_DIR}')
    print()

    # --- Load data ---
    print('Loading historical members ...')
    hist_series = load_historical_members(cfg.SMOC_DIR, cfg.END_YEAR)

    print('\nLoading piControl ...')
    pc_series = load_picontrol(cfg.PICONTROL_FILE)

    # --- Detection for each k ---
    eta_hist_rows = None   # η rows are k-independent; compute once
    eta_pc_rows   = None

    for k in cfg.K_VALUES:
        print(f'\n{"─"*50}')
        print(f'Running detection for k = {k}')

        # Historical members
        shifts_hist, eta_hist_rows_k = run_detection(
            hist_series,
            cfg.L_PRE, cfg.L_POST, k, cfg.D,
            end_year=cfg.END_YEAR,
            yr_split=cfg.YR_SPLIT,
            category_boundary=cfg.CATEGORY_BOUNDARY,
        )
        if eta_hist_rows is None:
            eta_hist_rows = eta_hist_rows_k  # store on first k only

        # piControl (no calendar end_year cutoff)
        shifts_pc, eta_pc_rows_k = run_detection(
            pc_series,
            cfg.L_PRE, cfg.L_POST, k, cfg.D,
            end_year=None,
            yr_split=cfg.YR_SPLIT,
            category_boundary=cfg.CATEGORY_BOUNDARY,
        )
        if eta_pc_rows is None:
            eta_pc_rows = eta_pc_rows_k

        # Print counts and comparison
        print_comparison(shifts_hist, k, 'historical members', cfg.EXISTING_CSV_DIR)
        print_comparison(shifts_pc,   k, 'piControl',         cfg.EXISTING_CSV_DIR)

        # Write shift CSVs
        write_csv(
            shifts_hist,
            SHIFTS_FIELDNAMES,
            os.path.join(cfg.REVISION_OUTPUT_DIR, f'shifts_k{k}_hist.csv'),
        )
        write_csv(
            shifts_pc,
            SHIFTS_FIELDNAMES,
            os.path.join(cfg.REVISION_OUTPUT_DIR, f'shifts_k{k}_picontrol.csv'),
        )

    # Write η CSVs (once, k-independent)
    print(f'\n{"─"*50}')
    write_csv(
        eta_hist_rows,
        ETA_FIELDNAMES,
        os.path.join(cfg.REVISION_OUTPUT_DIR, 'eta_hist.csv'),
    )
    write_csv(
        eta_pc_rows,
        ETA_FIELDNAMES,
        os.path.join(cfg.REVISION_OUTPUT_DIR, 'eta_picontrol.csv'),
    )

    print('\nDone.')


if __name__ == '__main__':
    main()
