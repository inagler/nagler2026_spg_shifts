# CLAUDE.md

## Project

Revision of **egusphere-2026-2588** — *Multidecadal North Atlantic circulation shifts, CESM2 Large Ensemble (100 members, 1850–2100)*. Central variable: **SMOC55** (density-weighted meridional overturning streamfunction at 55°N, subpolar gyre circulation proxy).

---

## Branch and tag rules

- **Work only on branch `revision`.**
- Never modify tag `v1.0-submitted` or branch `main`.
- All revision outputs go to the revision output directory (see below). **Never overwrite existing outputs.**

---

## Repository structure

```
publishable_code/
├── Plot_*.ipynb                      # Visualization notebooks (manuscript figures)
├── *.py                              # Data preparation and analysis scripts
├── detect_shifts_config.py           # Revision: all detection parameters and paths
├── detect_shifts.py                  # Revision: new shift detection script
├── REVISION_PLAN.md                  # Revision analysis plan and method definition
├── *.sh                              # SLURM job submission scripts (HPC)
├── sensitivity_change_point/         # CSV outputs from change point parameter sweep
├── environment.yml                   # Conda environment specification
└── *.pdf / *.png                     # Final publication figures
```

---

## Data input and output paths

### Raw model input (read-only, on cluster)
| Data | Path |
|---|---|
| CESM2-LE VVEL (monthly) | `/Data/gfi/share/ModData/CESM2_LENS2/ocean/monthly/VVEL/` |
| CESM2-LE TEMP (monthly) | `/Data/gfi/share/ModData/CESM2_LENS2/ocean/monthly/TEMP/` |
| CESM2-LE SALT (monthly) | `/Data/gfi/share/ModData/CESM2_LENS2/ocean/monthly/SALT/` |
| piControl VO (CMIP6)    | `/Data/gfi/share/ModData/CMIP6/raw/ocean/piControl/CESM2/VO/` |
| piControl THETAO (CMIP6)| `/Data/gfi/share/ModData/CMIP6/raw/ocean/piControl/CESM2/THETAO/` |
| piControl SO (CMIP6)    | `/Data/gfi/share/ModData/CMIP6/raw/ocean/piControl/CESM2/SO/` |

### Derived inputs (written by existing pipeline scripts)
| Data | Path |
|---|---|
| SMOC55 member timeseries | `/Data/skd/scratch/innag3580/comp/smoc/smoc55_member_*.nc` |
| SMOC55 piControl timeseries | `/Data/skd/scratch/innag3580/comp/smoc55_picontrol/smoc55_picontrol.nc` |
| Ensemble mean SMOC55 | `/Data/skd/scratch/innag3580/comp/smoc/smoc55_ensemble_mean.nc` |
| Composite netCDF outputs | `/Data/skd/scratch/innag3580/comp/composites/` |

### Existing v1.0 analysis outputs (never overwrite)
| Data | Path |
|---|---|
| Change point CSVs (canonical) | `publishable_code/plotting-change_point_indices_*.csv` |
| Change point CSVs (full sweep) | `publishable_code/sensitivity_change_point/` |

### Revision output directory
All outputs produced during the revision go to:
```
/Data/skd/scratch/innag3580/comp/revision/
```
Subdirectories are created per script (e.g., `shifts/`, `composites/`, `figures/`). **Do not write revision outputs to any other location.**

---

## Conda environment and how to run scripts

Environment name: **`nagler2026_spg_shifts`** (Python 3.10, defined in `environment.yml`).

```bash
# Activate
conda activate nagler2026_spg_shifts

# Run new detection script (revision)
python detect_shifts.py

# Run existing change point analysis (v1.0 pipeline, do not modify)
python change_point_analysis.py 40 20 3.0 5 False

# Run on HPC via SLURM
sbatch run_change_point.sh
sbatch run_composite.sh

# Launch Jupyter for plotting notebooks
jupyter notebook
```

Key dependencies: `xarray`, `dask`, `pop_tools`, `gsw`, `matplotlib`, `cartopy`, `cmocean`, `cmcrameri`.

Annual resampling uses `resample(time='1Y')` (not `'1YE'`); this is the correct alias for the xarray version in this environment.

---

## Known issues

1. **`CT_from_pt` called with `SALT` twice** in `Plot_smoc+dens_historical_trend.ipynb`: the correct call is `CT_from_pt(SA, pt)` but the notebook passes `SALT` as both arguments, producing incorrect conservative temperature and therefore incorrect sigma-2. Fix in the revision figure (RC item).

2. **Inconsistent composite file naming**: composite NetCDF files use mixed naming conventions across scripts (e.g., `Below_combined_VVEL_2.0_40_20_aa_hist.nc` vs `Below_combined_VVEL_3.0_40_20.nc`). The threshold/window suffix scheme is not applied consistently for all category names.

3. **sMOC block broken in `composites_stf.py`**: `ds_salt` and `ds_temp` are commented out (lines 30–34) but referenced at line 53 (`CT_from_pt(ds_salt['SALT'], ds_temp['TEMP'])`), causing a `NameError` at runtime for the `ghg_hist` and `aa_hist` branches.

4. **Detection logic duplicated in three places**: the sliding-window exceedance logic appears in `change_point_analysis.py`, `parallel_composite_analysis.py`, and notebooks. The revision replaces this with the single authoritative detection function in `detect_shifts.py`.

---

## Shift detection method

The revision detection method is defined precisely in **REVISION_PLAN.md Section 2**. Use the names defined there exactly:

- Input series: **Ψ_s(mem, yr)** — annual mean SMOC55 maximum
- Normalised shift magnitude: **η(mem, yr)**
- Window lengths: **L_pre = 40**, **L_post = 20**
- Minimum separation: **D = L_post = 20**
- Thresholds: **k = 2** (weakening yr_s < 1985), **k = 3** (all other categories)
- Categories: **strengthening**, **weakening_early** (yr_s < 1985), **weakening_late** (yr_s ≥ 1985)
- Detection end year: **2029**

See REVISION_PLAN.md for the full method, including clustering, separation rule, and piControl procedure.

---

## Workflow summary

1. **Prepare**: `smoc55_timeseries.py`, `smoc55_picontrol.py`, `dens2.py`, `ensemble_mean_temp_salt.py`
2. **Detect shifts (revision)**: `detect_shifts.py` (reads config from `detect_shifts_config.py`)
3. **Composites**: `composite_analysis.py` / `parallel_composite_analysis.py` / `composites_stf.py` (existing; revision will add new composite scripts)
4. **Plot**: `Plot_*.ipynb` notebooks

---

## Notes

- License: MIT (Copyright 2026 Ina Nagler)
- The four `plotting-change_point_indices_*.csv` files are the v1.0 canonical set; they use a gap-based (not cluster-based) detection and serve as a comparison baseline for the revision.
