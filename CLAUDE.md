# CLAUDE.md

## Project Overview

This repository contains the full analysis and plotting code for a scientific manuscript on **multidecadal North Atlantic circulation shifts under historical anthropogenic forcing**, using the CESM2 Large Ensemble (100 members, 1850–2100). The central variable is the **SMOC55 index** (density-weighted meridional overturning streamfunction at 55°N), a proxy for subpolar gyre circulation strength.

The analysis decomposes circulation shifts into aerosol-forced (AA) and greenhouse gas-forced (GHG) drivers.

---

## Repository Structure

```
publishable_code/
├── Plot_*.ipynb                   # Visualization notebooks (produce manuscript figures)
├── *.py                           # Data preparation and analysis scripts
├── *.sh                           # SLURM job submission scripts (HPC)
├── sensitivity_change_point/      # CSV outputs from change point parameter sweep
├── environment.yml                # Conda environment specification
└── *.pdf / *.png                  # Final publication figures
```

---

## Workflow

The pipeline has three phases:

### 1. Prepare (data extraction)
- `smoc55_timeseries.py` — extracts annual SMOC55 index from raw CESM2 monthly output for all 100 ensemble members
- `dens2.py` — computes sigma-2 density via TEOS-10 (requires `gsw`)
- `ensemble_mean_temp_salt.py` — calculates ensemble mean temperature/salinity for detrending
- `rename_vars.py` — utility for NetCDF variable renaming

### 2. Analyse (change point detection & composites)
- `change_point_analysis.py` — sliding window algorithm detecting abrupt shifts:
  - Compares 40-year baseline statistics to a 20-year test window
  - Identifies "Above" (strengthening) and "Below" (weakening) events
  - Parameterised by threshold (1–4σ), window lengths, gap, and detrended/raw mode
  - Outputs CSV files: `change_point_indices_<threshold>_<P1>_<P2>_<gap>[-detrended].csv`
- `composite_analysis.py` / `parallel_composite_analysis.py` — compute 3D spatiotemporal composites (TEMP, SALT, MLD, SHF, streamfunctions) around detected shifts; parallelised with Dask for HPC
- `composites_stf.py` — composite analysis specific to streamfunction variables
- `averages.py` / `standard_deviations.py` — spatial/temporal means and ensemble spread

Run analyses on the cluster via the SLURM scripts (`run_change_point.sh`, `run_composite.sh`).

### 3. Plot (figure generation)
All plotting is done in Jupyter notebooks. Each notebook reads composite netCDF files and CSV change point results, and outputs publication-ready PDFs at 300 DPI (Copernicus journal formatting).

| Notebook | Output figure |
|---|---|
| `Plot_increase.ipynb` | Strengthening event spatial patterns (flipbook + timeseries) |
| `Plot_decrease_aa.ipynb` | Aerosol-forced weakening spatial patterns |
| `Plot_decrease_ghg.ipynb` | GHG-forced weakening spatial patterns |
| `Plot_depth_density_anomalies.ipynb` | Vertical density anomaly profiles (T vs. S decomposition) |
| `Plot_map.ipynb` | Climatological density map with regional masks |
| `Plot_occurences.ipynb` | Shift occurrence histogram and ensemble timeseries |
| `Plot_smoc+dens_historical_trend.ipynb` | Historical trends in SMOC55 and density (1850–2030) |
| `Plot_schematic.ipynb` | Conceptual schematic diagrams |

---

## Key Data Paths (on cluster)

- **Raw CESM2 input:** `/Data/gfi/share/ModData/CESM2_LENS2/`
- **Composite netCDF outputs:** `/Data/skd/scratch/innag3580/comp/`
- **Change point CSVs:** `sensitivity_change_point/` (in repo) and repo root (`plotting-change_point_indices_*.csv`)

---

## Environment

Managed via conda. Install with:
```bash
conda env create -f environment.yml
```

Key dependencies:
- `xarray`, `dask` — array operations and out-of-core parallel processing
- `pop_tools` — POP ocean model mesh operations
- `gsw` — TEOS-10 seawater thermodynamics (sigma-2 density)
- `matplotlib`, `cartopy` — plotting and geographic projections
- `cmocean`, `cmcrameri` — scientific colormaps

---

## Key Scientific Concepts

- **SMOC55**: Density-weighted MOC streamfunction at 55°N — the primary circulation index
- **Sigma-2**: Potential density referenced to 2000 dbar (conservative water mass variable)
- **Change point detection**: Sliding window comparison; a shift is flagged when a 20-year mean moves beyond the baseline variability by N standard deviations
- **Composite analysis**: Ensemble-mean spatial/temporal anomaly fields averaged across members at detected shift events
- **Forcing decomposition**: AA = aerosol-driven scenarios; GHG = greenhouse gas-driven scenarios

---

## Notes

- The four CSV files prefixed `plotting-change_point_indices_*.csv` are the canonical set used for manuscript figures (filtered from the full sensitivity sweep of 45+ variants).
- Notebooks previously named `Prepare_*.ipynb` have been removed from the repo; preparation is now handled by standalone Python scripts.
- License: MIT (Copyright 2026 Ina Nagler)
