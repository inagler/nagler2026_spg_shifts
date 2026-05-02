# Multidecadal North Atlantic Circulation Shifts under Historical Anthropogenic Forcing in CESM2-LE

This repository contains the analysis code for the manuscript:

> **"Multidecadal North Atlantic Circulation Shifts under Historical Anthropogenic Forcing in CESM2-LE"**
> Ina Nagler, Helene Asbjørnsen, Andreas Born.

The code prepares CESM2 large ensemble data, performs change point and composite analyses, and produces all figures in the manuscript.

---

## Repository Structure

```
.
├── Prepare_smoc55.ipynb                 # Loads and preprocesses SMOC at 55°N
├── Prepare_dens2.ipynb                  # Prepares sigma-2 density fields
├── Prepare_streamfunctions.ipynb        # Computes overturning streamfunctions
├── Prepare_averages.ipynb               # Calculates spatial/temporal averages
├── Prepare_change_point_analysis.ipynb  # Sets up and runs change point detection
├── Prepare_composite_analysis.ipynb     # Prepares composites around events
├── Plot_increase.ipynb                  # Figures for strengthening events
├── Plot_decrease_aa.ipynb               # Figures for aerosol-driven weakening events
├── Plot_decrease_ghg.ipynb              # Figures for GHG-driven weakening events
├── Plot_depth_density_anomalies.ipynb   # Depth-density anomaly profiles
├── Plot_map.ipynb                       # Geographic map figures
├── Plot_occurences.ipynb                # Frequency of change point events
├── Plot_schematic.ipynb                 # Conceptual schematic figures
├── Plot_smoc+dens_historical_trend.ipynb # Historical SMOC and density trends
├── change_point_analysis.py             # Core change point detection algorithm
├── composite_analysis.py                # Composite analysis around events
├── parallel_composite_analysis.py       # Parallelised composite analysis for HPC
├── composites_stf.py                    # Streamfunction composites
├── averages.py                          # Spatial/temporal averaging functions
├── standard_deviations.py               # Ensemble standard deviation computation
├── dens2.py                             # Sigma-2 density calculation (TEOS-10)
├── smoc55_timeseries.py                 # SMOC55 index time series extraction
├── ensemble_mean_temp_salt.py           # Ensemble mean temperature and salinity
├── rename_vars.py                       # NetCDF variable renaming utility
├── run_averages.sh                      # SLURM job script for averages
├── run_change_point.sh                  # SLURM job script for change point analysis
├── run_composite.sh                     # SLURM job script for composite analysis
├── run_standard_deviations.sh           # SLURM job script for standard deviations
├── run_stf.sh                           # SLURM job script for streamfunction composites
├── change_point_indices_*.csv           # Change point detection results
├── plotting-change_point_indices_*.csv  # Filtered results used by plotting notebooks
└── sensitivity_change_point/            # CSVs for sensitivity analysis
```

---

## Workflow
1. **Prepare** — Run `smoc55_timeseries.py` to extract the SMOC55 index time series
2. **Analyse** — Submit `run_change_point.sh` and `run_composite.sh` to detect change points and compute composites
3. **Plot** — Run `Plot_*.ipynb` notebooks to reproduce all manuscript figures

---

## Data

This code uses output from the [CESM2 Large Ensemble](https://www.cesm.ucar.edu/community-projects/lens2) (100 members). The raw model data is publicly available through the CESM2-LE project. Users should download the required fields and run the full workflow from step 1.

---

## Requirements

Install dependencies via conda:

```bash
conda create -n nagler2026_spg_shifts python=3.10
conda activate nagler2026_spg_shifts
conda install -c conda-forge numpy pandas xarray dask matplotlib cartopy gsw cftime pop_tools cmocean cmcrameri psutil jupyter
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
