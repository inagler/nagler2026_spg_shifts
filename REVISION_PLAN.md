# REVISION_PLAN.md

## Section 1: Analysis plan

1. Shift detection
   - Implement method in section 2, for historical members and piControl
   - Avoid double counting via clustering and separation rule
   - Histogram of strengthening shift years to set yr_split
2. Composites
   - One file per variable storing all shift windows (TEMP, SALT, VVEL, N_HEAT, SHF, HMXL, sea ice)
   - One file producing exactly the data to plot
3. Recreate all existing metrics and maps
   - Fix CT_from_pt bug in historical trend figure
4. Statistics
   - Bootstrap all composites for confidence intervals on each composite pattern (RC3.1, RC3.2)
   - Permutation test A: compare against other real shifts to test whether categories are distinguishable (RC3.1)
   - Permutation test B: random samples of non-shift periods matched in sample size (RC3.1)
   - Significance for maps (stippling) and time series (confidence intervals) (RC3.2)
   - Surrogate approach (AR1 and/or piControl block resampling) to test whether the rate of detected shifts per category exceeds red noise (RC2.6, RC3.1)
5. New metrics
   - Density gradient between boundary and interior of the SPG (RC1.4)
   - Meridional density gradient between STG and SPG (RC1.4, RC2.9)
   - Stratification metric (RC2.8)
6. New maps
   - Stratification instead of density (RC2.8)
7. Lead-lag analysis between SMOC55 and MOC26 (RC2.7)
8. piControl check (RC3.1)
9. Check that 1895 and 1955 strengthening sub-composites agree (RC3.3)
10. Check non-stationarity of post-1985 shifts (RC3.3)
11. Supplementary sea ice composite for post-1985 weakening shifts (RC3.9)

---

## Section 2: Shift detection method

Input: Ψ_s(mem, yr), annual mean SMOC55 maximum, member mem, year yr. No detrending.
Windows: pre-shift [yr − L_pre, yr − 1]; post-shift [yr, yr + L_post − 1]; shift window L_pre + L_post. L_pre = 40, L_post = 20.
Normalised shift magnitude: η(mem, yr) = (mean_post − mean_pre) / std_pre.
Detection, run separately for k = 2 and k = 3:
- Exceedance: strengthening if η ≥ k, weakening if η ≤ −k.
- Cluster: run of consecutive same-sign exceedance years in one member.
- Shift year yr_s: year of maximum |η| within a cluster.
- Separation: same-sign shift years in one member at least D = L_post apart. Greedy: accept largest |η| first, discard any within D of an accepted shift, repeat.
- Strengthening and weakening shifts do not suppress each other.
Categories: strengthening (sub-split at yr_split, set from histogram, for RC3.3); weakening yr_s < 1985; weakening yr_s ≥ 1985.
Main analysis: k = 2 for weakening yr_s < 1985; k = 3 for all other categories. Other combinations are a sensitivity check.
Detection end year: 2029.
piControl: identical procedure.
