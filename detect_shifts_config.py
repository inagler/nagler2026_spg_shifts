# detect_shifts_config.py
# Single source of truth for all shift detection parameters and paths.
# Used by detect_shifts.py for both historical members and piControl.
# See REVISION_PLAN.md Section 2 for the full method definition.

# ---------------------------------------------------------------------------
# Window lengths (years)
# ---------------------------------------------------------------------------
L_PRE  = 40   # pre-shift baseline window length
L_POST = 20   # post-shift test window length

# ---------------------------------------------------------------------------
# Detection thresholds to run
# ---------------------------------------------------------------------------
# Main analysis uses k=2 for weakening_early, k=3 for all other categories.
# Both are run here; the correct k per category is applied at output stage.
K_VALUES = [2, 3]

# ---------------------------------------------------------------------------
# Separation parameter (years)
# ---------------------------------------------------------------------------
# D = L_POST: same-sign shifts in one member must be at least D years apart.
# Greedy: accept largest |η| first; discard any candidate within D of accepted.
D = L_POST  # = 20

# ---------------------------------------------------------------------------
# Detection end year (inclusive, calendar year)
# ---------------------------------------------------------------------------
# Historical members are truncated to this year before detection.
# piControl uses all available model years (no calendar truncation applied).
END_YEAR = 2039

# ---------------------------------------------------------------------------
# Category boundaries (calendar years)
# ---------------------------------------------------------------------------
# Weakening split: yr_s < CATEGORY_BOUNDARY → weakening_early
#                  yr_s ≥ CATEGORY_BOUNDARY → weakening_late
CATEGORY_BOUNDARY = 1985

# Strengthening sub-split year for RC3.3 (set from histogram of yr_s).
# None = do not sub-split; assign all strengthening shifts category 'strengthening'.
# Update this value once the histogram has been inspected.
YR_SPLIT = None

# ---------------------------------------------------------------------------
# Input paths
# ---------------------------------------------------------------------------
# Historical member timeseries (monthly SMOC55, written by smoc55_timeseries.py)
SMOC_DIR = '/Data/skd/scratch/innag3580/comp/smoc/'

# piControl timeseries (monthly smoc55_index, written by smoc55_picontrol.py)
PICONTROL_FILE = '/Data/skd/scratch/innag3580/comp/smoc55_picontrol/smoc55_picontrol.nc'

# Existing v1.0 change point CSVs (for comparison counts only; never overwritten)
EXISTING_CSV_DIR = '/shared_netapp3_home/home/innag3580/phase1_CONDA/publishable_code/'

# ---------------------------------------------------------------------------
# Output path (revision directory; never overlaps with v1.0 outputs)
# ---------------------------------------------------------------------------
REVISION_OUTPUT_DIR = '/Data/skd/scratch/innag3580/comp/revision/shifts/'
