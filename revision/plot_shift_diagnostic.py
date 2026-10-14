#!/usr/bin/env python
"""Diagnostic plot: all SMOC55 η exceedances vs detected shifts."""

import os
import pandas as pd
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = "/Data/skd/scratch/innag3580/comp/revision"
SH_DIR  = os.path.join(BASE, "shifts")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

eta_path = os.path.join(SH_DIR, "eta_hist.csv")
k2_path  = os.path.join(SH_DIR, "shifts_k2_hist.csv")
k3_path  = os.path.join(SH_DIR, "shifts_k3_hist.csv")

# ── Load data ─────────────────────────────────────────────────────────────────
eta    = pd.read_csv(eta_path)   # mem, yr, eta
k2     = pd.read_csv(k2_path)    # mem, yr_s, sign, eta, cluster_start, cluster_end, category
k3     = pd.read_csv(k3_path)

# Combine k2 and k3, drop duplicates (same mem+yr_s may appear in both)
shifts = pd.concat([k2, k3]).drop_duplicates(subset=["mem", "yr_s"])

# ── Print counts ──────────────────────────────────────────────────────────────
print("Shift counts per category (combined k2+k3, deduped):")
for cat, grp in shifts.groupby("category"):
    print(f"  {cat}: {len(grp)} shifts across {grp['mem'].nunique()} members")
print(f"  Total: {len(shifts)}")

# ── Colours per category ──────────────────────────────────────────────────────
cat_colors = {
    "strengthening":   "#2166ac",  # blue
    "weakening_early": "#d6604d",  # red-orange
    "weakening_late":  "#762a83",  # purple
}

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))

# Background: all η values
ax.scatter(eta["yr"], eta["eta"],
           color="grey", s=4, alpha=0.25, linewidths=0, zorder=1,
           label="all exceedances (η)")

# Detected shifts, one colour per category
for cat, color in cat_colors.items():
    sub = shifts[shifts["category"] == cat]
    if sub.empty:
        continue
    ax.scatter(sub["yr_s"], sub["eta"],
               color=color, s=40, alpha=0.85, linewidths=0.4,
               edgecolors="k", zorder=3, label=f"{cat} (n={len(sub)})")

# Horizontal dashed threshold lines
for level, ls in [(2, "--"), (-2, "--"), (3, ":"), (-3, ":")]:
    ax.axhline(level, color="black", linewidth=0.8, linestyle=ls, zorder=2)
    ax.text(1851, level + 0.07, f"η = {level:+d}",
            fontsize=7, va="bottom", color="black")

# Axes
ax.set_xlabel("Year")
ax.set_ylabel("η (normalised shift magnitude, s.d.)")
ax.set_title("Shift detection diagnostic: all exceedances vs detected shifts")
ax.set_xlim(1850, 2030)
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
ax.grid(True, linewidth=0.3, alpha=0.4)

fig.tight_layout()
out = os.path.join(FIG_DIR, "shift_detection_diagnostic.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}")
