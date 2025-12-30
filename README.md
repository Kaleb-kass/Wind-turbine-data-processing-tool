# Wind Turbine – Power vs RPM (GPR Optimization)

This project ingests aerodynamic blade data blocks from CSV, integrates torque to power, fits a Gaussian Process Regression (GPR) surrogate, finds the optimal RPM, and generates a suite of plots (power, Cl/Cd, AoA, Betz comparison). Interactive dialogs are included for file selection and RPM queries.

Main script: [rpm2.py](rpm2.py)

## What It Does
- Parses RPM blocks: a row whose first cell is `n` (next cell is the RPM), followed by a header row and data rows until a blank/non-numeric radius.
- Prefers Torque + Radius columns (integrates torque -> power); falls back to Power Integral columns. Optional: Cl, Cd, AoA.
- Cleans numeric values robustly (thousand separators, stray units, accounting parentheses, spaced scientific notation).
- Fits a GPR surrogate on `PowerIntegral(n)`, searches the maximum, and provides point predictions with 95% CI.
- Plots: Power vs RPM (confidence band), Cl/Cd vs Radius, AoA vs Radius, Cp vs Betz limit comparison.
- Interactive flows: Tk file picker (with console fallback), quick RPM popup, and results GUI.

## Requirements
- Python 3.10+ (tested on Windows)
- Core libs (see [requirements.txt](requirements.txt))
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - pillow (only if you load optional logo images in the GUIs)
- Optional: SciencePlots (styling; auto-fallback if missing)
- Tkinter: bundled with standard Python installers on Windows (needed for dialogs/GUI).

Install core dependencies with `pip install -r requirements.txt`.

## Quick Start (Windows / PowerShell)

```powershell
# 1) Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2) Upgrade pip (recommended)
python -m pip install --upgrade pip

# 3) Install core dependencies
pip install -r requirements.txt

# 4) Optional: nicer scientific styles
pip install SciencePlots

# 5) Run the main script (GUI file picker by default)
python rpm2.py

# OR: provide a CSV path directly and open the results window
python rpm2.py --csv "your-data.csv" --results
```

Outputs:
- Console: optimal RPM and max Power Integral (warns if extrapolating).
- Figures: `figures/power_vs_rpm.*`, `figures/cl_cd_vs_radius.*`, `figures/aoa_vs_radius.*`, `figures/betz_limit_comparison.*`.

### Common CLI examples

```powershell
# Predict at a given RPM (e.g., 450 rpm) and skip plotting
python rpm2.py --csv "your-data.csv" --rpm 450 --no-plot

# Launch a small GUI to query RPM interactively
python rpm2.py --csv "your-data.csv" --gui

# Show a minimal popup for a single RPM query
python rpm2.py --csv "your-data.csv" --popup

# Show the results window (optimal RPM, kernel, stats)
python rpm2.py --csv "your-data.csv" --results
```

## Data File
- Default: select via Tk file picker; or pass `--csv path/to/file.csv`.
- Block structure:
  - A row whose first cell is `n`; the next cell is the RPM value.
  - The following row is a header row.
  - Preferred: a Torque column ("torque", "torq", "moment", etc.) plus a Radius column.
  - Fallback: a Power Integral column ("power integral", "power", etc.).
  - Optional columns: Radius (if not first col), Cl, Cd, AoA.
  - Data rows continue until a blank or non-numeric radius.

## Optional Scripts
- [stoke.py](stoke.py): simple animated map (requires Cartopy; not needed for rpm2 workflow).

## Troubleshooting
- ModuleNotFoundError: `pip install -r requirements.txt` (and `pip install SciencePlots` if desired).
- CSV not found: ensure the path is correct or pass `--csv` explicitly.
- GUI not showing: verify Tkinter works:
  ```powershell
  python -c "import tkinter as tk; print('tk OK'); tk.Tk().destroy()"
  ```
  If it fails, reinstall Python with Tcl/Tk enabled.
- Style issues: SciencePlots is optional; the script falls back automatically.
- Cartopy errors: only relevant for [stoke.py](stoke.py); consider Conda/wheels if needed.

## Reproducibility
- GPR uses a fixed `random_state` for optimizer restarts to make results reproducible given the same data file.

## Notes
- Figures are saved automatically in `figures/`. PDF is suitable for reports; PNG is 300 DPI. The plots include minor ticks, scientific notation on y, confidence bands, and a clearly labeled optimum.
