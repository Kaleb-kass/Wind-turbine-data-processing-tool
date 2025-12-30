# type: ignore[reportOptionalMemberAccess, reportOptionalAttributeAccess]
import re
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter
import os
import sys
import argparse
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    HAS_TK = True
except Exception:
    HAS_TK = False
    tk = None  # type: ignore
    ttk = None  # type: ignore
    messagebox = None  # type: ignore

BLADE_MODELS = [
    "E176 Truncated", "E63 Truncated", "S1091 Truncated", "S2091 Truncated", "FX63-100 Truncated", "E174 Truncated", "E193 Truncated", "SD7034 Truncated",
    "S9000 Truncated",
]

# ---------- CLI (early) ----------
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--csv", "-c", type=str, default=None, help="Path to input CSV file")
parser.add_argument("--rpm", type=float, default=None, help="Query predicted Power Integral at this RPM")
parser.add_argument("--no-plot", action="store_true", help="Skip figure generation and showing plots")
parser.add_argument("--gui", action="store_true", help="Launch a small GUI to query RPM interactively")
parser.add_argument("--popup", action="store_true", help="Show a small popup to enter an RPM and get predicted power")
parser.add_argument("--results", action="store_true", help="Show a results window with key outputs")
args, unknown = parser.parse_known_args()

# ---------- Helpers ----------
def to_float_safe(x):
    """Try to convert x to float after cleaning typical formatting issues.
    Returns None if conversion fails."""
    if pd.isna(x):
        return None
    if isinstance(x, (int, float, np.floating, np.integer)):
        try:
            return float(x)
        except Exception:
            return None
    s = str(x).strip()
    if s == "":
        return None
    # remove thousand-separators like '1,234' -> '1234' but keep scientific notation commas (rare)
    # remove parentheses (e.g. accounting negatives) and common units
    s = s.replace("(", "-").replace(")", "")
    # remove common unit strings (W, kW, RPM, etc.) if present at the end
    s = re.sub(r"[A-Za-z/%°\s]+$", "", s)
    s = s.replace(",", "")  # remove commas after unit removal
    # final cleanup
    s = s.strip()
    try:
        return float(s)
    except Exception:
        # try to salvage exponential forms with spaces: "1.2 e3"
        try:
            s2 = s.replace(" e", "e").replace("E ", "E")
            return float(s2)
        except Exception:
            return None

def header_matches_power(h):
    """Return True if header cell h indicates a Power Integral column."""
    if not isinstance(h, str):
        return False
    hh = h.strip().lower()
    # common variants
    keywords = ["power integral", "power_integral", "powerint", "p_int", "p int", "power"]
    for kw in keywords:
        if kw in hh:
            return True
    return False

def header_matches_torque(h):
    """Return True if header cell h indicates a Torque column."""
    if not isinstance(h, str):
        return False
    hh = h.strip().lower()
    # common variants
    keywords = ["torque", "torq", "moment", "m [nm]", "m[nm]"]
    for kw in keywords:
        if kw in hh:
            return True
    return False

def header_matches_radius(h):
    """Return True if header cell h indicates a radius/radial column."""
    if not isinstance(h, str):
        return False
    hh = h.strip().lower()
    # common variants
    keywords = ["radius", "radial", "r [m]", "r[m]", "r (m)"]
    for kw in keywords:
        if kw in hh:
            return True
    return False

def header_matches_cl(h):
    """Return True if header cell h indicates a Lift Coefficient (Cl) column."""
    if not isinstance(h, str):
        return False
    hh = h.strip().lower()
    keywords = ["cl", "c_l", "lift coeff", "lift coefficient"]
    for kw in keywords:
        if kw in hh:
            return True
    return False

def header_matches_cd(h):
    """Return True if header cell h indicates a Drag Coefficient (Cd) column."""
    if not isinstance(h, str):
        return False
    hh = h.strip().lower()
    keywords = ["cd", "c_d", "drag coeff", "drag coefficient"]
    for kw in keywords:
        if kw in hh:
            return True
    return False

def header_matches_aoa(h):
    """Return True if header cell h indicates an Angle of Attack (AoA) column."""
    if not isinstance(h, str):
        return False
    hh = h.strip().lower()
    keywords = ["angle of attack", "aoa", "alpha", "a.o.a", "a o a", "angle", "attack"]
    for kw in keywords:
        if kw in hh:
            return True
    return False


# ---------- GUI blade metadata ----------
def prompt_blade_metadata():
    """Popup to collect blade model + parameters before CSV selection."""
    if not HAS_TK:
        return None

    result = {"model": None, "diameter": None, "air_density": None}

    root = tk.Tk()
    root.title("Blade configuration")
    try:
        root.minsize(460, 360)
    except Exception:
        pass
    try:
        root.lift()
        root.attributes('-topmost', True)
        root.after(250, lambda: root.attributes('-topmost', False))
    except Exception:
        pass

    frm = ttk.Frame(root, padding=16)
    frm.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    row_offset = 0
    logo_img = load_logo()
    logo_img2 = load_second_logo()
    if logo_img or logo_img2:
        logo_frame = ttk.Frame(frm)
        logo_frame.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        if logo_img2:
            logo_lbl2 = ttk.Label(logo_frame, image=logo_img2)
            logo_lbl2.image = logo_img2
            logo_lbl2.grid(row=0, column=0, padx=(0, 10))
        if logo_img:
            logo_lbl = ttk.Label(logo_frame, image=logo_img)
            logo_lbl.image = logo_img
            logo_lbl.grid(row=0, column=1, padx=(10, 0))
        row_offset = 1

    ttk.Label(frm, text="Blade model").grid(row=row_offset, column=0, sticky="w")
    model_var = tk.StringVar(value=BLADE_MODELS[0])
    model_box = ttk.Combobox(frm, textvariable=model_var, values=BLADE_MODELS, state="readonly", width=18)
    model_box.grid(row=row_offset, column=1, sticky="we", padx=(8, 0))

    ttk.Label(frm, text="Blade diameter [m]").grid(row=row_offset+1, column=0, sticky="w", pady=(8, 0))
    dia_var = tk.StringVar(value="100.0")
    dia_entry = ttk.Entry(frm, textvariable=dia_var, width=18)
    dia_entry.grid(row=row_offset+1, column=1, sticky="we", padx=(8, 0), pady=(8, 0))

    ttk.Label(frm, text="Air density [kg/m³]").grid(row=row_offset+2, column=0, sticky="w", pady=(8, 0))
    rho_var = tk.StringVar(value="1.225")
    rho_entry = ttk.Entry(frm, textvariable=rho_var, width=18)
    rho_entry.grid(row=row_offset+2, column=1, sticky="we", padx=(8, 0), pady=(8, 0))

    ttk.Label(frm, text="Wind speed Vin [m/s]").grid(row=row_offset+3, column=0, sticky="w", pady=(8, 0))
    vin_var = tk.StringVar(value="8.0")
    vin_entry = ttk.Entry(frm, textvariable=vin_var, width=18)
    vin_entry.grid(row=row_offset+3, column=1, sticky="we", padx=(8, 0), pady=(8, 0))

    status_var = tk.StringVar(value="")
    status_lbl = ttk.Label(frm, textvariable=status_var, foreground="#cc0000")
    status_lbl.grid(row=row_offset+4, column=0, columnspan=2, sticky="w", pady=(6, 0))

    def finish(ok: bool):
        if not ok:
            root.destroy()
            return
        try:
            dia = float(dia_var.get().strip())
            rho = float(rho_var.get().strip())
            vin = float(vin_var.get().strip())
        except Exception:
            status_var.set("Please enter numeric values for diameter, air density, and wind speed.")
            return
        result.update({"model": model_var.get(), "diameter": dia, "air_density": rho, "wind_speed": vin})
        root.destroy()

    btn_frame = ttk.Frame(frm)
    btn_frame.grid(row=row_offset+5, column=0, columnspan=2, pady=(12, 0), sticky="e")
    ttk.Button(btn_frame, text="Cancel", command=lambda: finish(False)).grid(row=0, column=0, padx=(0, 8))
    ttk.Button(btn_frame, text="OK", command=lambda: finish(True)).grid(row=0, column=1)

    dia_entry.focus_set()
    root.bind('<Return>', lambda e: finish(True))
    root.bind('<Escape>', lambda e: finish(False))
    root.mainloop()

    if result["model"] is None:
        return None
    return result


def format_blade_tag(meta):
    """Render a short label for blade metadata."""
    if not meta or not meta.get("model"):
        return "Blade: not set"
    tag = f"Blade {meta['model']}"
    dia = meta.get("diameter")
    rho = meta.get("air_density")
    if dia is not None:
        tag += f" | D={dia:g} m"
    if rho is not None:
        tag += f" | ρ={rho:g} kg/m^3"
    return tag


def load_second_logo(logo_path="logo2.png", max_width=140):
    """Load and return a second tkinter PhotoImage for the logo, or None if unavailable."""
    if not HAS_TK or not os.path.exists(logo_path):
        return None
    try:
        from PIL import Image, ImageTk
        img = Image.open(logo_path)
        # Resize if needed
        if img.width > max_width:
            ratio = max_width / img.width
            new_height = int(img.height * ratio)
            img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(img)
    except Exception:
        # Fallback to native tk (supports GIF/PPM/PGM)
        try:
            photo = tk.PhotoImage(file=logo_path)
            # Simple subsample if too large
            if photo.width() > max_width:
                factor = photo.width() // max_width + 1
                photo = photo.subsample(factor, factor)
            return photo
        except Exception:
            return None
        

def load_logo(logo_path="logo.png", max_width=140):
    """Load and return a tkinter PhotoImage for the logo, or None if unavailable."""
    if not HAS_TK or not os.path.exists(logo_path):
        return None
    try:
        from PIL import Image, ImageTk
        img = Image.open(logo_path)
        # Resize if needed
        if img.width > max_width:
            ratio = max_width / img.width
            new_height = int(img.height * ratio)
            img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(img)
    except Exception:
        # Fallback to native tk (supports GIF/PPM/PGM)
        try:
            photo = tk.PhotoImage(file=logo_path)
            # Simple subsample if too large
            if photo.width() > max_width:
                factor = photo.width() // max_width + 1
                photo = photo.subsample(factor, factor)
            return photo
        except Exception:
            return None

def compute_ideal_power(meta):
    """Return ideal power (W) using 0.5 * A * rho * Vin^3, or None if missing."""
    try:
        d = float(meta.get("diameter"))
        rho = float(meta.get("air_density"))
        vin = float(meta.get("wind_speed"))
    except Exception:
        return None
    if d <= 0 or rho <= 0 or vin is None or vin < 0:
        return None
    area = np.pi * (d / 2.0) ** 2
    return 0.5 * area * rho * (vin ** 3)


def format_ideal_power(meta):
    p = compute_ideal_power(meta) if meta else None
    if p is None:
        return "Ideal P: n/a"
    vin = meta.get("wind_speed")
    return f"Ideal P: {p:.3f} W at Vin={vin} m/s"

# ---------- 1. Load raw CSV ----------
# Determine CSV path: CLI arg, then file dialog (if available), then prompt fallback
csv_path = args.csv
blade_metadata = None
ideal_power_val = None

if csv_path is None:
    if HAS_TK:
        blade_metadata = prompt_blade_metadata()
        if blade_metadata is None:
            print("Blade selection cancelled - exiting.")
            sys.exit(1)
        ideal_power_val = compute_ideal_power(blade_metadata)
        print(
            f"Selected blade model: {blade_metadata['model']}, "
            f"Diameter: {blade_metadata['diameter']} m, "
            f"Air density: {blade_metadata['air_density']} kg/m^3, "
            f"Wind speed: {blade_metadata.get('wind_speed', 'n/a')} m/s"
        )
        if ideal_power_val is not None:
            print(f"Ideal power (0.5*A*rho*Vin^3): {ideal_power_val:.3f} W")
    if HAS_TK:
        try:
            from tkinter import filedialog as _fd
            _root = tk.Tk()
            _root.withdraw()
            csv_path = _fd.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv"), ("All files", "*")])
            _root.destroy()
        except Exception:
            csv_path = None
    else:
        try:
            csv_path = input("Enter path to CSV file: ").strip()
        except Exception:
            csv_path = None

if not csv_path:
    print("No CSV file selected - exiting.")
    sys.exit(1)

raw = pd.read_csv(csv_path, header=None, dtype=object)  # read as object to preserve text cells

# ---------- 2. Parse RPM blocks and Power Integral ----------
rpm_list = []
pint_list = []
rpm_cl_cd_data = {}  # Store Cl/Cd data for each RPM: {rpm_val: [(radius, cl, cd), ...]}
rpm_aoa_data = {}    # Store AoA data for each RPM: {rpm_val: [(radius, aoa), ...]}

n_rows = len(raw)
for idx in range(n_rows):
    val = raw.iloc[idx, 0]
    if isinstance(val, str) and val.strip().lower() == "n":
        # read RPM from next column
        rpm_cell = raw.iloc[idx, 1] if 1 < raw.shape[1] else None
        rpm_val = to_float_safe(rpm_cell)
        if rpm_val is None:
            print(f"Skipping block at row {idx}: couldn't parse RPM from cell '{rpm_cell}'")
            continue

        header_row = idx + 1
        if header_row >= n_rows:
            print(f"Skipping block at row {idx}: header row out-of-bounds")
            continue

        header_vals = raw.iloc[header_row].tolist()
        header = [h.strip() if isinstance(h, str) else h for h in header_vals]

        # Try to find Torque column first (new approach)
        torque_cols = []
        for j, h in enumerate(header):
            if header_matches_torque(h):
                torque_cols.append((j, h))
        
        # Find radius column
        radius_cols = []
        for j, h in enumerate(header):
            if header_matches_radius(h):
                radius_cols.append((j, h))
        
        # Find Cl and Cd columns
        cl_cols = []
        cd_cols = []
        aoa_cols = []
        for j, h in enumerate(header):
            if header_matches_cl(h):
                cl_cols.append((j, h))
            if header_matches_cd(h):
                cd_cols.append((j, h))
            if header_matches_aoa(h):
                aoa_cols.append((j, h))
        
        # If no explicit radius column, assume first column is radius
        if not radius_cols:
            radius_col_idx = 0
        else:
            radius_col_idx = radius_cols[0][0]
        
        # Determine if we have Torque or Power Integral
        if torque_cols:
            # Use Torque and integrate
            torque_col_idx = torque_cols[0][0]
            cl_col_idx = cl_cols[0][0] if cl_cols else None
            cd_col_idx = cd_cols[0][0] if cd_cols else None
            aoa_col_idx = aoa_cols[0][0] if aoa_cols else None
            
            # collect radius, torque, Cl, Cd and AoA data
            row = header_row + 1
            block_radius = []
            block_torque = []
            block_cl = []
            block_cd = []
            block_aoa = []
            while row < n_rows:
                first = raw.iloc[row, radius_col_idx]
                if pd.isna(first):
                    break
                # expect numeric radius
                r_float = to_float_safe(first)
                if r_float is None:
                    break
                
                val_t = raw.iloc[row, torque_col_idx] if torque_col_idx < raw.shape[1] else None
                t_float = to_float_safe(val_t)
                if t_float is not None:
                    block_radius.append(r_float)
                    block_torque.append(t_float)
                    
                    # Also collect Cl, Cd and AoA if available
                    if cl_col_idx is not None:
                        cl_val = to_float_safe(raw.iloc[row, cl_col_idx])
                        if cl_val is not None:
                            block_cl.append(cl_val)
                        else:
                            block_cl.append(None)
                    if cd_col_idx is not None:
                        cd_val = to_float_safe(raw.iloc[row, cd_col_idx])
                        if cd_val is not None:
                            block_cd.append(cd_val)
                        else:
                            block_cd.append(None)
                    if aoa_col_idx is not None:
                        aoa_val = to_float_safe(raw.iloc[row, aoa_col_idx])
                        if aoa_val is not None:
                            block_aoa.append(aoa_val)
                        else:
                            block_aoa.append(None)
                row += 1
            
            if len(block_torque) < 2:
                print(f"Skipping block (RPM={rpm_val}): insufficient torque data for integration (need at least 2 points)")
                continue
            
            # Store Cl/Cd data for plotting
            if block_cl and block_cd and len(block_cl) == len(block_cd) == len(block_radius):
                rpm_cl_cd_data[rpm_val] = list(zip(block_radius, block_cl, block_cd))
            
            # Store AoA data for plotting
            if block_aoa and len(block_aoa) == len(block_radius):
                rpm_aoa_data[rpm_val] = list(zip(block_radius, block_aoa))
            
            # Apply trapezoidal integration to get torque integral
            torque_integral = np.trapz(block_torque, block_radius)
            
            # Convert to Power using P = Torque × ω, where ω = 2π × n / 60
            omega = 2 * np.pi * rpm_val / 60.0  # rad/s
            P_int_block = torque_integral * omega
            
            print(f"Block RPM={rpm_val}: Integrated Torque={torque_integral:.6g}, Power={P_int_block:.6g} W")
        
        else:
            # Fall back to Power Integral column (original behavior)
            power_cols = []
            for j, h in enumerate(header):
                if header_matches_power(h):
                    power_cols.append((j, h))
            if not power_cols:
                print(f"Skipping block (RPM={rpm_val}): no header column matching 'Torque' or 'Power Integral' in header row {header_row}")
                continue
            power_col_idx = power_cols[0][0]

            # collect numeric rows starting at header_row+1 until break condition
            row = header_row + 1
            block_powers = []
            while row < n_rows:
                first = raw.iloc[row, 0]
                if pd.isna(first):
                    break
                # we expect the first column of data rows to be numeric radius; if not numeric, stop
                if to_float_safe(first) is None:
                    break

                val_p = raw.iloc[row, power_col_idx] if power_col_idx < raw.shape[1] else None
                p_float = to_float_safe(val_p)
                if p_float is not None:
                    block_powers.append(p_float)
                row += 1

            if len(block_powers) == 0:
                print(f"Skipping block (RPM={rpm_val}): no numeric power-integral data found")
                continue

            # pick last-nonzero as total if present, else max
            last_nonzero = None
            for v in reversed(block_powers):
                if v is not None and v != 0:
                    last_nonzero = v
                    break
            P_int_block = last_nonzero if last_nonzero is not None else max(block_powers)

        rpm_list.append(rpm_val)
        pint_list.append(P_int_block)

# Build dataframe
if len(rpm_list) == 0:
    raise RuntimeError("No RPM / PowerIntegral blocks parsed. Check CSV structure and matching logic.")

df_blocks = pd.DataFrame({"n": rpm_list, "PowerIntegral": pint_list})
df_blocks = df_blocks.sort_values("n").reset_index(drop=True)

# Add Cp column if ideal power is available
if ideal_power_val is not None and ideal_power_val > 0:
    df_blocks["Cp"] = df_blocks["PowerIntegral"] / ideal_power_val
    print("Parsed blocks:")
    print(df_blocks)
else:
    print("Parsed blocks:")
    print(df_blocks)
    print("(Cp not calculated - ideal power unavailable)")

# ---------- 3. GPR surrogate P_int(n) ----------
X = df_blocks[["n"]].values.astype(float)
y = df_blocks["PowerIntegral"].values.astype(float)

# scaling X
sc_X = StandardScaler().fit(X)
X_s = sc_X.transform(X)

# scaling y with guard for zero-variance
y_var = np.var(y)
use_y_scaler = True
if y_var == 0 or np.isclose(y_var, 0.0):
    print("Warning: y has zero variance. Skipping StandardScaler for y.")
    y_s = y.copy()
    sc_y = None
    use_y_scaler = False
else:
    sc_y = StandardScaler().fit(y.reshape(-1, 1))
    y_s = sc_y.transform(y.reshape(-1, 1)).ravel()

# GPR model with wider bounds and more restarts
kernel = C(1.0, (1e-4, 1e4)) * RBF(1.0, (1e-3, 1e3))
gpr = GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-6,
    n_restarts_optimizer=25,  # more restarts for better convergence
    normalize_y=False,
    optimizer='fmin_l_bfgs_b',
    random_state=42
)
gpr.fit(X_s, y_s)
print("Trained GPR kernel:", gpr.kernel_)

# ---------- 4. Evaluate + optimize ----------
# Build a dense RPM grid; GPR predicts smooth power and we search its maximum.
n_min, n_max = X.min(), X.max()
# Use dense grid (2000 points) for both plotting and optimization
n_grid = np.linspace(n_min, n_max, 2000).reshape(-1, 1)
n_grid_s = sc_X.transform(n_grid)

y_pred_s, y_std = gpr.predict(n_grid_s, return_std=True)

if use_y_scaler:
    y_pred = sc_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
    # y_std is in scaled units; convert to original units
    y_std_real = y_std * sc_y.scale_[0]
else:
    y_pred = y_pred_s
    y_std_real = y_std  # already in original units if no scaling used

idx_opt = np.argmax(y_pred)
n_opt = n_grid[idx_opt, 0]
P_opt = y_pred[idx_opt]

print(f"\nOptimal RPM (GPR surrogate): {n_opt:.2f}")
print(f"Max Power Integral (approx): {P_opt:.6g}")

# Helper to predict power at an arbitrary RPM
def predict_power_at(n_value):
    # Transform query RPM through scalers, predict mean/std in original units
    n_arr = np.array([[float(n_value)]])
    n_s = sc_X.transform(n_arr)
    y_pred_s, y_std_s = gpr.predict(n_s, return_std=True)
    if use_y_scaler:
        y_pred_v = sc_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()[0]
        y_std_v = float(y_std_s[0] * sc_y.scale_[0])
    else:
        y_pred_v = float(y_pred_s.ravel()[0])
        y_std_v = float(y_std_s[0])
    return y_pred_v, y_std_v

# ---------- 4.1 CLI arguments ----------
if args.rpm is not None:
    pred, sigma = predict_power_at(args.rpm)
    print(f"\nQuery RPM: {args.rpm:.2f}")
    print(f"Predicted Power Integral: {pred:.6g} (±{2*sigma:.3g} at 95% CI)")
    if args.rpm < n_min or args.rpm > n_max:
        print("Note: RPM is outside training range; prediction is extrapolated.")

# ---------- 4.2 Minimal GUI ----------
def launch_gui():
    if not HAS_TK:
        print("Tkinter is not available in this Python environment.")
        print("Please use an official Python installer with Tcl/Tk enabled, or run without --gui.")
        sys.exit(1)
    root = tk.Tk()
    root.title("Power vs RPM (GPR)")
    try:
        root.minsize(520, 420)
    except Exception:
        pass
    # Bring window to front
    try:
        root.lift()
        root.attributes('-topmost', True)
        root.after(250, lambda: root.attributes('-topmost', False))
    except Exception:
        pass

    main = ttk.Frame(root, padding=16)
    main.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    row_offset = 0
    logo_img = load_logo()
    logo_img2 = load_second_logo()
    if logo_img or logo_img2:
        logo_frame = ttk.Frame(main)
        logo_frame.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        if logo_img2:
            logo_lbl2 = ttk.Label(logo_frame, image=logo_img2)
            logo_lbl2.image = logo_img2
            logo_lbl2.grid(row=0, column=0, padx=(0, 10))
        if logo_img:
            logo_lbl = ttk.Label(logo_frame, image=logo_img)
            logo_lbl.image = logo_img
            logo_lbl.grid(row=0, column=1, padx=(10, 0))
        row_offset = 1

    ttk.Label(main, text=format_blade_tag(blade_metadata), font=("Segoe UI", 9, "italic")).grid(row=row_offset, column=0, columnspan=3, sticky="w", pady=(0, 6))

    ideal_text = format_ideal_power(blade_metadata)
    ttk.Label(main, text=ideal_text, font=("Segoe UI", 9)).grid(row=row_offset+1, column=0, columnspan=3, sticky="w", pady=(0, 6))

    ttk.Label(main, text="Enter RPM").grid(row=row_offset+2, column=0, sticky="w")
    rpm_var = tk.StringVar()
    rpm_entry = ttk.Entry(main, textvariable=rpm_var, width=18)
    rpm_entry.grid(row=row_offset+2, column=1, sticky="we", padx=(8, 0))

    range_lbl = ttk.Label(main, text=f"Training range: {n_min:.1f} – {n_max:.1f} rpm")
    range_lbl.grid(row=row_offset+3, column=0, columnspan=2, sticky="w", pady=(4, 6))

    out_var = tk.StringVar(value="Prediction will appear here…")
    out_lbl = ttk.Label(main, textvariable=out_var, font=("Segoe UI", 9))
    out_lbl.grid(row=row_offset+4, column=0, columnspan=2, sticky="w", pady=(6, 4))

    opt_var = tk.StringVar(value=f"Optimal n*: {n_opt:.2f} rpm | P*: {P_opt:.4g} W")
    opt_lbl = ttk.Label(main, textvariable=opt_var)
    opt_lbl.grid(row=row_offset+5, column=0, columnspan=2, sticky="w")

    def do_predict():
        txt = rpm_var.get().strip()
        try:
            n_val = float(txt)
        except Exception:
            messagebox.showerror("Invalid input", "Please enter a numeric RPM value.")
            return
        p, s = predict_power_at(n_val)
        ci95 = 2 * s
        msg = f"RPM {n_val:.2f} → P̂ = {p:.6g} W (±{ci95:.3g})"
        if n_val < n_min or n_val > n_max:
            msg += "  [extrapolated]"
        out_var.set(msg)

    btn = ttk.Button(main, text="Predict", command=do_predict)
    btn.grid(row=row_offset+2, column=2, sticky="w", padx=(8, 0))

    rpm_entry.focus_set()
    root.bind('<Return>', lambda e: do_predict())
    root.resizable(False, False)
    root.mainloop()

if args.gui:
    print("Launching GUI...", flush=True)
    launch_gui()
    sys.exit(0)

# ---------- 4.3 Small RPM popup ----------
def launch_rpm_popup():
    if not HAS_TK:
        print("Tkinter is not available in this Python environment. Cannot show RPM popup.")
        return
    win = tk.Tk()
    win.title("Quick RPM Query")
    try:
        win.minsize(440, 320)
    except Exception:
        pass
    try:
        win.lift()
        win.attributes('-topmost', True)
        win.after(200, lambda: win.attributes('-topmost', False))
    except Exception:
        pass

    frm = ttk.Frame(win, padding=14)
    frm.grid(row=0, column=0)

    row_offset = 0
    logo_img = load_logo()
    logo_img2 = load_second_logo()
    if logo_img or logo_img2:
        logo_frame = ttk.Frame(frm)
        logo_frame.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        if logo_img2:
            logo_lbl2 = ttk.Label(logo_frame, image=logo_img2)
            logo_lbl2.image = logo_img2
            logo_lbl2.grid(row=0, column=0, padx=(0, 10))
        if logo_img:
            logo_lbl = ttk.Label(logo_frame, image=logo_img)
            logo_lbl.image = logo_img
            logo_lbl.grid(row=0, column=1, padx=(10, 0))
        row_offset = 1

    ttk.Label(frm, text=format_blade_tag(blade_metadata), font=("Segoe UI", 9, "italic")).grid(row=row_offset, column=0, columnspan=3, sticky="w")
    ttk.Label(frm, text=format_ideal_power(blade_metadata), font=("Segoe UI", 9)).grid(row=row_offset+1, column=0, columnspan=3, sticky="w")
    ttk.Label(frm, text=f"Training range: {n_min:.1f} – {n_max:.1f} rpm").grid(row=row_offset+2, column=0, columnspan=3, sticky="w", pady=(4, 0))
    ttk.Label(frm, text="Enter RPM:").grid(row=row_offset+3, column=0, sticky="w", pady=(6,0))
    rpm_val = tk.StringVar()
    ent = ttk.Entry(frm, textvariable=rpm_val, width=18)
    ent.grid(row=row_offset+3, column=1, sticky="w", pady=(6,0))

    res_var = tk.StringVar(value="Prediction will appear here…")
    res_lbl = ttk.Label(frm, textvariable=res_var)
    res_lbl.grid(row=row_offset+4, column=0, columnspan=3, sticky="w", pady=(8,0))

    def on_predict():
        txt = rpm_val.get().strip()
        try:
            n = float(txt)
        except Exception:
            messagebox.showerror("Invalid input", "Please enter a numeric RPM value.")
            return
        p, s = predict_power_at(n)
        ci95 = 2 * s
        msg = f"RPM {n:.2f} → P̂ = {p:.6g} W (±{ci95:.3g})"
        if n < n_min or n > n_max:
            msg += "  [extrapolated]"
        res_var.set(msg)

    b = ttk.Button(frm, text="Predict", command=on_predict)
    b.grid(row=row_offset+3, column=2, padx=(8,0), sticky="w")

    ent.focus_set()
    win.bind('<Return>', lambda e: on_predict())
    win.resizable(False, False)
    win.mainloop()

if args.popup:
    launch_rpm_popup()


# ---------- 4.4 Results GUI ----------
def launch_results_gui():
    if not HAS_TK:
        print("Tkinter not available; cannot show results window.")
        return
    root = tk.Tk()
    root.title("Analysis Results")
    try:
        root.minsize(560, 420)
    except Exception:
        pass
    try:
        root.lift()
        root.attributes('-topmost', True)
        root.after(200, lambda: root.attributes('-topmost', False))
    except Exception:
        pass

    main = ttk.Frame(root, padding=16)
    main.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    row = 0
    logo_img = load_logo()
    logo_img2 = load_second_logo()
    if logo_img or logo_img2:
        logo_frame = ttk.Frame(main)
        logo_frame.grid(row=row, column=0, columnspan=2, pady=(0, 12))
        if logo_img2:
            lbl2 = ttk.Label(logo_frame, image=logo_img2)
            lbl2.image = logo_img2
            lbl2.grid(row=0, column=0, padx=(0, 10))
        if logo_img:
            lbl = ttk.Label(logo_frame, image=logo_img)
            lbl.image = logo_img
            lbl.grid(row=0, column=1, padx=(10, 0))
        row += 1

    csv_name = os.path.basename(csv_path) if csv_path else "(unknown)"
    ttk.Label(main, text=f"File: {csv_name}", font=("Segoe UI", 10, "bold")).grid(row=row, column=0, columnspan=2, sticky="w")
    row += 1

    ttk.Label(main, text=format_blade_tag(blade_metadata), font=("Segoe UI", 9, "italic")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(2, 0))
    row += 1
    ttk.Label(main, text=format_ideal_power(blade_metadata), font=("Segoe UI", 9)).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 6))
    row += 1

    ttk.Label(main, text=f"Training range: {n_min:.2f} – {n_max:.2f} rpm").grid(row=row, column=0, columnspan=2, sticky="w")
    row += 1
    ttk.Label(main, text=f"Samples: {len(df_blocks)}" ).grid(row=row, column=0, columnspan=2, sticky="w")
    row += 1
    ttk.Label(main, text=f"Optimal RPM (GPR): {n_opt:.3f} rpm").grid(row=row, column=0, columnspan=2, sticky="w", pady=(4, 0))
    row += 1
    ttk.Label(main, text=f"Max Power Integral: {P_opt:.6g} W").grid(row=row, column=0, columnspan=2, sticky="w")
    row += 1
    
    # Show Cp if available
    if ideal_power_val and ideal_power_val > 0:
        cp_opt = P_opt / ideal_power_val
        ttk.Label(main, text=f"Max Cp: {cp_opt:.4f}").grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

    ttk.Label(main, text="GPR Kernel:").grid(row=row, column=0, sticky="nw", pady=(6,0))
    kernel_txt = ttk.Label(main, text=str(gpr.kernel_), wraplength=500, justify="left")
    kernel_txt.grid(row=row, column=1, sticky="w", pady=(6,0))
    row += 1

    ttk.Label(main, text="Notes:").grid(row=row, column=0, sticky="nw", pady=(6,0))
    notes = "Confidence band ±2σ shown in plot. Use GUI to query RPM-specific predictions."
    ttk.Label(main, text=notes, wraplength=500, justify="left").grid(row=row, column=1, sticky="w", pady=(6,0))
    row += 1

    ttk.Button(main, text="Close", command=root.destroy).grid(row=row, column=0, columnspan=2, pady=(12,0))

    root.resizable(False, False)
    root.mainloop()

if args.results:
    launch_results_gui()

# ---------- 5. Plot ----------
# "Scientific" styling: try SciencePlots, else fall back to clean seaborn
try:
    import scienceplots  # type: ignore  # optional dependency
    plt.style.use(['science', 'grid', 'no-latex'])
except Exception:
    plt.style.use('seaborn-v0_8-whitegrid')

# Reasonable, journal-friendly rcParams
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 10.5,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'axes.labelweight': 'regular',
    'axes.linewidth': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.framealpha': 0.95,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

# Plot confidence interval first (so it's in the background)
if y_std_real is not None:
    lower = y_pred - 2 * y_std_real
    upper = y_pred + 2 * y_std_real
    ax.fill_between(n_grid.ravel(), lower, upper, alpha=0.22, color='#1f77b4', label="±2σ confidence")

# Plot GPR prediction line with better styling
ax.plot(n_grid.ravel(), y_pred, 'b-', linewidth=2.5, label="GPR surrogate", zorder=3, alpha=0.9)

# Plot data points with better styling
ax.scatter(X.ravel(), y, s=100, c='#d62728', edgecolors='black', linewidth=1.5, 
           label="Experimental data", zorder=5, alpha=0.9)

# Highlight optimal point
ax.axvline(n_opt, color="#2ca02c", linestyle="--", linewidth=2.5, 
           label=f"Optimal n ≈ {n_opt:.1f} rpm", alpha=0.8)
ax.scatter([n_opt], [P_opt], s=200, marker='*', c='gold', edgecolors='black', 
           linewidth=2, zorder=6, label=f"Max Power = {P_opt:.3g}")

# Enhanced labels and title
ax.set_xlabel("Rotational Speed [rpm]", fontweight='bold', fontsize=12)
ax.set_ylabel("Power Integral [W]", fontweight='bold', fontsize=12)
ax.set_title("Wind Turbine Power Optimization using Gaussian Process Regression", 
             fontweight='bold', fontsize=13, pad=15)

# Ticks, minor ticks, scientific notation on y
ax.minorticks_on()
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
sf = ScalarFormatter(useMathText=True)
sf.set_powerlimits((-3, 3))
ax.yaxis.set_major_formatter(sf)
ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))

# Improved grid
ax.grid(True, which='major', alpha=0.35, linestyle='--', linewidth=0.8)
ax.grid(True, which='minor', alpha=0.18, linestyle=':')
ax.set_axisbelow(True)

# Better legend
ax.legend(loc='best', framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)

# Annotation for the optimal point
ax.annotate(
    f"n* ≈ {n_opt:.1f} rpm\nP* ≈ {P_opt:.3g} W",
    xy=(n_opt, P_opt),
    xytext=(0.02, 0.98), textcoords='axes fraction',
    ha='left', va='top',
    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.4', alpha=0.95),
    arrowprops=dict(arrowstyle='->', color='0.35', lw=1.2),
)

if not args.no_plot and not args.gui:
    # Tight layout for better spacing and export in vector + raster formats
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    tag = format_blade_tag(blade_metadata)
    ax.text(0.99, 0.99, tag, transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.6', alpha=0.9), fontsize=9)
    fig.savefig('figures/power_vs_rpm.pdf')
    fig.savefig('figures/power_vs_rpm.png', dpi=300)
    plt.show()

# ---------- 6. Plot Cl/Cd vs Radius (for each RPM) ----------
# Visualize aerodynamic efficiency across blade span for each RPM.
if not args.no_plot and not args.gui and rpm_cl_cd_data:
    fig_clcd, ax_clcd = plt.subplots(figsize=(11, 7), dpi=100)
    
    # Define colors for different RPMs
    colors = plt.cm.tab20(np.linspace(0, 1, len(rpm_cl_cd_data)))
    
    for idx, (rpm_val, cl_cd_list) in enumerate(sorted(rpm_cl_cd_data.items())):
        if not cl_cd_list:
            continue
        
        # Extract radius, Cl, Cd data
        radii = [item[0] for item in cl_cd_list if item[1] is not None and item[2] is not None]
        cl_vals = [item[1] for item in cl_cd_list if item[1] is not None and item[2] is not None]
        cd_vals = [item[2] for item in cl_cd_list if item[1] is not None and item[2] is not None]
        
        if len(radii) < 2:
            continue
        
        # Calculate Cl/Cd ratio
        cl_cd_ratio = [cl / cd if cd != 0 else 0 for cl, cd in zip(cl_vals, cd_vals)]
        
        # Plot line with marker
        ax_clcd.plot(radii, cl_cd_ratio, marker='o', linestyle='-', linewidth=2.5, 
                    markersize=6, color=colors[idx], label=f"RPM {rpm_val:.0f}", 
                    alpha=0.85, zorder=3)
    
    ax_clcd.set_xlabel("Radius [m]", fontweight='bold', fontsize=12)
    ax_clcd.set_ylabel("Lift-to-Drag Ratio (Cl/Cd) [-]", fontweight='bold', fontsize=12)
    ax_clcd.set_title("Aerodynamic Efficiency (Cl/Cd) vs Blade Radius", 
                     fontweight='bold', fontsize=13, pad=15)
    
    # Ticks and grid
    ax_clcd.minorticks_on()
    ax_clcd.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax_clcd.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax_clcd.grid(True, which='major', alpha=0.35, linestyle='--', linewidth=0.8)
    ax_clcd.grid(True, which='minor', alpha=0.18, linestyle=':')
    ax_clcd.set_axisbelow(True)
    
    # Legend
    ax_clcd.legend(loc='best', framealpha=0.95, edgecolor='black', fancybox=True, 
                  shadow=True, fontsize=10)
    
    # Blade tag annotation
    if 'tag' in locals():
        ax_clcd.text(0.99, 0.99, tag, transform=ax_clcd.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.6', alpha=0.9), fontsize=9)
    
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    fig_clcd.savefig('figures/cl_cd_vs_radius.pdf')
    fig_clcd.savefig('figures/cl_cd_vs_radius.png', dpi=300)
    plt.show()

# ---------- 8. Plot Angle of Attack vs Radius (for each RPM) ----------
# Show how local blade loading (AoA) varies along radius for each RPM.
if not args.no_plot and not args.gui and rpm_aoa_data:
    fig_aoa, ax_aoa = plt.subplots(figsize=(11, 7), dpi=100)
    
    # Define colors for different RPMs
    colors = plt.cm.tab20(np.linspace(0, 1, len(rpm_aoa_data)))
    
    for idx, (rpm_val, aoa_list) in enumerate(sorted(rpm_aoa_data.items())):
        if not aoa_list:
            continue
        
        # Extract radius and AoA data
        radii = [item[0] for item in aoa_list if item[1] is not None]
        aoa_vals = [item[1] for item in aoa_list if item[1] is not None]
        
        if len(radii) < 2:
            continue
        
        # Plot line with marker
        ax_aoa.plot(radii, aoa_vals, marker='s', linestyle='-', linewidth=2.5, 
                   markersize=6, color=colors[idx], label=f"RPM {rpm_val:.0f}", 
                   alpha=0.85, zorder=3)
    
    ax_aoa.set_xlabel("Radius [m]", fontweight='bold', fontsize=12)
    ax_aoa.set_ylabel("Angle of Attack [degrees]", fontweight='bold', fontsize=12)
    ax_aoa.set_title("Blade Angle of Attack vs Radius (for each RPM)", 
                    fontweight='bold', fontsize=13, pad=15)
    
    # Ticks and grid
    ax_aoa.minorticks_on()
    ax_aoa.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax_aoa.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax_aoa.grid(True, which='major', alpha=0.35, linestyle='--', linewidth=0.8)
    ax_aoa.grid(True, which='minor', alpha=0.18, linestyle=':')
    ax_aoa.set_axisbelow(True)
    
    # Legend
    ax_aoa.legend(loc='best', framealpha=0.95, edgecolor='black', fancybox=True, 
                 shadow=True, fontsize=10)
    
    # Blade tag annotation
    if 'tag' in locals():
        ax_aoa.text(0.99, 0.99, tag, transform=ax_aoa.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.6', alpha=0.9), fontsize=9)
    
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    fig_aoa.savefig('figures/aoa_vs_radius.pdf')
    fig_aoa.savefig('figures/aoa_vs_radius.png', dpi=300)
    plt.show()

# ---------- 9. Betz Limit Comparison Plot ----------
# Compare predicted Cp curve to theoretical Betz limit and highlight gap.
if not args.no_plot and not args.gui and ideal_power_val is not None and ideal_power_val > 0:
    fig_betz, ax_betz = plt.subplots(figsize=(12, 7), dpi=100)
    
    betz_limit = 16.0 / 27.0  # = 0.5926
    
    # Calculate Cp using smooth GPR predictions
    cp_pred = y_pred / ideal_power_val
    cp_std = y_std_real / ideal_power_val if y_std_real is not None else None
    cp_data = [p / ideal_power_val for p in pint_list]
    cp_opt = P_opt / ideal_power_val
    
    # Plot Cp curve
    ax_betz.plot(n_grid.ravel(), cp_pred, 'b-', linewidth=3, label="Cp (GPR surrogate)", 
                zorder=4, alpha=0.9)
    
    # Plot experimental data
    ax_betz.scatter(rpm_list, cp_data, s=120, c='#d62728', edgecolors='black', linewidth=1.5,
                   label="Experimental data", zorder=5, alpha=0.9)
    
    # Plot Betz limit
    ax_betz.axhline(betz_limit, color='red', linestyle='--', linewidth=3.5, 
                   label=f"Betz limit (Cp = {betz_limit:.4f})", alpha=0.85, zorder=3)
    
    # Highlight optimal point
    ax_betz.scatter([n_opt], [cp_opt], s=250, marker='*', c='gold', edgecolors='black',
                   linewidth=2.5, zorder=6, label=f"Optimum (Cp = {cp_opt:.4f})")
    
    # Fill area between 0 and Betz limit
    ax_betz.fill_between(n_grid.ravel(), 0, betz_limit, alpha=0.08, color='green', label="Theoretically achievable")
    
    # Shading for gap from optimal to Betz
    ax_betz.fill_between([n_opt, n_opt], cp_opt, betz_limit, color='orange', alpha=0.15, 
                        label=f"Gap to Betz: {(betz_limit - cp_opt):.4f}")
    
    ax_betz.set_xlabel("Rotational Speed [rpm]", fontweight='bold', fontsize=12)
    ax_betz.set_ylabel("Power Coefficient Cp [-]", fontweight='bold', fontsize=12)
    ax_betz.set_title("Cp Performance vs Betz Limit (Theoretical Maximum)", 
                     fontweight='bold', fontsize=13, pad=15)
    
    # Set y-axis limits to show Betz limit clearly
    ax_betz.set_ylim(0, betz_limit * 1.15)
    
    # Ticks and grid
    ax_betz.minorticks_on()
    ax_betz.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax_betz.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax_betz.grid(True, which='major', alpha=0.35, linestyle='--', linewidth=0.8)
    ax_betz.grid(True, which='minor', alpha=0.18, linestyle=':')
    ax_betz.set_axisbelow(True)
    
    # Legend
    ax_betz.legend(loc='upper left', framealpha=0.95, edgecolor='black', fancybox=True, 
                  shadow=True, fontsize=11)
    
    # Annotation box with efficiency metrics
    percent_betz = (cp_opt / betz_limit) * 100
    gap_to_betz = betz_limit - cp_opt
    annotation_text = f"""Optimization Summary:
Optimal RPM: {n_opt:.2f}
Optimal Cp: {cp_opt:.4f}
Betz Limit: {betz_limit:.4f}
% of Betz: {percent_betz:.1f}%
Gap: {gap_to_betz:.4f}"""
    
    ax_betz.text(0.99, 0.5, annotation_text, transform=ax_betz.transAxes, 
                ha='right', va='center', fontfamily='monospace', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.6', fc='lightyellow', ec='black', 
                         alpha=0.92, linewidth=1.5))
    
    # Blade tag annotation
    if 'tag' in locals():
        ax_betz.text(0.01, 0.99, tag, transform=ax_betz.transAxes, ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.6', alpha=0.9), fontsize=9)
    
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    fig_betz.savefig('figures/betz_limit_comparison.pdf')
    fig_betz.savefig('figures/betz_limit_comparison.png', dpi=300)
    plt.show()