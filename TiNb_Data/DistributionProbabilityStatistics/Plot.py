# -*- coding: utf-8 -*-
"""
Mode-aware averaging and plotting for Omega-Twin spatial statistics.

This script reads:
    omega_twin_spatial_stats_ti20nb/Psum_and_RDF_like_by_r.csv
    omega_twin_spatial_stats_ti20nb/Psum_and_RDF_like_by_n.csv
    omega_twin_spatial_stats_ti20nb/Psum_and_RDF_like_by_shell_i.csv

    omega_twin_spatial_stats_ti30nb/...
    omega_twin_spatial_stats_ti40nb/...

Important:
    The CSV contains two modes:
        Omega_center__Twin_target  -> P_{T|Omega}
        Twin_center__Omega_target  -> P_{Omega|T}

This script averages by:
    composition + mode + x

Therefore, the two modes will not be mixed.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. User settings
# ============================================================

BASE_DIR = r"."
OUTPUT_DIR = r".\averaged_results_mode_aware"

COMPOSITIONS = {
    "Ti20Nb": r"omega_twin_spatial_stats_ti20nb",
    "Ti30Nb": r"omega_twin_spatial_stats_ti30nb",
    "Ti40Nb": r"omega_twin_spatial_stats_ti40nb",
}

FILES = {
    "by_r": {
        "filename": "Psum_and_RDF_like_by_r.csv",
        "x_col": "r_mid_A",
        "x_label": "r (Å)",
        "y_cols": ["CN_sum_r", "P_sum_r", "g_r_RDF_like"],
    },
#     "by_n": {"filename": "Psum_and_RDF_like_by_n.csv","x_col": "n","x_label": "Neighbor rank n","y_cols": ["CN_sum_n", "P_sum_n", "p_rank_target", "g_n_RDF_like"],},
#    "by_shell_i": {"filename": "Psum_and_RDF_like_by_shell_i.csv","x_col": "shell_i","x_label": "Shell index i","y_cols": ["CN_shell", "CN_sum_i", "P_sum_i", "p_shell_target", "g_i_RDF_like"],},
}

MODE_COLUMN = "mode"

MODE_LABELS = {
    "Omega_center__Twin_target": r"$P_{T|\Omega}$: Omega center, Twin target",
    "Twin_center__Omega_target": r"$P_{\Omega|T}$: Twin center, Omega target",
}

# Which error band to draw:
# "sem" = standard error
# "std" = standard deviation
# "none" = no error band
ERROR_BAND = "sem"

FIG_FORMAT = "png"
DPI = 300
FIGSIZE = (6.4, 4.4)


# ============================================================
# 2. Utility functions
# ============================================================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def sanitize_filename(name):
    name = str(name)
    name = name.replace("/", "_over_")
    name = name.replace("\\", "_")
    name = name.replace("|", "_")
    name = name.replace(" ", "_")
    name = name.replace("{", "")
    name = name.replace("}", "")
    name = re.sub(r"[^0-9a-zA-Z_\-\.\(\)]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def read_csv_safely(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find file: {path}")
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def get_available_y_cols(df, requested_cols):
    available = []
    missing = []

    for col in requested_cols:
        if col in df.columns:
            available.append(col)
        else:
            missing.append(col)

    if len(missing) > 0:
        print(f"  Warning: missing columns skipped: {missing}")

    if len(available) == 0:
        raise ValueError(
            "No requested y columns found.\n"
            f"Requested: {requested_cols}\n"
            f"Available: {list(df.columns)}"
        )

    return available


def group_average(df, x_col, y_cols):
    """
    Average by composition + mode + x.
    """
    group_cols = ["composition", MODE_COLUMN, x_col]

    grouped = df.groupby(group_cols, as_index=False)

    mean_df = grouped[y_cols].mean()
    std_df = grouped[y_cols].std()
    count_df = grouped[y_cols].count()

    out = mean_df[group_cols].copy()

    for col in y_cols:
        out[f"{col}__mean"] = mean_df[col]
        out[f"{col}__std"] = std_df[col]
        out[f"{col}__count"] = count_df[col]
        out[f"{col}__sem"] = std_df[col] / np.sqrt(count_df[col].replace(0, np.nan))

    out = out.sort_values(["composition", MODE_COLUMN, x_col]).reset_index(drop=True)
    return out


def read_and_merge_one_kind(kind, cfg):
    """
    Read Ti20Nb, Ti30Nb, Ti40Nb data for one file type.
    """
    filename = cfg["filename"]
    x_col = cfg["x_col"]
    requested_y_cols = cfg["y_cols"]

    all_raw = []
    common_y_cols = None

    print("\n" + "=" * 80)
    print(f"Processing {kind}: {filename}")
    print("=" * 80)

    for comp, folder in COMPOSITIONS.items():
        path = os.path.join(BASE_DIR, folder, filename)
        print(f"Reading {comp}: {path}")

        df = read_csv_safely(path)

        if x_col not in df.columns:
            raise KeyError(
                f"x column '{x_col}' not found in {path}\n"
                f"Available columns: {list(df.columns)}"
            )

        if MODE_COLUMN not in df.columns:
            raise KeyError(
                f"mode column '{MODE_COLUMN}' not found in {path}\n"
                f"Available columns: {list(df.columns)}"
            )

        y_cols = get_available_y_cols(df, requested_y_cols)

        if common_y_cols is None:
            common_y_cols = y_cols
        else:
            common_y_cols = [c for c in common_y_cols if c in y_cols]

        print(f"  modes: {df[MODE_COLUMN].unique().tolist()}")
        print(f"  x column: {x_col}")
        print(f"  y columns: {y_cols}")

        keep_cols = ["frame", "strain", MODE_COLUMN, x_col] + y_cols
        keep_cols = [c for c in keep_cols if c in df.columns]

        temp = df[keep_cols].copy()
        temp["composition"] = comp
        all_raw.append(temp)

    raw_all = pd.concat(all_raw, ignore_index=True)

    if common_y_cols is None or len(common_y_cols) == 0:
        raise ValueError(f"No common y columns found for {kind}.")

    avg_all = group_average(raw_all, x_col, common_y_cols)

    return raw_all, avg_all, common_y_cols


# ============================================================
# 3. Plotting
# ============================================================

def plot_one_y_by_mode(avg_all, kind, x_col, x_label, y_col):
    """
    For one y column, generate one figure for each mode.
    Each figure contains Ti20Nb, Ti30Nb, Ti40Nb curves.
    """
    fig_dir = os.path.join(OUTPUT_DIR, "figures", kind, sanitize_filename(y_col))
    ensure_dir(fig_dir)

    modes = avg_all[MODE_COLUMN].dropna().unique().tolist()

    saved_figs = []

    for mode in modes:
        sub_mode = avg_all[avg_all[MODE_COLUMN] == mode].copy()

        plt.figure(figsize=FIGSIZE)

        for comp in COMPOSITIONS.keys():
            sub = sub_mode[sub_mode["composition"] == comp].copy()
            if sub.empty:
                continue

            sub = sub.sort_values(x_col)

            x = sub[x_col].to_numpy(dtype=float)
            y = sub[f"{y_col}__mean"].to_numpy(dtype=float)

            plt.plot(x, y, label=comp)

            if ERROR_BAND.lower() in ["sem", "std"]:
                err_col = f"{y_col}__{ERROR_BAND.lower()}"
                if err_col in sub.columns:
                    err = sub[err_col].to_numpy(dtype=float)
                    plt.fill_between(x, y - err, y + err, alpha=0.20)

        mode_label = MODE_LABELS.get(mode, str(mode))

        plt.xlabel(x_label)
        plt.ylabel(y_col)
        plt.title(mode_label)
        plt.legend(frameon=False)
        plt.tight_layout()

        fig_name = f"{kind}_{sanitize_filename(y_col)}_{sanitize_filename(mode)}.{FIG_FORMAT}"
        fig_path = os.path.join(fig_dir, fig_name)
        plt.savefig(fig_path, dpi=DPI)
        plt.close()

        saved_figs.append(fig_path)
        print(f"  saved: {fig_path}")

    return saved_figs


def plot_mode_comparison_for_each_composition(avg_all, kind, x_col, x_label, y_col):
    """
    Optional plot:
    For each composition, plot the two modes in one figure.
    This helps compare P_{T|Omega} and P_{Omega|T}.
    """
    fig_dir = os.path.join(OUTPUT_DIR, "figures", kind, sanitize_filename(y_col), "mode_comparison")
    ensure_dir(fig_dir)

    saved_figs = []

    for comp in COMPOSITIONS.keys():
        sub_comp = avg_all[avg_all["composition"] == comp].copy()
        if sub_comp.empty:
            continue

        plt.figure(figsize=FIGSIZE)

        for mode in sub_comp[MODE_COLUMN].dropna().unique():
            sub = sub_comp[sub_comp[MODE_COLUMN] == mode].copy()
            sub = sub.sort_values(x_col)

            x = sub[x_col].to_numpy(dtype=float)
            y = sub[f"{y_col}__mean"].to_numpy(dtype=float)

            label = MODE_LABELS.get(mode, str(mode))
            plt.plot(x, y, label=label)

            if ERROR_BAND.lower() in ["sem", "std"]:
                err_col = f"{y_col}__{ERROR_BAND.lower()}"
                if err_col in sub.columns:
                    err = sub[err_col].to_numpy(dtype=float)
                    plt.fill_between(x, y - err, y + err, alpha=0.20)

        plt.xlabel(x_label)
        plt.ylabel(y_col)
        plt.title(comp)
        plt.legend(frameon=False, fontsize=8)
        plt.tight_layout()

        fig_name = f"{kind}_{sanitize_filename(y_col)}_{comp}_mode_comparison.{FIG_FORMAT}"
        fig_path = os.path.join(fig_dir, fig_name)
        plt.savefig(fig_path, dpi=DPI)
        plt.close()

        saved_figs.append(fig_path)
        print(f"  saved: {fig_path}")

    return saved_figs


def plot_all(avg_all, kind, cfg, y_cols):
    x_col = cfg["x_col"]
    x_label = cfg["x_label"]

    print(f"\nPlotting {kind} ...")

    for y_col in y_cols:
        if f"{y_col}__mean" not in avg_all.columns:
            continue

        # Figure type 1:
        # For each mode, compare Ti20Nb/Ti30Nb/Ti40Nb.
        plot_one_y_by_mode(avg_all, kind, x_col, x_label, y_col)

        # Figure type 2:
        # For each composition, compare two modes.
        plot_mode_comparison_for_each_composition(avg_all, kind, x_col, x_label, y_col)


# ============================================================
# 4. Main
# ============================================================

def main():
    ensure_dir(OUTPUT_DIR)

    print("Base directory:")
    print(os.path.abspath(BASE_DIR))

    print("\nOutput directory:")
    print(os.path.abspath(OUTPUT_DIR))

    for kind, cfg in FILES.items():
        raw_all, avg_all, y_cols = read_and_merge_one_kind(kind, cfg)

        raw_out = os.path.join(OUTPUT_DIR, f"raw_combined_{kind}.csv")
        avg_out = os.path.join(OUTPUT_DIR, f"average_{kind}.csv")

        raw_all.to_csv(raw_out, index=False)
        avg_all.to_csv(avg_out, index=False)

        print(f"Saved raw combined table: {raw_out}")
        print(f"Saved averaged table:     {avg_out}")

        plot_all(avg_all, kind, cfg, y_cols)

    print("\nAll done.")
    print(f"Results saved in: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()