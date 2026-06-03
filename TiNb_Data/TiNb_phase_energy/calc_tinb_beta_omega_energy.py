# calc_tinb_beta_omega_energy.py
# -*- coding: utf-8 -*-

"""
Purpose
-------
Calculate composition-dependent relative energy between beta-BCC and omega
structures in Ti-Nb random solid solutions:

    Delta E_{omega-beta} = E_omega - E_beta

Recommended use for Fig. 4a:
    mode = "single_point" or "min_fixed"

Notes
-----
1. single_point:
   Only evaluates the potential energy of the ideal topology.
   This is the most stable choice for phase-constrained beta/omega comparison.

2. min_fixed:
   Minimizes atomic positions with fixed box.
   This includes local chemical relaxation, but low-Nb beta may still collapse.

3. min_iso:
   Minimizes atomic positions and allows isotropic box relaxation.
   This is useful as a stability check, but not recommended as the only
   data source for Fig. 4a.

Requirements
------------
pip install ase numpy pandas matplotlib
LAMMPS executable must support the Ehemann-Wilkins Ti-Nb potential.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ase import Atoms
from ase.io import write


# =========================
# User parameters
# =========================

LAMMPS_EXE = r"lmp"  # Example: r"/home/lian/lammps/src/lmp_mpi"
POTENTIAL_FILE = r"ehemann_tinb.pt"

# If your LAMMPS uses the official style, keep "meam/spline".
# If you compiled the original supplemental style as gmeam/spline, change this.
PAIR_STYLE = "meam/spline"
PAIR_COEFF = f"* * {POTENTIAL_FILE} Ti Nb"

# Ti-Nb compositions
NB_AT_PCTS = [20, 30, 40]

# Random seeds. Increase to 10 or 20 for final paper-quality statistics.
SEEDS = [1001, 1002, 1003, 1004, 1005]

# Supercell sizes.
# BCC atoms = 2 * nx * ny * nz
# Omega atoms = 3 * nx * ny * nz
BCC_REPS = (8, 8, 8)       # 1024 atoms
OMEGA_REPS = (7, 7, 7)     # 1029 atoms

# Approximate beta-BCC lattice parameter in Angstrom.
# The omega initial lattice is generated from beta-omega crystallographic relation:
# a_omega = sqrt(2) * a_beta, c_omega = sqrt(3)/2 * a_beta
A_BETA0 = 3.28

# Volume or lattice scaling factors for phase-constrained E(V) scan.
# For quick test, use fewer points; for final, use e.g. np.linspace(0.94, 1.06, 13).
SCALES = np.linspace(0.96, 1.04, 9)

# Calculation mode:
# "single_point": no relaxation, just run 0
# "min_fixed": minimize atom positions with fixed box
# "min_iso": atom minimization + isotropic box relaxation
CALC_MODE = "single_point"

WORK_DIR = Path("cases")
RESULT_DIR = Path("results")

# LAMMPS minimization settings
ETOL = "1.0e-12"
FTOL = "1.0e-12"
MAXITER = 20000
MAXEVAL = 200000


# =========================
# Structure generation
# =========================

def build_bcc(a_beta: float, reps: tuple[int, int, int]) -> Atoms:
    """
    Build conventional beta-BCC supercell with two atoms per conventional cell.
    """
    nx, ny, nz = reps

    cell = np.array([
        [a_beta * nx, 0.0, 0.0],
        [0.0, a_beta * ny, 0.0],
        [0.0, 0.0, a_beta * nz],
    ])

    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
    ])

    scaled_positions = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for b in basis:
                    scaled_positions.append([
                        (i + b[0]) / nx,
                        (j + b[1]) / ny,
                        (k + b[2]) / nz,
                    ])

    symbols = ["Ti"] * len(scaled_positions)
    atoms = Atoms(symbols=symbols, scaled_positions=scaled_positions, cell=cell, pbc=True)
    return atoms


def build_omega(a_beta: float, reps: tuple[int, int, int]) -> Atoms:
    """
    Build ideal hexagonal omega supercell.

    Omega C32/P6/mmm-like basis:
        (0, 0, 0)
        (2/3, 1/3, 1/2)
        (1/3, 2/3, 1/2)

    Approximate beta-omega lattice relation:
        a_omega = sqrt(2) * a_beta
        c_omega = sqrt(3)/2 * a_beta
    """
    nx, ny, nz = reps

    a_omega = np.sqrt(2.0) * a_beta
    c_omega = np.sqrt(3.0) / 2.0 * a_beta

    # ASE cell vectors are row vectors.
    a1 = np.array([a_omega, 0.0, 0.0])
    a2 = np.array([-0.5 * a_omega, np.sqrt(3.0) / 2.0 * a_omega, 0.0])
    a3 = np.array([0.0, 0.0, c_omega])

    cell = np.array([
        nx * a1,
        ny * a2,
        nz * a3,
    ])

    basis = np.array([
        [0.0, 0.0, 0.0],
        [2.0 / 3.0, 1.0 / 3.0, 0.5],
        [1.0 / 3.0, 2.0 / 3.0, 0.5],
    ])

    scaled_positions = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for b in basis:
                    scaled_positions.append([
                        (i + b[0]) / nx,
                        (j + b[1]) / ny,
                        (k + b[2]) / nz,
                    ])

    symbols = ["Ti"] * len(scaled_positions)
    atoms = Atoms(symbols=symbols, scaled_positions=scaled_positions, cell=cell, pbc=True)
    return atoms


def assign_random_tinb(atoms: Atoms, nb_at_pct: float, seed: int) -> Atoms:
    """
    Randomly assign Nb atoms to a Ti matrix.
    Type mapping in LAMMPS data will be:
        1 = Ti
        2 = Nb
    """
    rng = np.random.default_rng(seed)
    atoms = atoms.copy()

    n_atoms = len(atoms)
    n_nb = int(round(n_atoms * nb_at_pct / 100.0))

    indices = np.arange(n_atoms)
    nb_indices = rng.choice(indices, size=n_nb, replace=False)

    symbols = np.array(["Ti"] * n_atoms, dtype=object)
    symbols[nb_indices] = "Nb"
    atoms.set_chemical_symbols(symbols.tolist())

    return atoms


def scale_cell_keep_scaled_positions(atoms: Atoms, scale: float) -> Atoms:
    """
    Uniformly scale the cell while keeping fractional coordinates unchanged.
    """
    atoms = atoms.copy()
    atoms.set_cell(atoms.cell.array * scale, scale_atoms=True)
    return atoms


# =========================
# LAMMPS input and execution
# =========================

def write_lammps_input(case_dir: Path, data_file: str, mode: str) -> None:
    """
    Write LAMMPS input file.
    """
    if mode not in {"single_point", "min_fixed", "min_iso"}:
        raise ValueError(f"Unknown mode: {mode}")

    if mode == "single_point":
        relax_block = """
# Single-point energy only.
run 0
"""
    elif mode == "min_fixed":
        relax_block = f"""
# Atomic relaxation with fixed simulation box.
min_style cg
minimize {ETOL} {FTOL} {MAXITER} {MAXEVAL}
run 0
"""
    else:
        relax_block = f"""
# Atomic relaxation + isotropic box relaxation.
# Use this only as a stability check, not as the only Fig. 4a data source.
fix boxrelax all box/relax iso 0.0 vmax 0.001
min_style cg
minimize {ETOL} {FTOL} {MAXITER} {MAXEVAL}
unfix boxrelax
run 0
"""

    in_text = f"""
clear
units metal
dimension 3
boundary p p p
atom_style atomic

read_data {data_file}

mass 1 47.867
mass 2 92.90637

pair_style {PAIR_STYLE}
pair_coeff {PAIR_COEFF}

neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes

thermo 50
thermo_style custom step atoms pe press pxx pyy pzz lx ly lz xy xz yz

compute peall all pe
variable n equal count(all)
variable epa equal pe/v_n

{relax_block}

variable pe_final equal pe
variable epa_final equal v_epa

print "RESULT pe ${{pe_final}} epa ${{epa_final}} natoms ${{n}}" file result.dat screen yes

write_data relaxed.data
"""
    (case_dir / "in.min").write_text(in_text, encoding="utf-8")


def run_lammps(case_dir: Path) -> None:
    """
    Run LAMMPS inside a case directory.
    """
    cmd = [LAMMPS_EXE, "-in", "in.min", "-log", "log.lammps"]
    subprocess.run(cmd, cwd=case_dir, check=True)


def parse_result(case_dir: Path) -> dict:
    """
    Parse result.dat.
    """
    text = (case_dir / "result.dat").read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"RESULT\s+pe\s+([\-0-9.eE+]+)\s+epa\s+([\-0-9.eE+]+)\s+natoms\s+([0-9]+)", text)
    if not m:
        raise RuntimeError(f"Cannot parse result.dat in {case_dir}")

    return {
        "pe": float(m.group(1)),
        "epa": float(m.group(2)),
        "natoms": int(m.group(3)),
    }


# =========================
# Main workflow
# =========================

def main() -> None:
    WORK_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)

    potential_src = Path(POTENTIAL_FILE).resolve()
    if not potential_src.exists():
        raise FileNotFoundError(f"Potential file not found: {potential_src}")

    records = []

    for nb in NB_AT_PCTS:
        for seed in SEEDS:
            for phase in ["beta_bcc", "omega"]:
                for scale in SCALES:
                    case_name = f"{phase}_Ti{100-nb}Nb{nb}_seed{seed}_scale{scale:.4f}_{CALC_MODE}"
                    case_dir = WORK_DIR / case_name
                    case_dir.mkdir(parents=True, exist_ok=True)

                    # Build structure.
                    if phase == "beta_bcc":
                        atoms = build_bcc(A_BETA0, BCC_REPS)
                    else:
                        atoms = build_omega(A_BETA0, OMEGA_REPS)

                    atoms = scale_cell_keep_scaled_positions(atoms, scale)
                    atoms = assign_random_tinb(atoms, nb_at_pct=nb, seed=seed)

                    data_file = "structure.data"

                    # Write LAMMPS data.
                    write(
                        case_dir / data_file,
                        atoms,
                        format="lammps-data",
                        specorder=["Ti", "Nb"],
                        masses=True,
                        atom_style="atomic",
                    )

                    # Copy potential file.
                    shutil.copy2(potential_src, case_dir / potential_src.name)

                    # Write input and run.
                    write_lammps_input(case_dir, data_file=data_file, mode=CALC_MODE)

                    print(f"Running {case_name}")
                    run_lammps(case_dir)

                    result = parse_result(case_dir)
                    records.append({
                        "nb_at_pct": nb,
                        "seed": seed,
                        "phase": phase,
                        "scale": scale,
                        "mode": CALC_MODE,
                        "natoms": result["natoms"],
                        "pe_eV": result["pe"],
                        "e_per_atom_eV": result["epa"],
                        "case_dir": str(case_dir),
                    })

    df = pd.DataFrame(records)
    raw_csv = RESULT_DIR / f"raw_phase_energy_{CALC_MODE}.csv"
    df.to_csv(raw_csv, index=False)
    print(f"Saved raw results: {raw_csv}")

    # For each phase/composition/seed, take minimum energy over volume scale.
    idx = df.groupby(["nb_at_pct", "seed", "phase"])["e_per_atom_eV"].idxmin()
    df_min = df.loc[idx].copy()
    min_csv = RESULT_DIR / f"min_phase_energy_{CALC_MODE}.csv"
    df_min.to_csv(min_csv, index=False)
    print(f"Saved minimum-energy results: {min_csv}")

    # Calculate Delta E = E_omega - E_beta for paired seeds.
    rows = []
    for nb in NB_AT_PCTS:
        for seed in SEEDS:
            sub = df_min[(df_min["nb_at_pct"] == nb) & (df_min["seed"] == seed)]
            e_beta = sub[sub["phase"] == "beta_bcc"]["e_per_atom_eV"].values
            e_omega = sub[sub["phase"] == "omega"]["e_per_atom_eV"].values
            if len(e_beta) != 1 or len(e_omega) != 1:
                continue

            rows.append({
                "nb_at_pct": nb,
                "seed": seed,
                "E_beta_eV_atom": e_beta[0],
                "E_omega_eV_atom": e_omega[0],
                "DeltaE_omega_minus_beta_eV_atom": e_omega[0] - e_beta[0],
                "DeltaE_omega_minus_beta_meV_atom": 1000.0 * (e_omega[0] - e_beta[0]),
            })

    df_delta = pd.DataFrame(rows)
    delta_csv = RESULT_DIR / f"deltaE_omega_minus_beta_{CALC_MODE}.csv"
    df_delta.to_csv(delta_csv, index=False)
    print(f"Saved Delta E results: {delta_csv}")

    # Summary statistics.
    summary = df_delta.groupby("nb_at_pct")["DeltaE_omega_minus_beta_meV_atom"].agg(
        mean="mean",
        std="std",
        count="count"
    ).reset_index()

    summary_csv = RESULT_DIR / f"summary_deltaE_{CALC_MODE}.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"Saved summary: {summary_csv}")

    # Plot Fig. 4a draft.
    plt.figure(figsize=(4.2, 3.2))
    plt.errorbar(
        summary["nb_at_pct"],
        summary["mean"],
        yerr=summary["std"],
        marker="o",
        capsize=4,
        linewidth=1.5,
    )
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel("Nb content (at. %)")
    plt.ylabel(r"$\Delta E_{\omega-\beta}$ (meV/atom)")
    plt.tight_layout()

    fig_path = RESULT_DIR / f"Fig4a_deltaE_omega_beta_{CALC_MODE}.png"
    plt.savefig(fig_path, dpi=600)
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()