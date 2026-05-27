# -*- coding: utf-8 -*-
"""
Omega-Twin spatial probability statistics with overlap handling.

Outputs:
    1. Psum_and_RDF_like_by_r.csv
    2. Psum_and_RDF_like_by_n.csv
    3. Psum_and_RDF_like_by_shell_i.csv
    4. overlap_summary.csv

Main modes:
    Omega_center__Twin_target
        center = Omega atoms
        target = Twin/Twin-boundary atoms
        corresponds to P_{T|Omega}

    Twin_center__Omega_target
        center = Twin/Twin-boundary atoms
        target = Omega atoms
        corresponds to P_{Omega|T}

Overlap handling:
    all_exclude_self:
        center and target both include overlap atoms.
        However, the same atom is never counted as its own target.
        This is recommended for main text.

    pure_only:
        center and target both exclude overlap atoms.
        This is more conservative and suitable for supplementary checks.

Author: ChatGPT
"""

import os
import numpy as np
import pandas as pd

from ovito.io import import_file
from ovito.data import CutoffNeighborFinder

try:
    from ovito.data import NearestNeighborFinder
    HAS_NEAREST_NEIGHBOR_FINDER = True
except Exception:
    HAS_NEAREST_NEIGHBOR_FINDER = False


# ============================================================
# 1. User settings
# ============================================================

INPUT_FILE = r"D:\Data\TiNb\ti40nb\tension_1e9\*_ptm.dump"
OUTPUT_DIR = r".\omega_twin_spatial_stats_ti40nb"

# 帧范围。
# END_FRAME = None 表示处理全部帧。
START_FRAME = 35
END_FRAME = None
FRAME_STEP = 1

# 如果你想给输出添加应变列：
# strain = STRAIN_OFFSET + frame * STRAIN_PER_FRAME
# 如果不知道，就保持 STRAIN_PER_FRAME = None。
STRAIN_OFFSET = 0.0
STRAIN_PER_FRAME = None


# ============================================================
# 2. Omega / Twin labels
# ============================================================

# Omega 原子的属性名和取值。
# 例子：
#   OMEGA_PROPERTY = "StructureType"
#   OMEGA_VALUES = [7, 8]
#
# 如果你的 Omega_A/Omega_B 编号不同，在这里改。
OMEGA_PROPERTY = "ptm_type"
OMEGA_VALUES = [9, 10]

# Twin 或 Twin-boundary 的属性名和取值。
# 例子：
#   TWIN_PROPERTY = "bda_type"
#   TWIN_VALUES = [1, 2]
#
# 如果你的 BDA 标签不同，在这里改。
TWIN_PROPERTY = "bda_defect"
TWIN_VALUES = [4]


# ============================================================
# 3. r-based settings
# ============================================================

# 半径统计的最大半径，单位与 dump 坐标一致。
# LAMMPS metal 通常是 Å。
RMAX_A = 10.0
DR_A = 0.5


# ============================================================
# 4. n-neighbor settings
# ============================================================

# 第 n 近邻统计的最大 n。
NMAX = 26

# 如果你的 OVITO 版本没有 NearestNeighborFinder，会自动用 CutoffNeighborFinder 兜底。
# 这时 SEARCH_CUTOFF_A 必须足够大，保证能找到至少 NMAX 个邻居。
SEARCH_CUTOFF_A = 25.0


# ============================================================
# 5. shell-i settings
# ============================================================

# 注意：
# 这里的壳层是“按第 n 近邻范围分壳”，不是按绝对半径分壳。
#
# 对 BCC 中心原子，常用：
#   shell 1: 1-8
#   shell 2: 9-14
#   shell 3: 15-26
#   shell 4: 27-50
#
# 对 Omega/HCP-like 中心原子，壳层定义更依赖你的模板。
# 下面给的是一个可用的默认 rank-shell 划分。
# 如果你已有更准确的 Omega 壳层定义，直接改这里。
SHELLS_FOR_OMEGA_CENTER = [
    (1, 12),
    (13, 18),
    (19, 20),
#    (21, 32),
#    (33, 38),
#    (39, 50),
#    (51, 62),
#    (63, 74),
#    (75, 86),
#    (87, 100),
#    (101, 120),
]

SHELLS_FOR_TWIN_CENTER = [
    (1, 8),
    (9, 14),
    (15, 26),
#    (27, 50),
#    (51, 58),
#    (59, 86),
#    (87, 110),
#    (111, 146),
]


# ============================================================
# 6. Analysis variants
# ============================================================

# all_exclude_self:
#   保留 overlap 原子，但排除同一原子作为自己的 target。
#
# pure_only:
#   只统计 omega_only 和 twin_only，完全排除 overlap 原子。
ANALYSIS_VARIANTS = ["all_exclude_self"]

DO_BY_R = True
DO_BY_N = False
DO_BY_SHELL = False


# ============================================================
# 7. Utility functions
# ============================================================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def get_num_frames(pipeline):
    if hasattr(pipeline, "source") and hasattr(pipeline.source, "num_frames"):
        return pipeline.source.num_frames
    if hasattr(pipeline, "num_frames"):
        return pipeline.num_frames
    return 1


def get_particle_property(data, name):
    if name not in data.particles:
        available = list(data.particles.keys())
        raise KeyError(
            f"Particle property '{name}' was not found.\n"
            f"Available properties:\n{available}"
        )
    return np.asarray(data.particles[name])


def make_mask(values, selected_values):
    return np.isin(values, selected_values)


def get_cell_volume(data):
    cell = data.cell
    matrix = np.asarray(cell[:3, :3])
    return abs(np.linalg.det(matrix))


def get_strain_from_frame(frame):
    if STRAIN_PER_FRAME is None:
        return np.nan
    return STRAIN_OFFSET + frame * STRAIN_PER_FRAME


def build_masks(data):
    omega_values = get_particle_property(data, OMEGA_PROPERTY)
    twin_values = get_particle_property(data, TWIN_PROPERTY)

    omega_mask = make_mask(omega_values, OMEGA_VALUES)
    twin_mask = make_mask(twin_values, TWIN_VALUES)

    overlap_mask = omega_mask & twin_mask
    omega_only_mask = omega_mask & (~twin_mask)
    twin_only_mask = twin_mask & (~omega_mask)

    return {
        "omega": omega_mask,
        "twin": twin_mask,
        "overlap": overlap_mask,
        "omega_only": omega_only_mask,
        "twin_only": twin_only_mask,
    }


def get_center_target_masks(mask_dict, mode, analysis_variant):
    """
    Return center_mask, target_mask, center_kind, target_kind.
    """
    if analysis_variant == "all_exclude_self":
        omega_for_use = mask_dict["omega"]
        twin_for_use = mask_dict["twin"]
    elif analysis_variant == "pure_only":
        omega_for_use = mask_dict["omega_only"]
        twin_for_use = mask_dict["twin_only"]
    else:
        raise ValueError(f"Unknown analysis_variant: {analysis_variant}")

    if mode == "Omega_center__Twin_target":
        return omega_for_use, twin_for_use, "Omega", "Twin"

    if mode == "Twin_center__Omega_target":
        return twin_for_use, omega_for_use, "Twin", "Omega"

    raise ValueError(f"Unknown mode: {mode}")


def get_shells_for_center(center_kind):
    if center_kind == "Omega":
        return SHELLS_FOR_OMEGA_CENTER
    if center_kind == "Twin":
        return SHELLS_FOR_TWIN_CENTER
    raise ValueError(f"Unknown center_kind: {center_kind}")


def overlap_summary_for_frame(frame, data, mask_dict):
    n_total = data.particles.count

    n_omega = int(np.count_nonzero(mask_dict["omega"]))
    n_twin = int(np.count_nonzero(mask_dict["twin"]))
    n_overlap = int(np.count_nonzero(mask_dict["overlap"]))
    n_omega_only = int(np.count_nonzero(mask_dict["omega_only"]))
    n_twin_only = int(np.count_nonzero(mask_dict["twin_only"]))

    f_overlap_in_omega = n_overlap / n_omega if n_omega > 0 else np.nan
    f_overlap_in_twin = n_overlap / n_twin if n_twin > 0 else np.nan
    f_overlap_in_total = n_overlap / n_total if n_total > 0 else np.nan

    return {
        "frame": frame,
        "strain": get_strain_from_frame(frame),
        "N_total": n_total,
        "N_omega": n_omega,
        "N_twin": n_twin,
        "N_overlap": n_overlap,
        "N_omega_only": n_omega_only,
        "N_twin_only": n_twin_only,
        "f_overlap_in_omega": f_overlap_in_omega,
        "f_overlap_in_twin": f_overlap_in_twin,
        "f_overlap_in_total": f_overlap_in_total,
    }


# ============================================================
# 8. r-based statistics
# ============================================================

def compute_by_r_for_one_case(data, frame, mode, analysis_variant,
                              center_mask, target_mask):
    """
    Compute cumulative probability P_sum(r) and RDF-like g(r).

    P_sum_r:
        fraction of center atoms with at least one target atom within r.

    CN_sum_r:
        average number of target atoms within r around each center atom.

    g_r_RDF_like:
        shell pair count normalized by N_center * rho_target * shell_volume.
    """
    center_indices = np.flatnonzero(center_mask)
    n_total = data.particles.count
    n_center = len(center_indices)
    n_target = int(np.count_nonzero(target_mask))

    volume = get_cell_volume(data)
    rho_target = n_target / volume if volume > 0 else np.nan

    bin_edges = np.arange(0.0, RMAX_A + DR_A * 0.5, DR_A)
    r_lower = bin_edges[:-1]
    r_upper = bin_edges[1:]
    r_mid = 0.5 * (r_lower + r_upper)
    n_bins = len(r_mid)

    pair_count_shell = np.zeros(n_bins, dtype=np.int64)
    nearest_distance = np.full(n_center, np.inf, dtype=float)

    if n_center > 0 and n_target > 0:
        finder = CutoffNeighborFinder(RMAX_A, data)

        for cpos, center_index in enumerate(center_indices):
            dmin = np.inf

            for neigh in finder.find(int(center_index)):
                j = neigh.index

                # 关键：排除 self-pair。
                # 如果这个原子既是 Omega 又是 Twin，也不能把自己当成距离 0 的目标。
                if j == center_index:
                    continue

                if not target_mask[j]:
                    continue

                d = neigh.distance
                if d < dmin:
                    dmin = d

                bin_id = int(np.floor(d / DR_A))
                if 0 <= bin_id < n_bins:
                    pair_count_shell[bin_id] += 1

            nearest_distance[cpos] = dmin

    pair_count_cumulative = np.cumsum(pair_count_shell)

    if n_center > 0:
        CN_sum_r = pair_count_cumulative / n_center
        P_sum_r = np.array(
            [np.count_nonzero(nearest_distance <= rr) / n_center for rr in r_mid],
            dtype=float
        )
    else:
        CN_sum_r = np.full(n_bins, np.nan)
        P_sum_r = np.full(n_bins, np.nan)

    shell_volume = (4.0 / 3.0) * np.pi * (r_upper**3 - r_lower**3)

    if n_center > 0 and rho_target > 0:
        expected_pair_shell = n_center * rho_target * shell_volume
        g_r = pair_count_shell / expected_pair_shell
    else:
        g_r = np.full(n_bins, np.nan)

    out = pd.DataFrame({
        "frame": frame,
        "strain": get_strain_from_frame(frame),
        "analysis_variant": analysis_variant,
        "mode": mode,
        "center_kind": mode.split("_center__")[0],
        "target_kind": mode.split("__")[1].replace("_target", ""),
        "r_lower_A": r_lower,
        "r_upper_A": r_upper,
        "r_mid_A": r_mid,
        "N_total": n_total,
        "N_center": n_center,
        "N_target": n_target,
        "rho_target_1_per_A3": rho_target,
        "pair_count_shell": pair_count_shell,
        "pair_count_cumulative": pair_count_cumulative,
        "CN_sum_r": CN_sum_r,
        "P_sum_r": P_sum_r,
        "g_r_RDF_like": g_r,
    })

    return out


# ============================================================
# 9. n-neighbor statistics
# ============================================================

def get_nearest_neighbors_by_rank_with_finder(finder, center_index, nmax):
    """
    Return neighbor indices ordered by distance using an already-created finder.
    """
    neigh_indices = []

    for neigh in finder.find(int(center_index)):
        j = neigh.index

        if j == center_index:
            continue

        neigh_indices.append(j)

        if len(neigh_indices) >= nmax:
            break

    return neigh_indices

def compute_by_n_for_one_case(data, frame, mode, analysis_variant,
                              center_mask, target_mask, nearest_finder):
    center_indices = np.flatnonzero(center_mask)
    n_total = data.particles.count
    n_center = len(center_indices)
    n_target = int(np.count_nonzero(target_mask))

    ranks = np.arange(1, NMAX + 1, dtype=int)

    target_at_rank_count = np.zeros(NMAX, dtype=np.int64)
    nearest_target_rank = np.full(n_center, np.inf, dtype=float)
    target_count_first_n_sum = np.zeros(NMAX, dtype=np.int64)

    if n_center > 0 and n_target > 0:
        for cpos, center_index in enumerate(center_indices):
            neigh_indices = get_nearest_neighbors_by_rank_with_finder(
                nearest_finder,
                center_index,
                NMAX
            )

            target_flags = np.zeros(NMAX, dtype=bool)

            for rank0, j in enumerate(neigh_indices):
                if target_mask[j]:
                    target_flags[rank0] = True

            hit_positions = np.flatnonzero(target_flags)
            if len(hit_positions) > 0:
                nearest_target_rank[cpos] = hit_positions[0] + 1

            target_at_rank_count += target_flags.astype(np.int64)
            target_count_first_n_sum += np.cumsum(target_flags.astype(np.int64))

    if n_center > 0:
        P_sum_n = np.array(
            [np.count_nonzero(nearest_target_rank <= n) / n_center for n in ranks],
            dtype=float
        )
        p_rank_target = target_at_rank_count / n_center
        CN_sum_n = target_count_first_n_sum / n_center
        f_target_in_first_n = CN_sum_n / ranks
    else:
        P_sum_n = np.full(NMAX, np.nan)
        p_rank_target = np.full(NMAX, np.nan)
        CN_sum_n = np.full(NMAX, np.nan)
        f_target_in_first_n = np.full(NMAX, np.nan)

    if n_total > 1:
        p_random_target = n_target / (n_total - 1)
    else:
        p_random_target = np.nan

    if p_random_target > 0:
        g_n = p_rank_target / p_random_target
    else:
        g_n = np.full(NMAX, np.nan)

    out = pd.DataFrame({
        "frame": frame,
        "strain": get_strain_from_frame(frame),
        "analysis_variant": analysis_variant,
        "mode": mode,
        "center_kind": mode.split("_center__")[0],
        "target_kind": mode.split("__")[1].replace("_target", ""),
        "n": ranks,
        "N_total": n_total,
        "N_center": n_center,
        "N_target": n_target,
        "p_random_target": p_random_target,
        "target_at_rank_count": target_at_rank_count,
        "target_count_first_n_sum": target_count_first_n_sum,
        "CN_sum_n": CN_sum_n,
        "P_sum_n": P_sum_n,
        "p_rank_target": p_rank_target,
        "f_target_in_first_n": f_target_in_first_n,
        "g_n_RDF_like": g_n,
    })

    return out

# ============================================================
# 10. shell-i statistics
# ============================================================

def compute_by_shell_i_for_one_case(data, frame, mode, analysis_variant,
                                    center_mask, target_mask, center_kind,
                                    nearest_finder):
    """
    Compute shell-index statistics based on neighbor-rank shells.

    CN_shell:
        average number of target atoms in shell i.

    CN_sum_i:
        cumulative average number of target atoms up to shell i.

    P_shell_target:
        fraction of center atoms having at least one target atom in shell i.

    P_sum_i:
        fraction of center atoms having at least one target atom up to shell i.

    p_shell_target:
        probability that a neighbor inside shell i is target.

    g_i_RDF_like:
        p_shell_target normalized by random target fraction.
    """
    shells = get_shells_for_center(center_kind)
    max_rank = max(upper for _, upper in shells)

    center_indices = np.flatnonzero(center_mask)
    n_total = data.particles.count
    n_center = len(center_indices)
    n_target = int(np.count_nonzero(target_mask))

    shell_count = len(shells)
    target_count_shell_sum = np.zeros(shell_count, dtype=np.int64)
    center_has_target_in_shell = np.zeros(shell_count, dtype=np.int64)
    center_has_target_up_to_shell = np.zeros(shell_count, dtype=np.int64)

    if n_center > 0 and n_target > 0:
        for center_index in center_indices:
            neigh_indices = get_nearest_neighbors_by_rank_with_finder(
                nearest_finder,
                center_index,
                max_rank
            )

            target_flags = np.zeros(max_rank, dtype=bool)

            for rank0, j in enumerate(neigh_indices):
                if rank0 >= max_rank:
                    break
                if target_mask[j]:
                    target_flags[rank0] = True

            cumulative_hit = False

            for shell_idx, (lower, upper) in enumerate(shells):
                # lower/upper 是 1-based rank。
                lo = lower - 1
                hi = upper

                shell_flags = target_flags[lo:hi]
                count_in_shell = int(np.count_nonzero(shell_flags))

                target_count_shell_sum[shell_idx] += count_in_shell

                if count_in_shell > 0:
                    center_has_target_in_shell[shell_idx] += 1

                if count_in_shell > 0:
                    cumulative_hit = True

                if cumulative_hit:
                    center_has_target_up_to_shell[shell_idx] += 1

    shell_i = np.arange(1, shell_count + 1, dtype=int)
    shell_lower_n = np.array([s[0] for s in shells], dtype=int)
    shell_upper_n = np.array([s[1] for s in shells], dtype=int)
    shell_size = shell_upper_n - shell_lower_n + 1

    target_count_cumulative_sum = np.cumsum(target_count_shell_sum)

    if n_center > 0:
        CN_shell = target_count_shell_sum / n_center
        CN_sum_i = target_count_cumulative_sum / n_center
        P_shell_target = center_has_target_in_shell / n_center
        P_sum_i = center_has_target_up_to_shell / n_center
        p_shell_target = target_count_shell_sum / (n_center * shell_size)
    else:
        CN_shell = np.full(shell_count, np.nan)
        CN_sum_i = np.full(shell_count, np.nan)
        P_shell_target = np.full(shell_count, np.nan)
        P_sum_i = np.full(shell_count, np.nan)
        p_shell_target = np.full(shell_count, np.nan)

    if n_total > 1:
        p_random_target = n_target / (n_total - 1)
    else:
        p_random_target = np.nan

    if p_random_target > 0:
        g_i = p_shell_target / p_random_target
    else:
        g_i = np.full(shell_count, np.nan)

    out = pd.DataFrame({
        "frame": frame,
        "strain": get_strain_from_frame(frame),
        "analysis_variant": analysis_variant,
        "mode": mode,
        "center_kind": center_kind,
        "target_kind": mode.split("__")[1].replace("_target", ""),
        "shell_i": shell_i,
        "shell_lower_n": shell_lower_n,
        "shell_upper_n": shell_upper_n,
        "shell_size": shell_size,
        "N_total": n_total,
        "N_center": n_center,
        "N_target": n_target,
        "p_random_target": p_random_target,
        "target_count_shell_sum": target_count_shell_sum,
        "target_count_cumulative_sum": target_count_cumulative_sum,
        "CN_shell": CN_shell,
        "CN_sum_i": CN_sum_i,
        "P_shell_target": P_shell_target,
        "P_sum_i": P_sum_i,
        "p_shell_target": p_shell_target,
        "g_i_RDF_like": g_i,
    })

    return out


# ============================================================
# 11. Main
# ============================================================

def process_frame(pipeline, frame):
    data = pipeline.compute(frame)
    mask_dict = build_masks(data)

    overlap_record = overlap_summary_for_frame(frame, data, mask_dict)

    by_r_frames = []
    by_n_frames = []
    by_shell_frames = []

    modes = [
        "Omega_center__Twin_target",
        "Twin_center__Omega_target",
    ]

    # 每帧只创建一次 NearestNeighborFinder
    nearest_finder = None
    if (DO_BY_N or DO_BY_SHELL) and HAS_NEAREST_NEIGHBOR_FINDER:
        n_needed = max(NMAX, max([s[1] for s in SHELLS_FOR_OMEGA_CENTER + SHELLS_FOR_TWIN_CENTER]))
        nearest_finder = NearestNeighborFinder(n_needed, data)

    for analysis_variant in ANALYSIS_VARIANTS:
        for mode in modes:
            center_mask, target_mask, center_kind, target_kind = get_center_target_masks(
                mask_dict=mask_dict,
                mode=mode,
                analysis_variant=analysis_variant,
            )

            if DO_BY_R:
                by_r = compute_by_r_for_one_case(
                    data=data,
                    frame=frame,
                    mode=mode,
                    analysis_variant=analysis_variant,
                    center_mask=center_mask,
                    target_mask=target_mask,
                )
                by_r_frames.append(by_r)

            if DO_BY_N:
                if nearest_finder is None:
                    raise RuntimeError(
                        "NearestNeighborFinder is unavailable. "
                        "Please reduce to DO_BY_R=True only, or install/update OVITO."
                    )

                by_n = compute_by_n_for_one_case(
                    data=data,
                    frame=frame,
                    mode=mode,
                    analysis_variant=analysis_variant,
                    center_mask=center_mask,
                    target_mask=target_mask,
                    nearest_finder=nearest_finder,
                )
                by_n_frames.append(by_n)

            if DO_BY_SHELL:
                if nearest_finder is None:
                    raise RuntimeError(
                        "NearestNeighborFinder is unavailable. "
                        "Please reduce to DO_BY_R=True only, or install/update OVITO."
                    )

                by_shell = compute_by_shell_i_for_one_case(
                    data=data,
                    frame=frame,
                    mode=mode,
                    analysis_variant=analysis_variant,
                    center_mask=center_mask,
                    target_mask=target_mask,
                    center_kind=center_kind,
                    nearest_finder=nearest_finder,
                )
                by_shell_frames.append(by_shell)

    by_r_out = pd.concat(by_r_frames, ignore_index=True) if len(by_r_frames) > 0 else None
    by_n_out = pd.concat(by_n_frames, ignore_index=True) if len(by_n_frames) > 0 else None
    by_shell_out = pd.concat(by_shell_frames, ignore_index=True) if len(by_shell_frames) > 0 else None

    return overlap_record, by_r_out, by_n_out, by_shell_out

def main():
    ensure_dir(OUTPUT_DIR)

    print("Loading trajectory:")
    print(INPUT_FILE)

    pipeline = import_file(INPUT_FILE)

    nframes_total = get_num_frames(pipeline)
    print(f"Detected frames: {nframes_total}")

    end_frame = nframes_total if END_FRAME is None else min(END_FRAME, nframes_total)
    frames = list(range(START_FRAME, end_frame, FRAME_STEP))

    if len(frames) == 0:
        raise ValueError("No frames selected. Check START_FRAME, END_FRAME, FRAME_STEP.")

    print(f"Frames to process: {frames[0]} ... {frames[-1]}, total = {len(frames)}")

    if not HAS_NEAREST_NEIGHBOR_FINDER:
        print("Warning: NearestNeighborFinder is not available.")
        print("The script will use CutoffNeighborFinder fallback for n and shell statistics.")
        print(f"SEARCH_CUTOFF_A = {SEARCH_CUTOFF_A}")
        print("Please make sure this cutoff is large enough.")

    all_overlap = []
    all_by_r = []
    all_by_n = []
    all_by_shell = []

    for frame in frames:
        print(f"Processing frame {frame} ...")

        overlap_record, by_r, by_n, by_shell = process_frame(pipeline, frame)

        all_overlap.append(overlap_record)
        if by_r is not None:
            all_by_r.append(by_r)

        if by_n is not None:
            all_by_n.append(by_n)

        if by_shell is not None:
            all_by_shell.append(by_shell)

    overlap_path = os.path.join(OUTPUT_DIR, "overlap_summary.csv")
    by_r_path = os.path.join(OUTPUT_DIR, "Psum_and_RDF_like_by_r.csv")
    by_n_path = os.path.join(OUTPUT_DIR, "Psum_and_RDF_like_by_n.csv")
    by_shell_path = os.path.join(OUTPUT_DIR, "Psum_and_RDF_like_by_shell_i.csv")

    overlap_df = pd.DataFrame(all_overlap)
    overlap_df.to_csv(overlap_path, index=False)

    if len(all_by_r) > 0:
        by_r_df = pd.concat(all_by_r, ignore_index=True)
        by_r_df.to_csv(by_r_path, index=False)
        print(f"Saved: {by_r_path}")

    if len(all_by_n) > 0:
        by_n_df = pd.concat(all_by_n, ignore_index=True)
        by_n_df.to_csv(by_n_path, index=False)
        print(f"Saved: {by_n_path}")

    if len(all_by_shell) > 0:
        by_shell_df = pd.concat(all_by_shell, ignore_index=True)
        by_shell_df.to_csv(by_shell_path, index=False)
        print(f"Saved: {by_shell_path}")

    print("\nDone.")
    print(f"Saved: {overlap_path}")


    print("\nImportant output interpretation:")
    print("  analysis_variant = all_exclude_self")
    print("      overlap atoms are retained, but the same atom is never counted as its own target.")
    print("      Recommended for main-text plots.")
    print("")
    print("  analysis_variant = pure_only")
    print("      overlap atoms are excluded from both center and target sets.")
    print("      Recommended for supplementary robustness checks.")
    print("")
    print("  mode = Omega_center__Twin_target")
    print("      center = Omega, target = Twin/TB, corresponding to P_{T|Omega}.")
    print("")
    print("  mode = Twin_center__Omega_target")
    print("      center = Twin/TB, target = Omega, corresponding to P_{Omega|T}.")
    print("")
    print("  overlap_summary.csv gives:")
    print("      f_overlap_in_omega = N_overlap / N_omega")
    print("      f_overlap_in_twin  = N_overlap / N_twin")


if __name__ == "__main__":
    main()