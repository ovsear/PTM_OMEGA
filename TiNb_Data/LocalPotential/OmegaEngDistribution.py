# -*- coding: utf-8 -*-
"""
统计 Omega 原子与局部势能波动的半径关系 E(r)

功能：
1. 以每个 Omega 原子为中心，统计半径 r 内邻域原子的平均局部势能变化。
2. 输出累计平均能量 E_cumulative(r)。
3. 输出壳层平均能量 E_shell(r)。
4. 输出随机中心原子基线，判断 Omega 区域是否真的偏高能。
5. 支持多帧轨迹批处理。
6. 使用 OVITO API 的 CutoffNeighborFinder 处理周期边界。

建议运行方式：
    用 ovitos 运行，或者在已安装 ovito Python 模块的环境中运行。

例如：
    ovitos 02_omega_energy_radius.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ovito.io import import_file
from ovito.data import CutoffNeighborFinder


# ============================================================
# 1. 用户需要修改的参数
# ============================================================

# 输入轨迹文件，可以是单个 dump，也可以是通配符文件序列
INPUT_FILE = r"D:\Data\TiNb\ti30nb\001\tension_1e9\*_ptm.dump"

# 输出文件
OUTPUT_RAW_CSV = r".\ti30nb\omega_energy_radius_raw.csv"
OUTPUT_SUMMARY_CSV = r".\ti30nb\omega_energy_radius_summary.csv"
FIGURE_DIR = r".\ti30nb\omega_energy_figures"

# ------------------------------------------------------------
# 轨迹帧设置
# ------------------------------------------------------------

# None 表示自动处理全部帧
FRAME_START = 35
FRAME_STOP = None
FRAME_STEP = 1

# 如果你希望跳过前期弛豫帧，可以改为：
# FRAME_START = 10

# ------------------------------------------------------------
# 应变信息
# ------------------------------------------------------------

# 如果没有应变信息，设为 "none"
# 如果想按帧号线性估计应变，设为 "linear"
STRAIN_MODE = "none"     # "none" or "linear"

# 仅在 STRAIN_MODE = "linear" 时使用
STRAIN_START = 0.0
STRAIN_INCREMENT_PER_FRAME = 0.002

# ------------------------------------------------------------
# 半径设置，单位 Angstrom
# ------------------------------------------------------------

R_MIN_A = 2.0
R_MAX_A = 10.0
DR_A = 0.5

# ------------------------------------------------------------
# 粒子属性列名
# ------------------------------------------------------------

ID_COLUMN = "Particle Identifier"
TYPE_COLUMN = "Particle Type"

# 单原子势能列
# 如果 OVITO 里显示为 "Potential Energy"，就改成 "Potential Energy"
PE_COLUMN = "c_3"

# 结构类型列
# 你之前使用的是 StructureType
STRUCTURE_COLUMN = "ptm_type"

# Omega 标记列
# 如果你自研 PTM-ω 输出为 OmegaType，保持这样
# 如果 Omega 直接写在 StructureType 里，则改成：
# OMEGA_COLUMN = "StructureType"
OMEGA_COLUMN = "ptm_type"

# Omega 的取值
# 例如 OMEGA_A = 1, OMEGA_B = 2
# 如果你的 StructureType 中 Omega_A/Omega_B 是 8/9，就改成 [8, 9]
OMEGA_VALUES = [9, 10]

# BCC 的结构类型值
# OVITO 标准 PTM 里通常是：
# 0 Other, 1 FCC, 2 HCP, 3 BCC
BCC_VALUES = [3]

# ------------------------------------------------------------
# 势能基准定义
# ------------------------------------------------------------

# 计算能量波动：
# Delta e_i = pe_i - reference_energy
#
# 推荐使用 type_bcc_median：
# 对每种元素类型，取同一帧中 BCC bulk 原子的势能中位数作为基准。
#
# 可选：
# "type_bcc_median"        每种元素分别用 BCC 原子中位数作基准，推荐
# "type_nonomega_median"   每种元素分别用非 Omega 原子中位数作基准
# "global_nonomega_median" 所有非 Omega 原子统一中位数作基准
REFERENCE_MODE = "type_bcc_median"

# 是否把 Omega 中心原子自身计入半径 r 内平均
# 如果想看 Omega 本体加邻域，用 True
# 如果只想看 Omega 周围环境，不包括中心原子，用 False
INCLUDE_CENTER_ATOM = True

# ------------------------------------------------------------
# 随机基线设置
# ------------------------------------------------------------

# 随机中心原子数量与 Omega 原子数量相同
USE_RANDOM_BASELINE = True

# 随机重复次数。越大越稳，但越慢
N_RANDOM_REPEATS = 10

# 随机中心是否匹配 Omega 原子的元素类型分布
# 推荐 True，避免 Ti/Nb 势能本征差异影响
MATCH_RANDOM_BY_TYPE = True

RANDOM_SEED = 12345

# 随机中心候选池是否排除 Omega 原子
EXCLUDE_OMEGA_FROM_RANDOM = True

# ------------------------------------------------------------
# 质量控制
# ------------------------------------------------------------

# 每帧 Omega 原子太少时跳过
MIN_OMEGA_COUNT = 5

# 画图
MAKE_FIGURES = True
DPI = 300


# ============================================================
# 2. 基础工具函数
# ============================================================

def ensure_parent_dir(path):
    folder = os.path.dirname(os.path.abspath(path))
    if folder and not os.path.exists(folder):
        os.makedirs(folder)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_num_frames(pipeline):
    try:
        return pipeline.source.num_frames
    except Exception:
        return pipeline.num_frames


def get_strain(frame_index):
    if STRAIN_MODE == "none":
        return np.nan
    elif STRAIN_MODE == "linear":
        return STRAIN_START + frame_index * STRAIN_INCREMENT_PER_FRAME
    else:
        raise ValueError("STRAIN_MODE must be 'none' or 'linear'.")


def get_property_array(data, name):
    """
    从 OVITO DataCollection 中读取粒子属性。
    如果找不到属性，打印当前所有属性名，方便修改脚本配置。
    """
    if name not in data.particles:
        available = list(data.particles.keys())
        raise KeyError(
            f"找不到粒子属性: {name}\n"
            f"当前可用属性有:\n{available}"
        )
    return np.asarray(data.particles[name].array)


def build_radii():
    return np.arange(R_MIN_A, R_MAX_A + 0.5 * DR_A, DR_A)


# ============================================================
# 3. 计算单原子能量波动 Delta e
# ============================================================

def compute_delta_energy(pe, particle_type, omega_mask, structure_type=None):
    """
    计算每个原子的局部势能波动 Delta e。

    Delta e_i = pe_i - reference_energy

    reference_energy 根据 REFERENCE_MODE 决定。
    """

    pe = np.asarray(pe, dtype=float)
    particle_type = np.asarray(particle_type)

    delta_e = np.zeros_like(pe, dtype=float)
    reference_records = []

    if REFERENCE_MODE == "type_bcc_median":
        if structure_type is None:
            raise ValueError("REFERENCE_MODE='type_bcc_median' 需要 STRUCTURE_COLUMN。")

        structure_type = np.asarray(structure_type)
        unique_types = np.unique(particle_type)

        for tp in unique_types:
            mask_type = (particle_type == tp)
            mask_ref = (
                mask_type
                & np.isin(structure_type, BCC_VALUES)
                & (~omega_mask)
            )

            # 如果 BCC 原子数量太少，退化为同类型非 Omega 原子中位数
            if np.sum(mask_ref) < 10:
                mask_ref = mask_type & (~omega_mask)

            # 如果仍然太少，退化为同类型全部原子
            if np.sum(mask_ref) < 10:
                mask_ref = mask_type

            ref = np.median(pe[mask_ref])
            delta_e[mask_type] = pe[mask_type] - ref

            reference_records.append({
                "type": tp,
                "reference_energy": ref,
                "reference_atom_count": int(np.sum(mask_ref)),
                "reference_mode_used": "type_bcc_median_or_fallback"
            })

    elif REFERENCE_MODE == "type_nonomega_median":
        unique_types = np.unique(particle_type)

        for tp in unique_types:
            mask_type = (particle_type == tp)
            mask_ref = mask_type & (~omega_mask)

            if np.sum(mask_ref) < 10:
                mask_ref = mask_type

            ref = np.median(pe[mask_ref])
            delta_e[mask_type] = pe[mask_type] - ref

            reference_records.append({
                "type": tp,
                "reference_energy": ref,
                "reference_atom_count": int(np.sum(mask_ref)),
                "reference_mode_used": "type_nonomega_median_or_fallback"
            })

    elif REFERENCE_MODE == "global_nonomega_median":
        mask_ref = ~omega_mask

        if np.sum(mask_ref) < 10:
            mask_ref = np.ones_like(omega_mask, dtype=bool)

        ref = np.median(pe[mask_ref])
        delta_e[:] = pe - ref

        reference_records.append({
            "type": "global",
            "reference_energy": ref,
            "reference_atom_count": int(np.sum(mask_ref)),
            "reference_mode_used": "global_nonomega_median_or_fallback"
        })

    else:
        raise ValueError(
            "REFERENCE_MODE must be one of: "
            "'type_bcc_median', 'type_nonomega_median', 'global_nonomega_median'."
        )

    return delta_e, reference_records


# ============================================================
# 4. 以指定中心原子计算 E(r)
# ============================================================

def compute_energy_profile_for_centers(data, center_indices, delta_e, radii):
    """
    对一组中心原子计算：
    1. 累计平均能量 E_cumulative(r)
    2. 壳层平均能量 E_shell(r)

    输出：
        per_center_mean:
            对每个中心先求均值，再对中心取平均。
            这样每个中心权重相同。
        pooled_mean:
            把所有中心的邻域原子混合后求平均。
            这样邻居数量更多的中心权重更大。
    """

    max_r = float(np.max(radii))
    finder = CutoffNeighborFinder(max_r, data)

    n_r = len(radii)

    center_cum_means = []
    center_shell_means = []

    pooled_cum_sum = np.zeros(n_r, dtype=float)
    pooled_cum_count = np.zeros(n_r, dtype=int)

    pooled_shell_sum = np.zeros(n_r, dtype=float)
    pooled_shell_count = np.zeros(n_r, dtype=int)

    mean_cum_neighbor_count = np.zeros(n_r, dtype=float)
    mean_shell_neighbor_count = np.zeros(n_r, dtype=float)

    valid_center_count = 0

    for center in center_indices:
        distances = []
        energies = []

        if INCLUDE_CENTER_ATOM:
            distances.append(0.0)
            energies.append(delta_e[center])

        for neigh in finder.find(int(center)):
            distances.append(float(neigh.distance))
            energies.append(delta_e[neigh.index])

        if len(distances) == 0:
            continue

        distances = np.asarray(distances, dtype=float)
        energies = np.asarray(energies, dtype=float)

        order = np.argsort(distances)
        distances = distances[order]
        energies = energies[order]

        cumsum_e = np.cumsum(energies)

        cum_mean = np.full(n_r, np.nan, dtype=float)
        shell_mean = np.full(n_r, np.nan, dtype=float)

        cum_counts_this = np.zeros(n_r, dtype=int)
        shell_counts_this = np.zeros(n_r, dtype=int)

        previous_idx = 0

        for k, r in enumerate(radii):
            idx = np.searchsorted(distances, r, side="right")

            # 累计区域：0 <= d <= r
            if idx > 0:
                s = cumsum_e[idx - 1]
                c = idx
                cum_mean[k] = s / c
                pooled_cum_sum[k] += s
                pooled_cum_count[k] += c
                cum_counts_this[k] = c

            # 壳层区域：r_{k-1} < d <= r_k
            shell_count = idx - previous_idx
            if shell_count > 0:
                shell_sum = cumsum_e[idx - 1]
                if previous_idx > 0:
                    shell_sum -= cumsum_e[previous_idx - 1]

                shell_mean[k] = shell_sum / shell_count
                pooled_shell_sum[k] += shell_sum
                pooled_shell_count[k] += shell_count
                shell_counts_this[k] = shell_count

            previous_idx = idx

        center_cum_means.append(cum_mean)
        center_shell_means.append(shell_mean)

        mean_cum_neighbor_count += cum_counts_this
        mean_shell_neighbor_count += shell_counts_this

        valid_center_count += 1

    if valid_center_count == 0:
        return None

    center_cum_means = np.asarray(center_cum_means)
    center_shell_means = np.asarray(center_shell_means)

    cumulative_center_mean = np.nanmean(center_cum_means, axis=0)
    shell_center_mean = np.nanmean(center_shell_means, axis=0)

    cumulative_center_std = np.nanstd(center_cum_means, axis=0)
    shell_center_std = np.nanstd(center_shell_means, axis=0)

    cumulative_pooled_mean = np.divide(
        pooled_cum_sum,
        pooled_cum_count,
        out=np.full_like(pooled_cum_sum, np.nan, dtype=float),
        where=pooled_cum_count > 0
    )

    shell_pooled_mean = np.divide(
        pooled_shell_sum,
        pooled_shell_count,
        out=np.full_like(pooled_shell_sum, np.nan, dtype=float),
        where=pooled_shell_count > 0
    )

    mean_cum_neighbor_count = mean_cum_neighbor_count / valid_center_count
    mean_shell_neighbor_count = mean_shell_neighbor_count / valid_center_count

    return {
        "valid_center_count": valid_center_count,

        "E_cumulative_center_mean": cumulative_center_mean,
        "E_cumulative_center_std": cumulative_center_std,
        "E_cumulative_pooled_mean": cumulative_pooled_mean,

        "E_shell_center_mean": shell_center_mean,
        "E_shell_center_std": shell_center_std,
        "E_shell_pooled_mean": shell_pooled_mean,

        "mean_cumulative_neighbor_count": mean_cum_neighbor_count,
        "mean_shell_neighbor_count": mean_shell_neighbor_count,
    }


# ============================================================
# 5. 随机中心原子选择
# ============================================================

def sample_random_centers(rng, omega_indices, omega_types, all_indices, particle_type, omega_mask):
    """
    随机选择与 Omega 原子数量相同的中心原子。

    如果 MATCH_RANDOM_BY_TYPE=True：
        随机中心的元素类型分布与 Omega 中心相同。
    """

    if EXCLUDE_OMEGA_FROM_RANDOM:
        candidate_mask = ~omega_mask
    else:
        candidate_mask = np.ones_like(omega_mask, dtype=bool)

    random_centers = []

    if MATCH_RANDOM_BY_TYPE:
        unique_types = np.unique(omega_types)

        for tp in unique_types:
            n_need = int(np.sum(omega_types == tp))

            candidates = all_indices[candidate_mask & (particle_type == tp)]

            if len(candidates) == 0:
                continue

            replace = len(candidates) < n_need
            sampled = rng.choice(candidates, size=n_need, replace=replace)
            random_centers.extend(sampled.tolist())

        random_centers = np.asarray(random_centers, dtype=int)

        # 如果由于某些 type 没有候选，数量不足，则从总候选池补齐
        if len(random_centers) < len(omega_indices):
            candidates_all = all_indices[candidate_mask]
            n_need = len(omega_indices) - len(random_centers)
            replace = len(candidates_all) < n_need
            extra = rng.choice(candidates_all, size=n_need, replace=replace)
            random_centers = np.concatenate([random_centers, extra])

    else:
        candidates = all_indices[candidate_mask]
        replace = len(candidates) < len(omega_indices)
        random_centers = rng.choice(candidates, size=len(omega_indices), replace=replace)

    return random_centers.astype(int)


def compute_random_baseline(data, rng, omega_indices, particle_type, omega_mask, delta_e, radii):
    """
    计算随机中心的 E(r) 基线。
    多次随机重复后取平均。
    """

    all_indices = np.arange(len(particle_type), dtype=int)
    omega_types = particle_type[omega_indices]

    profiles = []

    for _ in range(N_RANDOM_REPEATS):
        centers = sample_random_centers(
            rng=rng,
            omega_indices=omega_indices,
            omega_types=omega_types,
            all_indices=all_indices,
            particle_type=particle_type,
            omega_mask=omega_mask
        )

        profile = compute_energy_profile_for_centers(
            data=data,
            center_indices=centers,
            delta_e=delta_e,
            radii=radii
        )

        if profile is not None:
            profiles.append(profile)

    if len(profiles) == 0:
        return None

    keys_to_average = [
        "E_cumulative_center_mean",
        "E_cumulative_pooled_mean",
        "E_shell_center_mean",
        "E_shell_pooled_mean",
        "mean_cumulative_neighbor_count",
        "mean_shell_neighbor_count",
    ]

    out = {}
    for key in keys_to_average:
        values = np.asarray([p[key] for p in profiles])
        out[key] = np.nanmean(values, axis=0)
        out[key + "_std_between_random"] = np.nanstd(values, axis=0)

    out["valid_center_count"] = int(np.mean([p["valid_center_count"] for p in profiles]))
    return out


# ============================================================
# 6. 单帧分析
# ============================================================

def analyze_frame(pipeline, frame, radii, rng):
    data = pipeline.compute(frame)

    particle_type = get_property_array(data, TYPE_COLUMN)
    pe = get_property_array(data, PE_COLUMN)

    omega_label = get_property_array(data, OMEGA_COLUMN)
    omega_mask = np.isin(omega_label, OMEGA_VALUES)

    if STRUCTURE_COLUMN is not None and STRUCTURE_COLUMN in data.particles:
        structure_type = get_property_array(data, STRUCTURE_COLUMN)
    else:
        structure_type = None

    omega_indices = np.where(omega_mask)[0].astype(int)
    n_omega = len(omega_indices)
    n_atoms = len(pe)

    if n_omega < MIN_OMEGA_COUNT:
        print(f"[Frame {frame}] Omega 原子数过少: {n_omega}，跳过。")
        return []

    delta_e, reference_records = compute_delta_energy(
        pe=pe,
        particle_type=particle_type,
        omega_mask=omega_mask,
        structure_type=structure_type
    )

    omega_profile = compute_energy_profile_for_centers(
        data=data,
        center_indices=omega_indices,
        delta_e=delta_e,
        radii=radii
    )

    if omega_profile is None:
        print(f"[Frame {frame}] Omega profile 为空，跳过。")
        return []

    random_profile = None
    if USE_RANDOM_BASELINE:
        random_profile = compute_random_baseline(
            data=data,
            rng=rng,
            omega_indices=omega_indices,
            particle_type=particle_type,
            omega_mask=omega_mask,
            delta_e=delta_e,
            radii=radii
        )

    strain = get_strain(frame)

    rows = []

    for k, r in enumerate(radii):
        row = {
            "frame": frame,
            "strain": strain,
            "radius_A": float(r),

            "n_atoms": int(n_atoms),
            "n_omega": int(n_omega),

            "E_omega_cumulative_center_mean": omega_profile["E_cumulative_center_mean"][k],
            "E_omega_cumulative_center_std": omega_profile["E_cumulative_center_std"][k],
            "E_omega_cumulative_pooled_mean": omega_profile["E_cumulative_pooled_mean"][k],

            "E_omega_shell_center_mean": omega_profile["E_shell_center_mean"][k],
            "E_omega_shell_center_std": omega_profile["E_shell_center_std"][k],
            "E_omega_shell_pooled_mean": omega_profile["E_shell_pooled_mean"][k],

            "omega_mean_cumulative_neighbor_count": omega_profile["mean_cumulative_neighbor_count"][k],
            "omega_mean_shell_neighbor_count": omega_profile["mean_shell_neighbor_count"][k],
        }

        if random_profile is not None:
            row.update({
                "E_random_cumulative_center_mean": random_profile["E_cumulative_center_mean"][k],
                "E_random_cumulative_pooled_mean": random_profile["E_cumulative_pooled_mean"][k],
                "E_random_shell_center_mean": random_profile["E_shell_center_mean"][k],
                "E_random_shell_pooled_mean": random_profile["E_shell_pooled_mean"][k],

                "random_mean_cumulative_neighbor_count": random_profile["mean_cumulative_neighbor_count"][k],
                "random_mean_shell_neighbor_count": random_profile["mean_shell_neighbor_count"][k],
            })

            row["DeltaE_cumulative_vs_random"] = (
                row["E_omega_cumulative_center_mean"]
                - row["E_random_cumulative_center_mean"]
            )

            row["DeltaE_shell_vs_random"] = (
                row["E_omega_shell_center_mean"]
                - row["E_random_shell_center_mean"]
            )

        rows.append(row)

    return rows


# ============================================================
# 7. 汇总与绘图
# ============================================================

def make_summary(raw_df):
    """
    对所有帧按半径汇总。
    """
    numeric_cols = [
        c for c in raw_df.columns
        if c not in ["frame"]
    ]

    summary = raw_df.groupby("radius_A")[numeric_cols].agg(["mean", "std", "count"])
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    return summary


def plot_curves(raw_df, summary_df):
    ensure_dir(FIGURE_DIR)

    # 累计能量曲线
    plt.figure(figsize=(6, 4))
    plt.plot(
        summary_df["radius_A"],
        summary_df["E_omega_cumulative_center_mean_mean"],
        label="Omega centers"
    )

    if "E_random_cumulative_center_mean_mean" in summary_df.columns:
        plt.plot(
            summary_df["radius_A"],
            summary_df["E_random_cumulative_center_mean_mean"],
            label="Random centers"
        )

    plt.xlabel("Radius r (Angstrom)")
    plt.ylabel("Cumulative mean energy excess (eV/atom)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "E_cumulative_vs_r.png"), dpi=DPI)
    plt.close()

    # 壳层能量曲线
    plt.figure(figsize=(6, 4))
    plt.plot(
        summary_df["radius_A"],
        summary_df["E_omega_shell_center_mean_mean"],
        label="Omega centers"
    )

    if "E_random_shell_center_mean_mean" in summary_df.columns:
        plt.plot(
            summary_df["radius_A"],
            summary_df["E_random_shell_center_mean_mean"],
            label="Random centers"
        )

    plt.xlabel("Radius r (Angstrom)")
    plt.ylabel("Shell mean energy excess (eV/atom)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "E_shell_vs_r.png"), dpi=DPI)
    plt.close()

    # Omega 相对随机基线的能量差
    if "DeltaE_cumulative_vs_random_mean" in summary_df.columns:
        plt.figure(figsize=(6, 4))
        plt.plot(
            summary_df["radius_A"],
            summary_df["DeltaE_cumulative_vs_random_mean"]
        )
        plt.axhline(0.0, linewidth=1, linestyle="--")
        plt.xlabel("Radius r (Angstrom)")
        plt.ylabel("Omega - random cumulative energy excess (eV/atom)")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, "DeltaE_cumulative_vs_random.png"), dpi=DPI)
        plt.close()

    if "DeltaE_shell_vs_random_mean" in summary_df.columns:
        plt.figure(figsize=(6, 4))
        plt.plot(
            summary_df["radius_A"],
            summary_df["DeltaE_shell_vs_random_mean"]
        )
        plt.axhline(0.0, linewidth=1, linestyle="--")
        plt.xlabel("Radius r (Angstrom)")
        plt.ylabel("Omega - random shell energy excess (eV/atom)")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, "DeltaE_shell_vs_random.png"), dpi=DPI)
        plt.close()


# ============================================================
# 8. 主程序
# ============================================================

def main():
    print("读取轨迹...")
    pipeline = import_file(INPUT_FILE)

    num_frames = get_num_frames(pipeline)
    print(f"检测到帧数: {num_frames}")

    frame_stop = FRAME_STOP
    if frame_stop is None:
        frame_stop = num_frames

    frames = list(range(FRAME_START, frame_stop, FRAME_STEP))
    print(f"将处理帧: {frames[:10]} ... 共 {len(frames)} 帧")

    radii = build_radii()
    print(f"半径范围: {radii[0]} - {radii[-1]} Å, step = {DR_A} Å")

    rng = np.random.default_rng(RANDOM_SEED)

    all_rows = []

    for frame in frames:
        print(f"处理 frame {frame} ...")
        rows = analyze_frame(
            pipeline=pipeline,
            frame=frame,
            radii=radii,
            rng=rng
        )
        all_rows.extend(rows)

    if len(all_rows) == 0:
        print("没有得到任何有效统计结果。请检查 Omega 标签、势能列名和帧范围。")
        return

    raw_df = pd.DataFrame(all_rows)

    ensure_parent_dir(OUTPUT_RAW_CSV)
    raw_df.to_csv(OUTPUT_RAW_CSV, index=False)
    print(f"已输出逐帧结果: {OUTPUT_RAW_CSV}")

    summary_df = make_summary(raw_df)

    ensure_parent_dir(OUTPUT_SUMMARY_CSV)
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False)
    print(f"已输出汇总结果: {OUTPUT_SUMMARY_CSV}")

    if MAKE_FIGURES:
        print("绘图...")
        plot_curves(raw_df, summary_df)
        print(f"图像已输出至: {FIGURE_DIR}")

    print("完成。")

    print("\n建议优先查看以下列：")
    print("1. E_omega_cumulative_center_mean")
    print("2. E_random_cumulative_center_mean")
    print("3. DeltaE_cumulative_vs_random")
    print("4. E_omega_shell_center_mean")
    print("5. DeltaE_shell_vs_random")


if __name__ == "__main__":
    main()