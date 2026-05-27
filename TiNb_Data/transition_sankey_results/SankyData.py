# -*- coding: utf-8 -*-
"""
transition_sankey_omega_twin.py

用途：
1. 读取 OVITO/LAMMPS dump 轨迹；
2. 基于 PTM-omega 和 BDA 结果给每个原子定义状态；
3. 统计原子状态转移矩阵；
4. 输出桑基图数据和 HTML；
5. 计算 omega 是否先于 twin 出现，从而判断“前驱体”还是“竞争关系”。

建议输入：
dump 文件中至少包含：
- Particle Identifier
- Structure Type 或你的 PTM 结构类型属性
- BDA Type 或你的 BDA 缺陷类型属性

运行方式：
直接在 IDE 中运行，先修改下面“手动参数区”。
"""

from pathlib import Path
import numpy as np
import pandas as pd

from ovito.io import import_file


# ============================================================
# 手动参数区
# ============================================================

CASES = [
    {
        "name": "Ti20Nb",
        "input": r"D:\Data\TiNb\ti20nb\tension_1e9\*_ptm.dump",
    },
    {
        "name": "Ti30Nb",
        "input": r"D:\Data\TiNb\ti30nb\001\tension_1e9\*_ptm.dump",
    },
    {
        "name": "Ti40Nb",
        "input": r"D:\Data\TiNb\ti40nb\tension_1e9\*_ptm.dump",
    },
]

OUTPUT_ROOT = Path(r"C:\Users\017\Desktop\TiNb_Data\Code\transition_sankey_results")

# 若你的轨迹不是每帧一个文件，而是单个多帧 dump，也可以直接写成：
# "input": r"C:\Users\017\Desktop\TiNb_Data\Ti20Nb\dump.ptm_bda.lammpstrj"

# 用于估算应变。若你的每帧已经对应固定应变间隔，填这里。
# 例如拉伸到 20%，一共 101 帧，则 STRAIN_PER_FRAME = 0.20 / 100
# 不知道就先设为 None，脚本会只用 frame index。
STRAIN_PER_FRAME = None

# 统计转移矩阵的时间间隔。
# LAG_FRAMES = 1 表示统计相邻帧 state(t) -> state(t+1)
LAG_FRAMES = 1

# 前驱性判断窗口。
# 例如 5 表示：omega 出现后 5 帧内是否转为 twin/twin boundary。
PRECURSOR_WINDOW_FRAMES = 5

# 桑基图不建议使用全部帧，否则节点和连线太多。
# 例如每 5 帧取一次。
SANKEY_EVERY_N_FRAMES = 5

# 只显示计数大于等于该值的桑基图边。
MIN_SANKEY_EDGE_COUNT = 10


# ------------------------------------------------------------
# 粒子属性候选名
# 你可以根据 dump 中实际字段修改。
# ------------------------------------------------------------

ID_PROP_CANDIDATES = [
    "Particle Identifier",
    "id",
    "ID",
]

STRUCTURE_PROP_CANDIDATES = [
    "ptm_type",
    "structure_type",
    "StructureType",
]

BDA_PROP_CANDIDATES = [
    "bda_defect",
    "bda_type",
    "BDA",
    "Defect Type",
    "defect_type",
    "BDAType",
]


# ------------------------------------------------------------
# 结构编号设置
# 这里的默认假设是：
# OVITO 常规 PTM: OTHER=0, FCC=1, HCP=2, BCC=3
# 你的自定义 OMEGA_A/OMEGA_B 编号需要自己确认。
# ------------------------------------------------------------

STRUCTURE_CODES = {
    "OTHER": {0},
    "FCC": {1},
    "HCP": {2},
    "BCC": {3},

    # 下面两个需要按你的 PTM-omega 输出编号修改。
    # 例如你自定义 OMEGA_A=8, OMEGA_B=9，就保持如下。
    "OMEGA": {9, 10},
}

# BDA 编号也需要按你自己的代码输出修改。
# 下面只是占位示例。
BDA_CODES = {
    # 孪晶内部原子
    "TWIN": {-1},

    # 孪晶界原子
    "TB": {4},
}


# ------------------------------------------------------------
# 互斥状态定义
# 数字越小越普通，数字越大越偏向特殊结构。
# 注意：OMEGA_TB 和 OMEGA_TWIN 用于保留 omega 与 twin/tb 重叠信息。
# ------------------------------------------------------------

STATE_NAMES = [
    "BCC",
    "FCC_like",
    "OMEGA",
    "TB",
    "OMEGA_TB",
    "OTHER",
    "TWIN",
    "OMEGA_TWIN"
]

STATE_ID = {name: i for i, name in enumerate(STATE_NAMES)}


# ============================================================
# 工具函数
# ============================================================

def get_num_frames(pipeline):
    """兼容不同 OVITO 版本的帧数获取方式。"""
    if hasattr(pipeline, "num_frames"):
        return int(pipeline.num_frames)
    if hasattr(pipeline.source, "num_frames"):
        return int(pipeline.source.num_frames)
    return 1


def get_particle_property(data, candidates, required=True):
    """按候选名读取粒子属性。"""
    particles = data.particles
    available = list(particles.keys())

    for name in candidates:
        if name in available:
            return np.asarray(particles[name]), name

    if required:
        raise KeyError(
            "没有找到所需粒子属性。候选名为：{}\n当前可用属性为：{}".format(
                candidates, available
            )
        )

    return None, None


def to_int_array(arr):
    """把属性数组转成整数数组。"""
    arr = np.asarray(arr)
    if arr.ndim > 1:
        arr = arr[:, 0]
    return arr.astype(np.int64)


def isin_codes(arr, codes):
    """判断 arr 是否属于 codes 集合。"""
    arr = to_int_array(arr)
    return np.isin(arr, list(codes))


def build_state(structure, bda):
    """
    根据 structure 和 bda 生成互斥状态。

    状态优先级：
    1. 默认 OTHER
    2. BCC
    3. FCC_like
    4. TWIN / TB
    5. OMEGA
    6. OMEGA_TWIN / OMEGA_TB

    这样可以保留 omega 与 twin/tb 重叠的信息。
    """
    n = len(structure)
    state = np.full(n, STATE_ID["OTHER"], dtype=np.int16)

    is_bcc = isin_codes(structure, STRUCTURE_CODES["BCC"])
    is_fcc = isin_codes(structure, STRUCTURE_CODES["FCC"])
    is_omega = isin_codes(structure, STRUCTURE_CODES["OMEGA"])

    if bda is None:
        is_twin = np.zeros(n, dtype=bool)
        is_tb = np.zeros(n, dtype=bool)
    else:
        is_twin = isin_codes(bda, BDA_CODES["TWIN"])
        is_tb = isin_codes(bda, BDA_CODES["TB"])

    state[is_bcc] = STATE_ID["BCC"]
    state[is_fcc] = STATE_ID["FCC_like"]
    state[is_twin] = STATE_ID["TWIN"]
    state[is_tb] = STATE_ID["TB"]
    state[is_omega] = STATE_ID["OMEGA"]

    state[is_omega & is_twin] = STATE_ID["OMEGA_TWIN"]
    state[is_omega & is_tb] = STATE_ID["OMEGA_TB"]

    omega_mask = is_omega
    twin_mask = is_twin | is_tb

    return state, omega_mask, twin_mask


def frame_to_strain(frame_index):
    if STRAIN_PER_FRAME is None:
        return np.nan
    return frame_index * STRAIN_PER_FRAME


def compute_transition_matrix(states, lag_frames=1):
    """
    states: shape = [n_frames, n_atoms]
    返回所有相邻 lag 的总转移矩阵。
    """
    n_states = len(STATE_NAMES)
    matrix = np.zeros((n_states, n_states), dtype=np.int64)

    n_frames = states.shape[0]
    for f in range(n_frames - lag_frames):
        s0 = states[f]
        s1 = states[f + lag_frames]
        valid = (s0 >= 0) & (s1 >= 0)

        a = s0[valid]
        b = s1[valid]

        np.add.at(matrix, (a, b), 1)

    return matrix


def matrix_to_dataframe(matrix):
    df = pd.DataFrame(matrix, index=STATE_NAMES, columns=STATE_NAMES)
    return df


def normalize_rows(matrix):
    matrix = matrix.astype(float)
    row_sum = matrix.sum(axis=1, keepdims=True)
    out = np.divide(matrix, row_sum, out=np.zeros_like(matrix), where=row_sum > 0)
    return out


def compute_state_fraction(states):
    n_frames = states.shape[0]
    n_states = len(STATE_NAMES)

    rows = []
    for f in range(n_frames):
        s = states[f]
        valid = s >= 0
        counts = np.bincount(s[valid], minlength=n_states)
        total = counts.sum()

        row = {
            "frame": f,
            "strain": frame_to_strain(f),
            "total_atoms": int(total),
        }

        for i, name in enumerate(STATE_NAMES):
            row[f"count_{name}"] = int(counts[i])
            row[f"fraction_{name}"] = counts[i] / total if total > 0 else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def compute_onset_analysis(states, omega_matrix, twin_matrix):
    """
    判断 omega 和 twin 的先后关系。

    omega_onset: 某原子第一次被识别为 omega 的帧
    twin_onset: 某原子第一次被识别为 twin 或 twin boundary 的帧

    若 twin_onset - omega_onset > 0:
        omega 先于 twin，支持前驱体路径。
    若 twin_onset - omega_onset < 0:
        twin 先于 omega，支持 twin-boundary-induced omega。
    若二者都出现但间隔很小：
        更像同步形成。
    若 omega 出现但窗口内不转为 twin：
        倾向竞争或旁路路径。
    """
    n_frames, n_atoms = states.shape

    omega_onset = np.full(n_atoms, -1, dtype=np.int32)
    twin_onset = np.full(n_atoms, -1, dtype=np.int32)

    for f in range(n_frames):
        omega_now = omega_matrix[f]
        twin_now = twin_matrix[f]

        omega_new = (omega_onset < 0) & omega_now
        twin_new = (twin_onset < 0) & twin_now

        omega_onset[omega_new] = f
        twin_onset[twin_new] = f

    has_omega = omega_onset >= 0
    has_twin = twin_onset >= 0
    has_both = has_omega & has_twin

    delta_frame = np.full(n_atoms, np.nan)
    delta_frame[has_both] = twin_onset[has_both] - omega_onset[has_both]

    # 分类
    cls = np.full(n_atoms, "none", dtype=object)

    cls[has_omega & (~has_twin)] = "omega_only"
    cls[(~has_omega) & has_twin] = "twin_only"

    cls[has_both & (delta_frame > 0)] = "omega_before_twin"
    cls[has_both & (delta_frame < 0)] = "twin_before_omega"
    cls[has_both & (delta_frame == 0)] = "simultaneous"

    # 窗口内前驱体：omega 出现后 W 帧内出现 twin
    w = PRECURSOR_WINDOW_FRAMES
    precursor_window = has_both & (delta_frame > 0) & (delta_frame <= w)

    # omega 竞争路径：出现 omega，但 W 帧内没有转为 twin
    # 包括 omega_only，以及虽然后来转 twin 但超过窗口。
    omega_competition_like = has_omega & (
        (~has_twin) | (delta_frame <= 0) | (delta_frame > w)
    )

    rows = []
    for label in [
        "omega_only",
        "twin_only",
        "omega_before_twin",
        "twin_before_omega",
        "simultaneous",
        "none",
    ]:
        rows.append({
            "class": label,
            "count": int(np.sum(cls == label)),
            "fraction_all_atoms": float(np.mean(cls == label)),
        })

    summary = pd.DataFrame(rows)

    n_omega = int(np.sum(has_omega))
    n_twin = int(np.sum(has_twin))
    n_both = int(np.sum(has_both))

    metrics = {
        "n_atoms": int(n_atoms),
        "n_omega_ever": n_omega,
        "n_twin_ever": n_twin,
        "n_both_omega_and_twin": n_both,

        "fraction_omega_ever": n_omega / n_atoms,
        "fraction_twin_ever": n_twin / n_atoms,
        "fraction_both_given_omega": n_both / n_omega if n_omega > 0 else np.nan,
        "fraction_both_given_twin": n_both / n_twin if n_twin > 0 else np.nan,

        "fraction_omega_before_twin_given_both": (
            np.sum(has_both & (delta_frame > 0)) / n_both if n_both > 0 else np.nan
        ),
        "fraction_twin_before_omega_given_both": (
            np.sum(has_both & (delta_frame < 0)) / n_both if n_both > 0 else np.nan
        ),
        "fraction_simultaneous_given_both": (
            np.sum(has_both & (delta_frame == 0)) / n_both if n_both > 0 else np.nan
        ),

        "precursor_window_frames": int(w),
        "fraction_twin_within_window_after_omega_given_omega": (
            np.sum(precursor_window) / n_omega if n_omega > 0 else np.nan
        ),
        "fraction_omega_competition_like_given_omega": (
            np.sum(omega_competition_like) / n_omega if n_omega > 0 else np.nan
        ),

        "mean_delta_frame_twin_minus_omega_for_both": (
            float(np.nanmean(delta_frame[has_both])) if n_both > 0 else np.nan
        ),
        "median_delta_frame_twin_minus_omega_for_both": (
            float(np.nanmedian(delta_frame[has_both])) if n_both > 0 else np.nan
        ),
    }

    onset_df = pd.DataFrame({
        "atom_index": np.arange(n_atoms),
        "omega_onset_frame": omega_onset,
        "twin_onset_frame": twin_onset,
        "delta_frame_twin_minus_omega": delta_frame,
        "class": cls,
        "precursor_window": precursor_window,
        "omega_competition_like": omega_competition_like,
    })

    # 滞后分布
    lag_df = (
        onset_df.loc[has_both, "delta_frame_twin_minus_omega"]
        .value_counts()
        .sort_index()
        .rename_axis("delta_frame_twin_minus_omega")
        .reset_index(name="count")
    )
    lag_df["fraction_given_both"] = lag_df["count"] / n_both if n_both > 0 else np.nan

    return summary, pd.DataFrame([metrics]), onset_df, lag_df


def build_sankey_edges(states, frame_indices):
    """
    生成桑基图边表：
    frame_i:state_A -> frame_j:state_B
    """
    rows = []

    for a, b in zip(frame_indices[:-1], frame_indices[1:]):
        s0 = states[a]
        s1 = states[b]
        valid = (s0 >= 0) & (s1 >= 0)

        pair = pd.DataFrame({
            "source_state_id": s0[valid],
            "target_state_id": s1[valid],
        })

        grouped = (
            pair.groupby(["source_state_id", "target_state_id"])
            .size()
            .reset_index(name="count")
        )

        for _, r in grouped.iterrows():
            count = int(r["count"])
            if count < MIN_SANKEY_EDGE_COUNT:
                continue

            source_name = STATE_NAMES[int(r["source_state_id"])]
            target_name = STATE_NAMES[int(r["target_state_id"])]

            rows.append({
                "source_frame": int(a),
                "target_frame": int(b),
                "source": f"F{a}_{source_name}",
                "target": f"F{b}_{target_name}",
                "source_state": source_name,
                "target_state": target_name,
                "count": count,
            })

    return pd.DataFrame(rows)


def write_sankey_html(edges_df, output_html):
    """
    用 Plotly 输出桑基图 HTML。
    若没有安装 plotly，则只输出 csv，不中断。
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("没有安装 plotly，跳过 HTML 桑基图。可运行：pip install plotly")
        return

    if edges_df.empty:
        print("桑基图边表为空，跳过 HTML 输出。")
        return

    labels = pd.Index(pd.concat([edges_df["source"], edges_df["target"]]).unique())
    label_to_id = {label: i for i, label in enumerate(labels)}

    source = edges_df["source"].map(label_to_id).to_numpy()
    target = edges_df["target"].map(label_to_id).to_numpy()
    value = edges_df["count"].to_numpy()

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=15,
                    thickness=14,
                    line=dict(width=0.3),
                    label=labels.tolist(),
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value,
                ),
            )
        ]
    )

    fig.update_layout(
        title_text="Atom-state transition Sankey diagram",
        font_size=11,
        width=1500,
        height=900,
    )

    fig.write_html(str(output_html))
    print(f"已输出桑基图：{output_html}")


def process_case(case):
    name = case["name"]
    input_path = case["input"]
    out_dir = OUTPUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"开始处理：{name}")
    print(f"输入：{input_path}")

    pipeline = import_file(input_path)
    n_frames = get_num_frames(pipeline)
    print(f"读取到帧数：{n_frames}")

    # 第一帧确定原子 ID
    data0 = pipeline.compute(0)

    ids0, id_prop_name = get_particle_property(data0, ID_PROP_CANDIDATES, required=False)
    if ids0 is None:
        print("警告：没有找到 Particle Identifier。将使用当前粒子顺序作为 ID。")
        ids0 = np.arange(data0.particles.count, dtype=np.int64)
    else:
        ids0 = to_int_array(ids0)

    order0 = np.argsort(ids0)
    master_ids = ids0[order0]
    n_atoms = len(master_ids)

    states = np.full((n_frames, n_atoms), -1, dtype=np.int16)
    omega_matrix = np.zeros((n_frames, n_atoms), dtype=bool)
    twin_matrix = np.zeros((n_frames, n_atoms), dtype=bool)

    frame_info_rows = []

    structure_prop_used = None
    bda_prop_used = None

    for f in range(n_frames):
        data = pipeline.compute(f)

        ids, _ = get_particle_property(data, ID_PROP_CANDIDATES, required=False)
        if ids is None:
            ids = np.arange(data.particles.count, dtype=np.int64)
        else:
            ids = to_int_array(ids)

        structure, structure_name = get_particle_property(
            data, STRUCTURE_PROP_CANDIDATES, required=True
        )
        structure = to_int_array(structure)
        structure_prop_used = structure_name

        bda, bda_name = get_particle_property(
            data, BDA_PROP_CANDIDATES, required=False
        )
        if bda is not None:
            bda = to_int_array(bda)
            bda_prop_used = bda_name

        state, omega_mask, twin_mask = build_state(structure, bda)

        # 对齐到 master_ids
        order = np.argsort(ids)
        ids_sorted = ids[order]

        if len(ids_sorted) != n_atoms or not np.array_equal(ids_sorted, master_ids):
            # 允许部分缺失，但需要按 ID 对齐
            pos = np.searchsorted(master_ids, ids_sorted)
            valid = (pos >= 0) & (pos < n_atoms) & (master_ids[pos] == ids_sorted)

            states[f, pos[valid]] = state[order][valid]
            omega_matrix[f, pos[valid]] = omega_mask[order][valid]
            twin_matrix[f, pos[valid]] = twin_mask[order][valid]
        else:
            states[f, :] = state[order]
            omega_matrix[f, :] = omega_mask[order]
            twin_matrix[f, :] = twin_mask[order]

        timestep = data.attributes.get("Timestep", np.nan)
        frame_info_rows.append({
            "frame": f,
            "timestep": timestep,
            "strain": frame_to_strain(f),
        })

        if f % max(1, n_frames // 10) == 0:
            print(f"  已处理帧 {f}/{n_frames - 1}")

    pd.DataFrame(frame_info_rows).to_csv(out_dir / "frame_info.csv", index=False)

    # 输出状态分数
    fraction_df = compute_state_fraction(states)
    fraction_df.to_csv(out_dir / "state_fraction_vs_frame.csv", index=False)

    # 输出转移矩阵
    matrix = compute_transition_matrix(states, lag_frames=LAG_FRAMES)
    matrix_df = matrix_to_dataframe(matrix)
    matrix_df.to_csv(out_dir / f"transition_matrix_counts_lag{LAG_FRAMES}.csv")

    matrix_norm = normalize_rows(matrix)
    matrix_norm_df = matrix_to_dataframe(matrix_norm)
    matrix_norm_df.to_csv(out_dir / f"transition_matrix_row_normalized_lag{LAG_FRAMES}.csv")

    # 输出长表转移边，便于后续自定义画图
    transition_long = []
    for i, src in enumerate(STATE_NAMES):
        for j, tgt in enumerate(STATE_NAMES):
            transition_long.append({
                "source_state": src,
                "target_state": tgt,
                "count": int(matrix[i, j]),
                "row_normalized_probability": float(matrix_norm[i, j]),
            })
    pd.DataFrame(transition_long).to_csv(
        out_dir / f"transition_edges_long_lag{LAG_FRAMES}.csv",
        index=False
    )

    # 前驱性分析
    summary_df, metrics_df, onset_df, lag_df = compute_onset_analysis(
        states, omega_matrix, twin_matrix
    )

    summary_df.to_csv(out_dir / "omega_twin_onset_class_summary.csv", index=False)
    metrics_df.to_csv(out_dir / "omega_twin_onset_metrics.csv", index=False)
    onset_df.to_csv(out_dir / "omega_twin_onset_per_atom.csv", index=False)
    lag_df.to_csv(out_dir / "omega_twin_lag_distribution.csv", index=False)

    # 桑基图
    frame_indices = list(range(0, n_frames, SANKEY_EVERY_N_FRAMES))
    if frame_indices[-1] != n_frames - 1:
        frame_indices.append(n_frames - 1)

    edges_df = build_sankey_edges(states, frame_indices)
    edges_df.to_csv(out_dir / "sankey_edges.csv", index=False)
    write_sankey_html(edges_df, out_dir / "sankey_atom_state_transition.html")

    # 输出本算例的简短判断
    metrics = metrics_df.iloc[0].to_dict()

    print("\n关键指标：")
    print(metrics_df.T)

    precursor = metrics["fraction_twin_within_window_after_omega_given_omega"]
    competition = metrics["fraction_omega_competition_like_given_omega"]
    omega_before = metrics["fraction_omega_before_twin_given_both"]
    twin_before = metrics["fraction_twin_before_omega_given_both"]

    print("\n初步机制判断：")

    if np.isfinite(precursor) and np.isfinite(competition):
        if precursor > 0.5 and omega_before > twin_before:
            print(
                "  倾向：omega 具有孪晶前驱体特征。"
                "  依据：较多 omega 原子在窗口内转为 twin/tb，且 omega 先于 twin 的比例更高。"
            )
        elif competition > 0.5 and precursor < 0.3:
            print(
                "  倾向：omega 与 twin 更像竞争或旁路关系。"
                "  依据：多数 omega 原子在设定窗口内没有转为 twin/tb。"
            )
        else:
            print(
                "  倾向：omega 与 twin 可能是同步耦合或 twin-boundary 诱导关系。"
                "  需要结合空间距离和 cluster tracking 判断。"
            )
    else:
        print("  omega 或 twin 数量不足，无法稳定判断。")

    print(f"\n输出目录：{out_dir}")
    print(f"使用结构属性：{structure_prop_used}")
    print(f"使用 BDA 属性：{bda_prop_used}")

    return metrics_df


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    for case in CASES:
        metrics_df = process_case(case)
        metrics_df.insert(0, "case", case["name"])
        all_metrics.append(metrics_df)

    if all_metrics:
        all_metrics_df = pd.concat(all_metrics, ignore_index=True)
        all_metrics_df.to_csv(
            OUTPUT_ROOT / "all_cases_omega_twin_onset_metrics.csv",
            index=False
        )

    print("\n全部完成。")


if __name__ == "__main__":
    main()