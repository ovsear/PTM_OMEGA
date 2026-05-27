# -*- coding: utf-8 -*-
"""
visualize_transition_results.py

用途：
基于前一个脚本输出的后处理 csv，批量生成 Ti20Nb / Ti30Nb / Ti40Nb 的可视化图片。

主要输出：
1. 单个成分的状态分数曲线
2. 单个成分的状态堆叠面积图
3. 三个成分的状态分数对比曲线
4. 状态转移矩阵热图
5. omega 相关状态转移柱状图
6. omega-twin onset 滞后分布
7. omega 前驱体 / 竞争 / twin-boundary 诱导指标对比
8. Sankey 桑基图 html
9. 可选：omega-twin 空间概率曲线
10. 可选：omega-energy 半径关系曲线

运行方式：
直接在 IDE 中运行。先修改“手动参数区”。
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 手动参数区
# ============================================================

RESULT_ROOT = Path(r"C:\Users\017\Desktop\TiNb_Data\Code\transition_sankey_results")

CASES = ["Ti20Nb", "Ti30Nb", "Ti40Nb"]

OUTPUT_DIR = RESULT_ROOT / "_figures_all"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 横坐标优先使用 strain。若 csv 中 strain 全是空值，则自动用 frame。
X_AXIS_MODE = "auto"   # "auto", "strain", "frame"

# 图片格式
FIG_DPI = 300
SAVE_PDF = True

# 是否画 Plotly 桑基图
MAKE_SANKEY = True

# 桑基图中最小显示流量。太小会导致图非常乱。
MIN_SANKEY_COUNT = 100

# 需要重点关注的状态
STATE_ORDER = [
    "BCC",
    "FCC_like",
    "OMEGA",
    "TWIN",
    "TB",
    "OMEGA_TWIN",
    "OMEGA_TB",
    "OTHER",
]

# 颜色可以后续按论文风格再改
STATE_COLORS = {
    "BCC": "#4C78A8",
    "FCC_like": "#F58518",
    "OMEGA": "#E45756",
    "TWIN": "#72B7B2",
    "TB": "#54A24B",
    "OMEGA_TWIN": "#B279A2",
    "OMEGA_TB": "#FF9DA6",
    "OTHER": "#9D9D9D",
}

CASE_COLORS = {
    "Ti20Nb": "#4C78A8",
    "Ti30Nb": "#F58518",
    "Ti40Nb": "#E45756",
}

# 可选：如果你已经有 omega-twin 空间概率 csv，可以填这里。
# 要求 csv 中至少有半径列 r/radius/distance 之一；
# 概率列脚本会自动猜 P_omega_given_tb / P_tb_given_omega 等名字。
DISTANCE_FILES = {
    # "Ti20Nb": r"C:\...\omega_twin_distance_ti20nb.csv",
    # "Ti30Nb": r"C:\...\omega_twin_distance_ti30nb.csv",
    # "Ti40Nb": r"C:\...\omega_twin_distance_ti40nb.csv",
}

# 可选：如果你已经有 omega-energy 半径关系 csv，可以填这里。
ENERGY_FILES = {
    # "Ti20Nb": r"C:\...\omega_energy_radius_summary_ti20nb.csv",
    # "Ti30Nb": r"C:\...\omega_energy_radius_summary_ti30nb.csv",
    # "Ti40Nb": r"C:\...\omega_energy_radius_summary_ti40nb.csv",
}


# ============================================================
# 基础工具函数
# ============================================================

def savefig(fig, path_base):
    """同时保存 png 和可选 pdf。"""
    path_base = Path(path_base)
    fig.savefig(path_base.with_suffix(".png"), dpi=FIG_DPI, bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def read_csv_safe(path):
    path = Path(path)
    if not path.exists():
        warnings.warn(f"文件不存在，跳过：{path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        warnings.warn(f"读取失败，跳过：{path}\n{e}")
        return None


def read_csv_matrix(path):
    path = Path(path)
    if not path.exists():
        warnings.warn(f"文件不存在，跳过：{path}")
        return None
    try:
        return pd.read_csv(path, index_col=0)
    except Exception as e:
        warnings.warn(f"读取矩阵失败，跳过：{path}\n{e}")
        return None


def choose_x(df):
    """自动选择 frame 或 strain 作为横坐标。"""
    if X_AXIS_MODE == "frame":
        return df["frame"].to_numpy(), "Frame"
    if X_AXIS_MODE == "strain":
        return df["strain"].to_numpy(), "Strain"

    # auto
    if "strain" in df.columns:
        x = df["strain"].to_numpy()
        if np.isfinite(x).sum() > 0 and np.nanmax(x) > 0:
            return x, "Strain"
    return df["frame"].to_numpy(), "Frame"


def get_fraction_columns(df):
    cols = []
    for st in STATE_ORDER:
        c = f"fraction_{st}"
        if c in df.columns:
            cols.append(c)
    return cols


def case_dir(case):
    return RESULT_ROOT / case


def nice_state_name_from_fraction_col(col):
    return col.replace("fraction_", "")


def ensure_all_case_dirs():
    missing = []
    for case in CASES:
        if not case_dir(case).exists():
            missing.append(str(case_dir(case)))
    if missing:
        print("以下算例目录不存在，相关图会跳过：")
        for m in missing:
            print("  ", m)


# ============================================================
# 图 1：单个成分状态分数曲线
# ============================================================

def plot_state_fraction_lines(case):
    path = case_dir(case) / "state_fraction_vs_frame.csv"
    df = read_csv_safe(path)
    if df is None:
        return

    frac_cols = get_fraction_columns(df)
    if not frac_cols:
        warnings.warn(f"{case} 没有找到 fraction_* 列")
        return

    x, xlabel = choose_x(df)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    for col in frac_cols:
        state = nice_state_name_from_fraction_col(col)
        y = df[col].to_numpy()
        ax.plot(
            x,
            y,
            label=state,
            linewidth=1.8,
            color=STATE_COLORS.get(state, None),
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Atomic fraction")
    ax.set_title(f"{case}: state fraction evolution")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    ax.grid(True, alpha=0.25)

    out = OUTPUT_DIR / f"{case}_state_fraction_lines"
    savefig(fig, out)


# ============================================================
# 图 2：单个成分状态分数堆叠面积图
# ============================================================

def plot_state_fraction_stack(case):
    path = case_dir(case) / "state_fraction_vs_frame.csv"
    df = read_csv_safe(path)
    if df is None:
        return

    frac_cols = get_fraction_columns(df)
    if not frac_cols:
        return

    x, xlabel = choose_x(df)

    states = [nice_state_name_from_fraction_col(c) for c in frac_cols]
    ys = [df[c].to_numpy() for c in frac_cols]
    colors = [STATE_COLORS.get(s, None) for s in states]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.stackplot(x, ys, labels=states, colors=colors, alpha=0.85)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Atomic fraction")
    ax.set_title(f"{case}: stacked state fractions")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=2, fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.2)

    out = OUTPUT_DIR / f"{case}_state_fraction_stack"
    savefig(fig, out)


# ============================================================
# 图 3：跨成分对比，每个状态一张图
# ============================================================

def plot_state_fraction_compare_by_state():
    state_to_case_data = {}

    for case in CASES:
        path = case_dir(case) / "state_fraction_vs_frame.csv"
        df = read_csv_safe(path)
        if df is None:
            continue

        x, xlabel = choose_x(df)

        for state in STATE_ORDER:
            col = f"fraction_{state}"
            if col not in df.columns:
                continue
            state_to_case_data.setdefault(state, [])
            state_to_case_data[state].append((case, x, df[col].to_numpy(), xlabel))

    for state, data_list in state_to_case_data.items():
        fig, ax = plt.subplots(figsize=(7.2, 4.6))

        xlabel = "Frame"
        for case, x, y, xlabel in data_list:
            ax.plot(
                x,
                y,
                label=case,
                linewidth=2.0,
                color=CASE_COLORS.get(case, None),
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"Fraction of {state}")
        ax.set_title(f"Composition dependence of {state}")
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.25)

        out = OUTPUT_DIR / f"compare_fraction_{state}"
        savefig(fig, out)


# ============================================================
# 图 4：转移矩阵热图
# ============================================================

def plot_transition_heatmap(case):
    path = case_dir(case) / "transition_matrix_row_normalized_lag1.csv"
    mat = read_csv_matrix(path)
    if mat is None:
        return

    # 按 STATE_ORDER 重排，缺失的自动忽略
    rows = [s for s in STATE_ORDER if s in mat.index]
    cols = [s for s in STATE_ORDER if s in mat.columns]
    mat = mat.loc[rows, cols]

    values = mat.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    im = ax.imshow(values, vmin=0, vmax=np.nanmax(values), aspect="auto")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Transition probability")

    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(rows)

    ax.set_xlabel("State at t + Δt")
    ax.set_ylabel("State at t")
    ax.set_title(f"{case}: row-normalized transition matrix")

    # 标数字
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isfinite(v) and v > 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)

    out = OUTPUT_DIR / f"{case}_transition_matrix_heatmap"
    savefig(fig, out)


# ============================================================
# 图 5：omega 相关转移柱状图
# ============================================================

def plot_omega_transition_bars(case):
    path = case_dir(case) / "transition_edges_long_lag1.csv"
    df = read_csv_safe(path)
    if df is None:
        return

    required = {"source_state", "target_state"}
    if not required.issubset(df.columns):
        warnings.warn(f"{case} transition_edges_long_lag1.csv 缺少必要列")
        return

    value_col = None
    if "row_normalized_probability" in df.columns:
        value_col = "row_normalized_probability"
        ylabel = "Row-normalized probability"
    elif "count" in df.columns:
        value_col = "count"
        ylabel = "Count"
    else:
        warnings.warn(f"{case} 没有 count 或 row_normalized_probability")
        return

    # omega 作为源状态
    src_omega = df[df["source_state"].isin(["OMEGA", "OMEGA_TWIN", "OMEGA_TB"])].copy()
    if src_omega.empty:
        return

    grouped = (
        src_omega.groupby("target_state")[value_col]
        .sum()
        .reindex(STATE_ORDER)
        .dropna()
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    colors = [STATE_COLORS.get(s, None) for s in grouped.index]

    ax.bar(grouped.index, grouped.values, color=colors)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Target state")
    ax.set_title(f"{case}: transitions from omega-related states")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.25)

    out = OUTPUT_DIR / f"{case}_omega_to_state_bar"
    savefig(fig, out)

    # twin/tb 作为源状态到 omega
    src_twin = df[df["source_state"].isin(["TWIN", "TB"])].copy()
    if src_twin.empty:
        return

    grouped2 = (
        src_twin.groupby("target_state")[value_col]
        .sum()
        .reindex(STATE_ORDER)
        .dropna()
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    colors = [STATE_COLORS.get(s, None) for s in grouped2.index]

    ax.bar(grouped2.index, grouped2.values, color=colors)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Target state")
    ax.set_title(f"{case}: transitions from twin/TB states")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.25)

    out = OUTPUT_DIR / f"{case}_twin_tb_to_state_bar"
    savefig(fig, out)


# ============================================================
# 图 6：omega-twin onset 滞后分布
# ============================================================

def plot_lag_distribution(case):
    path = case_dir(case) / "omega_twin_lag_distribution.csv"
    df = read_csv_safe(path)
    if df is None:
        return

    if "delta_frame_twin_minus_omega" not in df.columns:
        warnings.warn(f"{case} lag_distribution 缺少 delta_frame_twin_minus_omega")
        return

    if "fraction_given_both" in df.columns:
        ycol = "fraction_given_both"
        ylabel = "Fraction given both"
    elif "count" in df.columns:
        ycol = "count"
        ylabel = "Count"
    else:
        return

    x = df["delta_frame_twin_minus_omega"].to_numpy()
    y = df[ycol].to_numpy()

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(x, y, width=0.8)

    ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Δframe = twin onset frame - omega onset frame")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{case}: omega-twin onset lag distribution")
    ax.grid(True, axis="y", alpha=0.25)

    # 区域标注
    y_max = np.nanmax(y) if len(y) else 1
    ax.text(np.nanmin(x), y_max * 0.95, "twin first", ha="left", va="top", fontsize=9)
    ax.text(np.nanmax(x), y_max * 0.95, "omega first", ha="right", va="top", fontsize=9)

    out = OUTPUT_DIR / f"{case}_omega_twin_lag_distribution"
    savefig(fig, out)


# ============================================================
# 图 7：onset 分类堆叠柱状图
# ============================================================

def plot_onset_class_summary_all_cases():
    rows = []

    for case in CASES:
        path = case_dir(case) / "omega_twin_onset_class_summary.csv"
        df = read_csv_safe(path)
        if df is None:
            continue

        if not {"class", "fraction_all_atoms"}.issubset(df.columns):
            continue

        for _, r in df.iterrows():
            rows.append({
                "case": case,
                "class": r["class"],
                "fraction": r["fraction_all_atoms"],
            })

    if not rows:
        return

    data = pd.DataFrame(rows)
    pivot = data.pivot_table(
        index="case",
        columns="class",
        values="fraction",
        aggfunc="sum",
        fill_value=0,
    )

    class_order = [
        "omega_before_twin",
        "simultaneous",
        "twin_before_omega",
        "omega_only",
        "twin_only",
        "none",
    ]
    class_order = [c for c in class_order if c in pivot.columns]
    pivot = pivot.reindex(CASES).fillna(0)

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    bottom = np.zeros(len(pivot))

    class_colors = {
        "omega_before_twin": "#E45756",
        "simultaneous": "#B279A2",
        "twin_before_omega": "#54A24B",
        "omega_only": "#F58518",
        "twin_only": "#72B7B2",
        "none": "#B0B0B0",
    }

    x = np.arange(len(pivot.index))

    for cls in class_order:
        values = pivot[cls].to_numpy()
        ax.bar(
            x,
            values,
            bottom=bottom,
            label=cls,
            color=class_colors.get(cls, None),
        )
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel("Fraction of all atoms")
    ax.set_title("Omega-twin onset class summary")
    ax.legend(frameon=False, fontsize=9, ncol=2)
    ax.grid(True, axis="y", alpha=0.25)

    out = OUTPUT_DIR / "all_cases_onset_class_stacked_bar"
    savefig(fig, out)


# ============================================================
# 图 8：前驱体 / 竞争 / twin-boundary 诱导指标对比
# ============================================================

def plot_metrics_comparison():
    rows = []

    for case in CASES:
        path = case_dir(case) / "omega_twin_onset_metrics.csv"
        df = read_csv_safe(path)
        if df is None or df.empty:
            continue

        r = df.iloc[0].to_dict()
        r["case"] = case
        rows.append(r)

    if not rows:
        return

    data = pd.DataFrame(rows)

    metric_cols = [
        "fraction_omega_before_twin_given_both",
        "fraction_twin_before_omega_given_both",
        "fraction_simultaneous_given_both",
        "fraction_twin_within_window_after_omega_given_omega",
        "fraction_omega_competition_like_given_omega",
    ]

    metric_cols = [c for c in metric_cols if c in data.columns]

    if not metric_cols:
        return

    # 保存整合指标
    data.to_csv(OUTPUT_DIR / "all_cases_onset_metrics_merged.csv", index=False)

    # 每个指标一张柱状图
    for metric in metric_cols:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))

        values = data.set_index("case").reindex(CASES)[metric]
        x = np.arange(len(values))

        colors = [CASE_COLORS.get(c, None) for c in values.index]
        ax.bar(x, values.to_numpy(dtype=float), color=colors)

        ax.set_xticks(x)
        ax.set_xticklabels(values.index)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction")
        ax.set_title(metric)
        ax.grid(True, axis="y", alpha=0.25)

        out = OUTPUT_DIR / f"compare_metric_{metric}"
        savefig(fig, out)

    # 汇总雷达图风格的极坐标图
    labels = [
        "ω before twin",
        "twin before ω",
        "simultaneous",
        "ω→twin window",
        "ω competition",
    ]
    labels = labels[:len(metric_cols)]

    angles = np.linspace(0, 2 * np.pi, len(metric_cols), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(6.6, 6.2))
    ax = plt.subplot(111, polar=True)

    for _, row in data.iterrows():
        case = row["case"]
        vals = [row[m] if pd.notna(row[m]) else 0 for m in metric_cols]
        vals += vals[:1]
        ax.plot(
            angles,
            vals,
            label=case,
            linewidth=2,
            color=CASE_COLORS.get(case, None),
        )
        ax.fill(angles, vals, alpha=0.08, color=CASE_COLORS.get(case, None))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Omega-twin temporal relationship metrics", y=1.08)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.35, 1.10))

    out = OUTPUT_DIR / "all_cases_temporal_metrics_radar"
    savefig(fig, out)


# ============================================================
# 图 9：Plotly 桑基图
# ============================================================

def plot_sankey(case):
    if not MAKE_SANKEY:
        return

    try:
        import plotly.graph_objects as go
    except ImportError:
        warnings.warn("未安装 plotly，跳过桑基图。可运行 pip install plotly")
        return

    path = case_dir(case) / "sankey_edges.csv"
    df = read_csv_safe(path)
    if df is None or df.empty:
        return

    required = {"source", "target", "count"}
    if not required.issubset(df.columns):
        warnings.warn(f"{case} sankey_edges.csv 缺少 source/target/count")
        return

    df = df[df["count"] >= MIN_SANKEY_COUNT].copy()
    if df.empty:
        warnings.warn(f"{case} 桑基图经过 MIN_SANKEY_COUNT 过滤后为空")
        return

    labels = pd.Index(pd.concat([df["source"], df["target"]]).unique())
    label_to_id = {lab: i for i, lab in enumerate(labels)}

    source = df["source"].map(label_to_id).to_numpy()
    target = df["target"].map(label_to_id).to_numpy()
    value = df["count"].to_numpy()

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=12,
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
        title_text=f"{case}: atom-state transition Sankey diagram",
        font_size=10,
        width=1600,
        height=900,
    )

    out = OUTPUT_DIR / f"{case}_sankey_transition.html"
    fig.write_html(str(out))
    print(f"已输出桑基图：{out}")


# ============================================================
# 图 10：可选 omega-twin 空间概率曲线
# ============================================================

def find_column_by_keywords(df, keywords):
    cols = list(df.columns)
    lower_map = {c: c.lower() for c in cols}
    for c, lc in lower_map.items():
        ok = True
        for kw in keywords:
            if kw.lower() not in lc:
                ok = False
                break
        if ok:
            return c
    return None


def plot_optional_distance_probability():
    if not DISTANCE_FILES:
        print("未设置 DISTANCE_FILES，跳过 omega-twin 空间概率曲线。")
        return

    curves = []

    for case, path in DISTANCE_FILES.items():
        df = read_csv_safe(path)
        if df is None:
            continue

        r_col = (
            find_column_by_keywords(df, ["radius"])
            or find_column_by_keywords(df, ["distance"])
            or find_column_by_keywords(df, ["r"])
        )

        p_omega_tb_col = (
            find_column_by_keywords(df, ["omega", "tb"])
            or find_column_by_keywords(df, ["omega", "twin"])
        )

        p_tb_omega_col = (
            find_column_by_keywords(df, ["tb", "omega"])
            or find_column_by_keywords(df, ["twin", "omega"])
        )

        if r_col is None:
            warnings.warn(f"{case} 距离概率文件中没有识别到半径列")
            continue

        curves.append((case, df, r_col, p_omega_tb_col, p_tb_omega_col))

    # P_omega|TB
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    any_line = False

    for case, df, r_col, p1, p2 in curves:
        if p1 is None:
            continue
        ax.plot(
            df[r_col],
            df[p1],
            label=case,
            linewidth=2,
            color=CASE_COLORS.get(case, None),
        )
        any_line = True

    if any_line:
        ax.set_xlabel("r")
        ax.set_ylabel("Probability")
        ax.set_title("P(omega | twin boundary) versus r")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.25)
        savefig(fig, OUTPUT_DIR / "compare_P_omega_given_TB")
    else:
        plt.close(fig)

    # P_TB|omega
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    any_line = False

    for case, df, r_col, p1, p2 in curves:
        if p2 is None:
            continue
        ax.plot(
            df[r_col],
            df[p2],
            label=case,
            linewidth=2,
            color=CASE_COLORS.get(case, None),
        )
        any_line = True

    if any_line:
        ax.set_xlabel("r")
        ax.set_ylabel("Probability")
        ax.set_title("P(twin boundary | omega) versus r")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.25)
        savefig(fig, OUTPUT_DIR / "compare_P_TB_given_omega")
    else:
        plt.close(fig)


# ============================================================
# 图 11：可选 omega-energy 半径关系
# ============================================================

def plot_optional_energy_radius():
    if not ENERGY_FILES:
        print("未设置 ENERGY_FILES，跳过 omega-energy 半径关系曲线。")
        return

    loaded = []

    for case, path in ENERGY_FILES.items():
        df = read_csv_safe(path)
        if df is None:
            continue

        r_col = (
            find_column_by_keywords(df, ["radius"])
            or find_column_by_keywords(df, ["distance"])
            or find_column_by_keywords(df, ["r"])
        )

        # 尽量自动识别能量相关列
        energy_cols = []
        for c in df.columns:
            lc = c.lower()
            if ("energy" in lc or "pe" in lc or "delta_e" in lc or "de" == lc) and c != r_col:
                if pd.api.types.is_numeric_dtype(df[c]):
                    energy_cols.append(c)

        if r_col is None or not energy_cols:
            warnings.warn(f"{case} energy 文件没有识别到 r 列或能量列")
            continue

        loaded.append((case, df, r_col, energy_cols))

    if not loaded:
        return

    # 每个能量列单独比较
    all_energy_names = sorted(set(c for _, _, _, cols in loaded for c in cols))

    for e_col in all_energy_names:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        any_line = False

        for case, df, r_col, energy_cols in loaded:
            if e_col not in energy_cols:
                continue
            ax.plot(
                df[r_col],
                df[e_col],
                label=case,
                linewidth=2,
                color=CASE_COLORS.get(case, None),
            )
            any_line = True

        if not any_line:
            plt.close(fig)
            continue

        ax.set_xlabel("r")
        ax.set_ylabel(e_col)
        ax.set_title(f"Omega-neighborhood energy profile: {e_col}")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.25)

        safe_name = e_col.replace("/", "_").replace("\\", "_").replace(" ", "_")
        savefig(fig, OUTPUT_DIR / f"compare_energy_radius_{safe_name}")


# ============================================================
# 总控
# ============================================================

def main():
    ensure_all_case_dirs()

    print(f"结果根目录：{RESULT_ROOT}")
    print(f"图片输出目录：{OUTPUT_DIR}")

    for case in CASES:
        print(f"\n处理可视化：{case}")

        plot_state_fraction_lines(case)
        plot_state_fraction_stack(case)
        plot_transition_heatmap(case)
        plot_omega_transition_bars(case)
        plot_lag_distribution(case)
        plot_sankey(case)

    print("\n绘制跨成分对比图")
    plot_state_fraction_compare_by_state()
    plot_onset_class_summary_all_cases()
    plot_metrics_comparison()

    print("\n绘制可选空间/能量统计图")
    plot_optional_distance_probability()
    plot_optional_energy_radius()

    print("\n全部可视化完成。")


if __name__ == "__main__":
    main()