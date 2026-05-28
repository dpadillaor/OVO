"""OVO Results Dashboard — Streamlit GUI.

Run from the repo root:
    streamlit run gui/app.py
"""

import sys
from pathlib import Path

# Make sure the repo root is importable regardless of CWD.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import re

import pandas as pd
import streamlit as st

from ovo.utils.results_utils import (
    filter_experiments,
    filter_scene_results,
    get_available_datasets,
    get_available_dates,
    get_available_fusion_criteria,
    get_available_fusions,
    get_available_jump_counts,
    get_available_labels,
    get_available_methods,
    get_available_slam_configs,
    get_available_noise_levels,
    get_available_scenes,
    load_experiments,
    load_reference_results,
    load_scene_results,
    plot_bar_metrics,
    plot_class_heatmap,
    plot_compare_scene_bars,
    plot_confusion_matrix,
    plot_delta_heatmap,
    plot_fusion_compare,
    plot_instance_counts,
    plot_line_metrics,
    plot_radar_chart,
    save_figure,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(layout="wide", page_title="OVO Results Dashboard")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DATE_COL_CFG = {"Date": st.column_config.DateColumn("Date", format="DD-MM-YYYY")}

_SLAM_NOISE_RE = re.compile(r'Noise|-T\d+p\d+|-R\d+p\d+')


def _slam_base_name(config: str) -> str:
    return _SLAM_NOISE_RE.sub('', config)


def _show_table(df: pd.DataFrame, key: str, label: str = "📋 Table") -> None:
    """Render a collapsible DataFrame."""
    with st.expander(label, expanded=False):
        st.dataframe(df, width='stretch', column_config=_DATE_COL_CFG)


def _save_expander(fig, default_path: str, key_prefix: str) -> None:
    with st.expander("💾 Save figure"):
        path_in = st.text_input("Path", default_path, key=f"{key_prefix}_path")
        dpi_in = st.slider("DPI", 72, 300, 150, key=f"{key_prefix}_dpi")
        if st.button("Save", key=f"{key_prefix}_btn"):
            save_figure(fig, Path(path_in), dpi=dpi_in)
            st.success(f"Saved to {path_in}")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("OVO Results Dashboard")

    output_dir = st.text_input(
        "Output Directory",
        value=st.session_state.get("output_dir", "data/output"),
    )

    load_clicked = st.button("🔄 Load Experiments", width='stretch')

    if load_clicked:
        with st.spinner("Loading experiments…"):
            try:
                df_exp, df_class = load_experiments(Path(output_dir))
                df_scene, df_scene_class = load_scene_results(Path(output_dir))
                for _df in (df_exp, df_class, df_scene, df_scene_class):
                    if not _df.empty and 'Date' in _df.columns:
                        _df["Date"] = pd.to_datetime(_df["Date"], format="%Y%m%d")
                df_ref = load_reference_results(Path("references/paper_results.csv"))
                st.session_state["df_exp"] = df_exp
                st.session_state["df_class"] = df_class
                st.session_state["df_scene"] = df_scene
                st.session_state["df_scene_class"] = df_scene_class
                st.session_state["df_ref"] = df_ref
                st.session_state["output_dir"] = output_dir
            except Exception as exc:
                st.error(f"Failed to load experiments: {exc}")

    # Filters — only shown after data is loaded
    if "df_exp" in st.session_state:
        df_all: "pd.DataFrame" = st.session_state["df_exp"]
        _df_scene_all: "pd.DataFrame" = st.session_state.get("df_scene", pd.DataFrame())

        st.divider()
        st.subheader("Filters")
        st.caption("Leave empty to include all.")

        sel_dates = st.multiselect(
            "Dates",
            options=get_available_dates(df_all),
            format_func=lambda d: d.strftime("%d-%m-%Y"),
        )
        sel_datasets = st.multiselect(
            "Dataset", options=get_available_datasets(df_all)
        )
        sel_methods = st.multiselect(
            "Methods", options=get_available_methods(df_all)
        )
        sel_fusions = st.multiselect(
            "Fusion", options=get_available_fusions(df_all)
        )
        _all_slam_configs = get_available_slam_configs(df_all)
        _slam_base_options = sorted({_slam_base_name(s) for s in _all_slam_configs})
        sel_slam_base = st.multiselect("SLAM", options=_slam_base_options)
        sel_slam = (
            [s for s in _all_slam_configs if _slam_base_name(s) in sel_slam_base]
            if sel_slam_base else []
        )
        sel_noise = st.multiselect(
            "Trans Noise",
            options=get_available_noise_levels(df_all),
            format_func=lambda v: f"{v:.4g}",
        )
        sel_labels = st.multiselect(
            "Labels", options=get_available_labels(df_all)
        )
        sel_fusion_criteria = st.multiselect(
            "Fusion Criteria", options=get_available_fusion_criteria(df_all)
        )
        _exp_id_options = sorted(df_all["Experiment_ID"].dropna().unique().tolist())
        _exp_id_labels = {
            row["Experiment_ID"]: (
                f"{row['Method']}  "
                f"({row['Date'].strftime('%d/%m/%y') if hasattr(row['Date'], 'strftime') else str(row['Date'])[:10]})"
            )
            for _, row in df_all.drop_duplicates("Experiment_ID").iterrows()
        }
        sel_exclude_experiments = st.multiselect(
            "Exclude experiments",
            options=_exp_id_options,
            format_func=lambda x: _exp_id_labels.get(x, x),
        )
        _scene_options = (
            get_available_scenes(_df_scene_all) if not _df_scene_all.empty else []
        )
        sel_scenes = st.multiselect("Scenes", options=_scene_options)

        st.divider()
        _df_ref_sidebar = st.session_state.get("df_ref", pd.DataFrame())
        _paper_options = (
            _df_ref_sidebar["Method"].tolist() if not _df_ref_sidebar.empty else []
        )
        sel_paper = st.multiselect(
            "Paper baselines",
            options=_paper_options,
            default=_paper_options,
            key="sel_paper",
        )
        total = len(df_all)
        df_filtered = filter_experiments(
            df_all,
            dates=sel_dates or None,
            datasets=sel_datasets or None,
            methods=sel_methods or None,
            fusions=sel_fusions or None,
            slam_configs=sel_slam or None,
            noise_levels=sel_noise or None,
            labels=sel_labels or None,
            fusion_criteria=sel_fusion_criteria or None,
            exclude_experiment_ids=sel_exclude_experiments or None,
        )
        after = len(df_filtered)
        st.info(f"**{total}** experiments loaded · **{after}** after filter")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

if "df_exp" not in st.session_state:
    st.info("👈 Enter an output directory and click **Load Experiments** to begin.")
    st.stop()

# df_filtered may not exist if the sidebar block didn't run (e.g. first render
# after a hot-reload where session_state is pre-populated).
if "df_filtered" not in dir():
    df_all = st.session_state["df_exp"]
    df_filtered = df_all.copy()

df_ref = st.session_state.get("df_ref", pd.DataFrame())
_sel_paper = st.session_state.get("sel_paper", [])
if _sel_paper and not df_ref.empty:
    df_ref_sel = df_ref[df_ref["Method"].isin(_sel_paper)]
    df_all = pd.concat([df_all, df_ref_sel], ignore_index=True)
    df_filtered = pd.concat([df_filtered, df_ref_sel], ignore_index=True)

df_class_all: "pd.DataFrame" = st.session_state["df_class"]
df_scene_all: "pd.DataFrame" = st.session_state.get("df_scene", pd.DataFrame())
df_scene_class_all: "pd.DataFrame" = st.session_state.get("df_scene_class", pd.DataFrame())

_dates            = (sel_dates              or None) if "sel_dates"              in dir() else None
_datasets         = (sel_datasets           or None) if "sel_datasets"           in dir() else None
_methods          = (sel_methods            or None) if "sel_methods"            in dir() else None
_fusions          = (sel_fusions            or None) if "sel_fusions"            in dir() else None
_slam             = (sel_slam               or None) if "sel_slam"               in dir() else None
_noise            = (sel_noise              or None) if "sel_noise"              in dir() else None
_labels           = (sel_labels             or None) if "sel_labels"             in dir() else None
_fusion_criteria  = (sel_fusion_criteria    or None) if "sel_fusion_criteria"    in dir() else None
_exclude_experiment_ids = (sel_exclude_experiments or None) if "sel_exclude_experiments" in dir() else None
_scenes           = (sel_scenes             or None) if "sel_scenes"             in dir() else None

df_class_filtered = filter_experiments(
    df_class_all,
    dates=_dates, datasets=_datasets, methods=_methods,
    fusions=_fusions, slam_configs=_slam, noise_levels=_noise,
    labels=_labels, fusion_criteria=_fusion_criteria,
    exclude_experiment_ids=_exclude_experiment_ids,
)
df_scene_filtered = (
    filter_scene_results(
        df_scene_all,
        dates=_dates, datasets=_datasets, methods=_methods,
        fusions=_fusions, slam_configs=_slam, noise_levels=_noise,
        labels=_labels, fusion_criteria=_fusion_criteria, scenes=_scenes,
    )
    if not df_scene_all.empty else df_scene_all
)
df_scene_class_filtered = (
    filter_scene_results(
        df_scene_class_all,
        dates=_dates, datasets=_datasets, methods=_methods,
        fusions=_fusions, slam_configs=_slam, noise_levels=_noise,
        labels=_labels, fusion_criteria=_fusion_criteria, scenes=_scenes,
    )
    if not df_scene_class_all.empty else df_scene_class_all
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_noise, tab_class, tab_confmat, tab_scene, tab_compare = st.tabs([
    "📊 Overview",
    "📈 Noise Analysis",
    "🎨 Per-class",
    "🗂️ Conf. Matrix",
    "🗺️ By Scene",
    "🔬 Compare",
])

# ── Overview ────────────────────────────────────────────────────────────────

with tab_overview:
    _show_table(df_filtered, key="ov_table", label="📋 Experiment Table")

    st.subheader("Bar Chart")
    col1, col2 = st.columns([1, 3])
    with col1:
        ov_segment = st.radio(
            "Segment", ["All", "Head", "Common", "Tail"], key="ov_segment"
        )
        ov_type = st.radio("Type", ["IoU", "Acc"], key="ov_type")
        _prefix = "" if ov_segment == "All" else f"{ov_segment}_"
        _suffix = "mIoU" if ov_type == "IoU" else "mAcc"
        ov_metric = f"{_prefix}{_suffix}"
        ov_hue = st.radio(
            "Color by",
            ["SLAM - Fusion", "Fusion", "Method", "SLAM_Config", "Label"],
            key="ov_hue",
        )
    with col2:
        if df_filtered.empty:
            st.warning("No experiments match the current filters.")
        else:
            df_plot = df_filtered.copy()
            if ov_hue == "SLAM - Fusion":
                import re as _re
                _slam_base = df_plot["SLAM_Config"].str.replace(
                    r'Noise|-T\d+p\d+|-R\d+p\d+', '', regex=True
                )
                df_plot["SLAM - Fusion"] = _slam_base + " - " + df_plot["Fusion"]
            fig = plot_bar_metrics(df_plot, metric=ov_metric, hue=ov_hue)
            st.pyplot(fig, width='stretch')
            _save_expander(fig, "overview_bar.png", "ov_save")

# ── Noise Analysis ──────────────────────────────────────────────────────────

with tab_noise:
    st.subheader("Drift Analysis")

    col1, col2 = st.columns([1, 3])
    with col1:
        noise_x = st.radio(
            "X axis",
            options=["Trans_Noise", "Jump_Count"],
            key="noise_x",
        )
        _noise_metric_opts = [
            m for m in ["mIoU", "mAcc", "AP_agnostic", "AP_agnostic_50"]
            if m in df_filtered.columns or (
                not df_scene_filtered.empty and m in df_scene_filtered.columns
            )
        ] or ["mIoU", "mAcc"]
        noise_metrics = st.multiselect(
            "Metrics",
            options=_noise_metric_opts,
            default=[m for m in ["mIoU", "mAcc"] if m in _noise_metric_opts],
            key="noise_metrics",
        )
        noise_per_scene = st.checkbox("Per-scene lines", key="noise_per_scene")
        noise_hue = st.radio(
            "Color by",
            options=["Method", "SLAM_Config", "Fusion", "Label"],
            key="noise_hue",
        )

    with col2:
        if not noise_metrics:
            st.warning("Select at least one metric.")
        elif noise_per_scene:
            if df_scene_filtered.empty:
                st.warning("No per-scene data. Re-run experiments to generate statistics_{scene}.txt files.")
            else:
                _noise_df = df_scene_filtered.copy()
                if noise_x == "Jump_Count" and _noise_df["Jump_Count"].nunique() < 2:
                    st.info("Only one Jump_Count value in data — line will be a single point per scene.")
                fig = plot_line_metrics(
                    _noise_df, metrics=noise_metrics, x=noise_x,
                    hue=noise_hue, style="Scene",
                )
                st.pyplot(fig, width='stretch')
                _save_expander(fig, "noise_line_scene.png", "noise_sc_save")
        else:
            if df_filtered.empty:
                st.warning("No experiments match the current filters.")
            else:
                _noise_df = df_filtered.copy()
                if noise_x == "Jump_Count" and _noise_df["Jump_Count"].nunique() < 2:
                    st.info("Only one Jump_Count value in data — line will be a single point.")
                fig = plot_line_metrics(
                    _noise_df, metrics=noise_metrics, x=noise_x, hue=noise_hue,
                )
                st.pyplot(fig, width='stretch')
                _save_expander(fig, "noise_line.png", "noise_save")

    _show_table(df_filtered, key="noise_table", label="📋 Experiment Table")

# ── Per-class ────────────────────────────────────────────────────────────────

with tab_class:
    subtab_agg, subtab_scene = st.tabs(["Aggregate", "By Scene"])

    with subtab_agg:
        _show_table(df_filtered, key="cls_agg_table", label="📋 Experiment Table")

        st.subheader("Per-class Heatmap — Aggregate")
        col1, col2 = st.columns([1, 3])
        with col1:
            cls_value = st.radio("Value", ["IoU", "Acc"], key="cls_value")
        with col2:
            if df_class_filtered.empty:
                st.warning("No per-class data available for the current filters.")
            else:
                fig = plot_class_heatmap(df_class_filtered, value=cls_value)
                st.pyplot(fig, width='stretch')
                _save_expander(fig, "class_heatmap.png", "cls_save")

    with subtab_scene:
        st.subheader("Per-class Heatmap — By Scene")
        if df_scene_class_filtered.empty:
            st.info(
                "No per-scene per-class data available. "
                "Re-run experiments to generate `statistics_{scene}.txt` files."
            )
        else:
            _show_table(df_scene_filtered, key="cls_sc_table", label="📋 Scene Table")

            col1, col2 = st.columns([1, 3])
            with col1:
                cls_sc_value = st.radio("Value", ["IoU", "Acc"], key="cls_sc_value")
                available_scenes = get_available_scenes(df_scene_class_filtered)
                cls_sc_scene = st.selectbox("Scene", options=available_scenes, key="cls_sc_scene")
            with col2:
                df_sc_cls_sel = df_scene_class_filtered[
                    df_scene_class_filtered["Scene"] == cls_sc_scene
                ]
                if df_sc_cls_sel.empty:
                    st.warning("No data for the selected scene.")
                else:
                    fig = plot_class_heatmap(df_sc_cls_sel, value=cls_sc_value)
                    st.pyplot(fig, width='stretch')
                    _save_expander(fig, f"class_heatmap_{cls_sc_scene}.png", "cls_sc_save")

# ── Confusion Matrix ─────────────────────────────────────────────────────────

with tab_confmat:
    st.subheader("Confusion Matrix")
    if df_filtered.empty:
        st.warning("No experiments match the current filters.")
    else:
        exp_ids = df_filtered["Experiment_ID"].tolist()
        chosen = st.selectbox("Experiment", options=exp_ids, key="confmat_exp")
        if chosen:
            out_dir = Path(st.session_state["output_dir"])
            fig = plot_confusion_matrix(chosen, out_dir)
            if fig is None:
                st.warning(f"No confmat.png found for **{chosen}**.")
            else:
                st.pyplot(fig, width='stretch')
                _save_expander(fig, f"confmat_{chosen}.png", "cm_save")

# ── By Scene ─────────────────────────────────────────────────────────────────

with tab_scene:
    if df_scene_all.empty:
        st.info(
            "No per-scene statistics found. "
            "Re-run experiments to generate `statistics_{scene}.txt` files."
        )
    else:
        _show_table(df_scene_filtered, key="sc_table", label="📋 Scene Table")

        st.subheader("Bar Chart")
        col1, col2 = st.columns([1, 3])
        with col1:
            sc_metric = st.radio("Metric", ["mIoU", "mAcc"], key="sc_metric")
        with col2:
            if df_scene_filtered.empty:
                st.warning("No scene data matches the current filters.")
            else:
                fig = plot_bar_metrics(
                    df_scene_filtered, metric=sc_metric, x="Scene", hue="Method"
                )
                st.pyplot(fig, width='stretch')
                _save_expander(fig, "scene_bar.png", "sc_bar_save")

        st.subheader("Instance Counts")
        if df_scene_filtered.empty:
            st.warning("No scene data matches the current filters.")
        else:
            fig = plot_instance_counts(df_scene_filtered, x="Scene", hue="Method")
            st.pyplot(fig, width='stretch')
            _save_expander(fig, "instances.png", "inst_save")

        st.subheader("Per-class Heatmap")
        col1, col2 = st.columns([1, 3])
        with col1:
            sc_cls_value = st.radio("Value", ["IoU", "Acc"], key="sc_cls_value")
        with col2:
            if df_scene_class_filtered.empty:
                st.warning("No per-class scene data available for the current filters.")
            else:
                fig = plot_class_heatmap(df_scene_class_filtered, value=sc_cls_value)
                st.pyplot(fig, width='stretch')
                _save_expander(fig, "scene_class_heatmap.png", "sc_cls_save")

# ── Compare ──────────────────────────────────────────────────────────────────

with tab_compare:
    st.caption(
        "Pick experiments directly — independent of sidebar filters. "
        "Baseline is the reference; all deltas are computed against it."
    )

    _all_exp = st.session_state["df_exp"]
    _exp_label_map = {
        row["Experiment_ID"]: (
            f"{row['Label']}  ·  "
            f"{row['Date'].strftime('%d/%m/%y') if hasattr(row['Date'], 'strftime') else str(row['Date'])[:10]}  "
            f"[{row['SLAM_Config']}]"
        )
        for _, row in _all_exp.drop_duplicates("Experiment_ID").iterrows()
    }
    _exp_id_list = sorted(_exp_label_map.keys())

    col_bl, col_cmp = st.columns(2)
    with col_bl:
        cmp_baseline = st.selectbox(
            "Baseline experiment",
            options=_exp_id_list,
            format_func=lambda x: _exp_label_map.get(x, x),
            key="cmp_baseline",
        )
    with col_cmp:
        cmp_compare = st.multiselect(
            "Compare against",
            options=[x for x in _exp_id_list if x != cmp_baseline],
            format_func=lambda x: _exp_label_map.get(x, x),
            key="cmp_compare",
        )

    if not cmp_compare:
        st.info("Select at least one experiment to compare.")
    else:
        _cmp_all_ids = [cmp_baseline] + cmp_compare
        _cmp_df     = _all_exp[_all_exp["Experiment_ID"].isin(_cmp_all_ids)].copy()
        _cmp_scene  = (
            st.session_state["df_scene"][
                st.session_state["df_scene"]["Experiment_ID"].isin(_cmp_all_ids)
            ].copy()
            if "df_scene" in st.session_state and not st.session_state["df_scene"].empty
            else pd.DataFrame()
        )

        # ── Summary table ────────────────────────────────────────────────────
        st.subheader("Summary")
        _sum_metrics = ["mIoU", "mAcc", "AP_agnostic", "AP_agnostic_50",
                        "Num_Instances", "Fusion_Accept_Rate"]
        _sum_cols = ["Experiment_ID", "Label", "SLAM_Config"] + [
            m for m in _sum_metrics if m in _cmp_df.columns
        ]
        _sum_table = _cmp_df[_sum_cols].set_index("Experiment_ID")

        _baseline_row = _sum_table.loc[cmp_baseline] if cmp_baseline in _sum_table.index else None
        _numeric_metrics = [m for m in _sum_metrics if m in _sum_table.columns]

        if _baseline_row is not None:
            delta_rows = {}
            for exp_id in cmp_compare:
                if exp_id not in _sum_table.index:
                    continue
                row = _sum_table.loc[exp_id]
                deltas = {}
                for m in _numeric_metrics:
                    try:
                        d = float(row[m]) - float(_baseline_row[m])
                        arrow = "↑" if d > 0 else ("↓" if d < 0 else "=")
                        deltas[f"Δ {m}"] = f"{arrow} {d:+.4f}"
                    except (TypeError, ValueError):
                        deltas[f"Δ {m}"] = "—"
                delta_rows[exp_id] = deltas
            _delta_df = pd.DataFrame(delta_rows).T
            _display = pd.concat([_sum_table, _delta_df], axis=1).fillna("—")
            with st.expander("📋 Summary table", expanded=True):
                st.dataframe(_display, use_container_width=True)

        # ── Radar chart ──────────────────────────────────────────────────────
        st.subheader("Radar Chart")
        col1, col2 = st.columns([1, 3])
        with col1:
            _radar_metrics_opts = [m for m in
                ["mIoU", "mAcc", "AP_agnostic", "Num_Instances", "Fusion_Accept_Rate"]
                if m in _cmp_df.columns]
            _radar_sel = st.multiselect(
                "Radar axes",
                options=_radar_metrics_opts,
                default=_radar_metrics_opts,
                key="cmp_radar_axes",
            )
        with col2:
            if _radar_sel:
                fig = plot_radar_chart(_cmp_df, _cmp_all_ids, metrics=_radar_sel)
                st.pyplot(fig, width='stretch')
                _save_expander(fig, "compare_radar.png", "cmp_radar_save")

        # ── Scene-by-scene bars ──────────────────────────────────────────────
        st.subheader("Scene-by-Scene")
        if _cmp_scene.empty:
            st.info("No per-scene data available for selected experiments.")
        else:
            col1, col2 = st.columns([1, 3])
            with col1:
                _scene_metric_opts = [m for m in
                    ["mIoU", "mAcc", "AP_agnostic", "Num_Instances"]
                    if m in _cmp_scene.columns]
                _scene_metric_sel = st.multiselect(
                    "Metrics",
                    options=_scene_metric_opts,
                    default=_scene_metric_opts[:3],
                    key="cmp_scene_metrics",
                )
            with col2:
                if _scene_metric_sel:
                    fig = plot_compare_scene_bars(
                        _cmp_scene, _cmp_all_ids, metrics=_scene_metric_sel
                    )
                    st.pyplot(fig, width='stretch')
                    _save_expander(fig, "compare_scene_bars.png", "cmp_scene_save")

        # ── Delta heatmap ────────────────────────────────────────────────────
        st.subheader("Delta Heatmap (vs Baseline)")
        if _cmp_scene.empty:
            st.info("No per-scene data available.")
        else:
            col1, col2 = st.columns([1, 3])
            with col1:
                _delta_metric_opts = [m for m in
                    ["mIoU", "mAcc", "AP_agnostic", "AP_agnostic_50", "AP_agnostic_25"]
                    if m in _cmp_scene.columns]
                _delta_metric_sel = st.multiselect(
                    "Metrics",
                    options=_delta_metric_opts,
                    default=_delta_metric_opts[:3],
                    key="cmp_delta_metrics",
                )
            with col2:
                if _delta_metric_sel and cmp_compare:
                    fig = plot_delta_heatmap(
                        _cmp_scene, cmp_baseline, cmp_compare,
                        metrics=_delta_metric_sel,
                    )
                    st.pyplot(fig, width='stretch')
                    _save_expander(fig, "compare_delta.png", "cmp_delta_save")

        # ── Scene summary table ──────────────────────────────────────────────
        st.subheader("Scene Summary")
        if not _cmp_scene.empty:
            _sc_sum_metrics = [m for m in
                ["mIoU", "mAcc", "AP_agnostic", "AP_agnostic_50", "Num_Instances", "Fusion_Accept_Rate"]
                if m in _cmp_scene.columns]
            _sc_base = _cmp_scene[_cmp_scene["Experiment_ID"] == cmp_baseline].set_index("Scene")
            _sc_table_rows = []
            for _, row in _cmp_scene.sort_values(["Scene", "Experiment_ID"]).iterrows():
                r = {"Scene": row["Scene"], "Experiment": row["Label"]}
                for m in _sc_sum_metrics:
                    r[m] = round(float(row[m]), 4) if pd.notna(row.get(m)) else float("nan")
                    if row["Experiment_ID"] != cmp_baseline and row["Scene"] in _sc_base.index:
                        try:
                            d = float(row[m]) - float(_sc_base.loc[row["Scene"], m])
                            arrow = "↑" if d > 0 else ("↓" if d < 0 else "=")
                            r[f"Δ {m}"] = f"{arrow}{d:+.4f}"
                        except (TypeError, ValueError, KeyError):
                            r[f"Δ {m}"] = "—"
                _sc_table_rows.append(r)
            _sc_display = pd.DataFrame(_sc_table_rows)
            with st.expander("📋 Scene table (all experiments)", expanded=False):
                st.dataframe(_sc_display, use_container_width=True)

        # ── Fusion behaviour ─────────────────────────────────────────────────
        st.subheader("Fusion Behaviour")
        if _cmp_scene.empty:
            st.info("No per-scene data available.")
        elif "Fusion_Accept_Rate" not in _cmp_scene.columns or _cmp_scene["Fusion_Total"].sum() == 0:
            st.info("No fusion decision data found for selected experiments.")
        else:
            fig = plot_fusion_compare(_cmp_scene, _cmp_all_ids)
            st.pyplot(fig, width='stretch')
            _save_expander(fig, "compare_fusion.png", "cmp_fusion_save")

        # ── Scene deep-dive ──────────────────────────────────────────────────
        st.subheader("Scene Deep-Dive")
        if _cmp_scene.empty:
            st.info("No per-scene data available.")
        else:
            _cmp_scene_class = (
                st.session_state["df_scene_class"][
                    st.session_state["df_scene_class"]["Experiment_ID"].isin(_cmp_all_ids)
                ].copy()
                if "df_scene_class" in st.session_state
                and not st.session_state["df_scene_class"].empty
                else pd.DataFrame()
            )
            _dive_scenes = sorted(_cmp_scene["Scene"].dropna().unique().tolist())
            col1, col2 = st.columns([1, 3])
            with col1:
                dive_scene = st.selectbox(
                    "Scene", options=_dive_scenes, key="cmp_dive_scene"
                )
                dive_cls_value = st.radio(
                    "Value", ["IoU", "Acc"], key="cmp_dive_cls_value"
                )
            with col2:
                _dive_sc_row = _cmp_scene[_cmp_scene["Scene"] == dive_scene]
                if not _dive_sc_row.empty:
                    _dive_cols = ["Experiment_ID", "Label"] + [
                        m for m in _sc_sum_metrics if m in _dive_sc_row.columns
                    ]
                    _dive_metric_table = _dive_sc_row[_dive_cols].set_index("Experiment_ID")
                    st.dataframe(_dive_metric_table, use_container_width=True)

            if not _cmp_scene_class.empty:
                _dive_cls = _cmp_scene_class[_cmp_scene_class["Scene"] == dive_scene]
                if _dive_cls.empty:
                    st.info(f"No per-class data for scene **{dive_scene}**.")
                else:
                    st.markdown(f"**Per-class {dive_cls_value} — {dive_scene}**")
                    fig = plot_class_heatmap(_dive_cls, value=dive_cls_value)
                    st.pyplot(fig, width='stretch')
                    _save_expander(fig, f"compare_cls_{dive_scene}.png", "cmp_dive_cls_save")

            # Fusion breakdown for this scene only
            _dive_fusion = _cmp_scene[_cmp_scene["Scene"] == dive_scene]
            if (
                not _dive_fusion.empty
                and "Fusion_Total" in _dive_fusion.columns
                and _dive_fusion["Fusion_Total"].sum() > 0
            ):
                st.markdown(f"**Fusion decisions — {dive_scene}**")
                fig = plot_fusion_compare(_dive_fusion, _cmp_all_ids)
                st.pyplot(fig, width='stretch')
                _save_expander(fig, f"compare_fusion_{dive_scene}.png", "cmp_dive_fusion_save")
