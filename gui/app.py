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

import pandas as pd
import streamlit as st

from ovo.utils.results_utils import (
    filter_experiments,
    filter_scene_results,
    get_available_datasets,
    get_available_dates,
    get_available_methods,
    get_available_noise_levels,
    get_available_scenes,
    load_experiments,
    load_reference_results,
    load_scene_results,
    plot_bar_metrics,
    plot_class_heatmap,
    plot_confusion_matrix,
    plot_instance_counts,
    plot_line_metrics,
    save_figure,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(layout="wide", page_title="OVO Results Dashboard")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("OVO Results Dashboard")

    output_dir = st.text_input(
        "Output Directory",
        value=st.session_state.get("output_dir", "data/output"),
    )

    load_clicked = st.button("🔄 Load Experiments", use_container_width=True)

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
        sel_noise = st.multiselect(
            "Trans Noise",
            options=get_available_noise_levels(df_all),
            format_func=lambda v: f"{v:.4g}",
        )

        st.divider()
        show_paper = st.checkbox(
            "Show paper baselines", value=True, key="show_paper"
        )
        total = len(df_all)
        df_filtered = filter_experiments(
            df_all,
            dates=sel_dates or None,
            datasets=sel_datasets or None,
            methods=sel_methods or None,
            noise_levels=sel_noise or None,
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
_show_paper = st.session_state.get("show_paper", True)
if _show_paper and not df_ref.empty:
    df_all = pd.concat([df_all, df_ref], ignore_index=True)
    df_filtered = pd.concat([df_filtered, df_ref], ignore_index=True)

df_class_all: "pd.DataFrame" = st.session_state["df_class"]
df_scene_all: "pd.DataFrame" = st.session_state.get("df_scene", pd.DataFrame())
df_scene_class_all: "pd.DataFrame" = st.session_state.get("df_scene_class", pd.DataFrame())

_dates   = (sel_dates    or None) if "sel_dates"    in dir() else None
_datasets= (sel_datasets or None) if "sel_datasets" in dir() else None
_methods = (sel_methods  or None) if "sel_methods"  in dir() else None
_noise   = (sel_noise    or None) if "sel_noise"    in dir() else None

df_class_filtered = filter_experiments(
    df_class_all,
    dates=_dates, datasets=_datasets, methods=_methods, noise_levels=_noise,
)
df_scene_filtered = (
    filter_scene_results(
        df_scene_all,
        dates=_dates, datasets=_datasets, methods=_methods, noise_levels=_noise,
    )
    if not df_scene_all.empty else df_scene_all
)
df_scene_class_filtered = (
    filter_scene_results(
        df_scene_class_all,
        dates=_dates, datasets=_datasets, methods=_methods, noise_levels=_noise,
    )
    if not df_scene_class_all.empty else df_scene_class_all
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_noise, tab_class, tab_instances, tab_confmat, tab_scene = st.tabs([
    "📊 Overview",
    "📈 Noise Analysis",
    "🎨 Per-class",
    "🔢 Instances",
    "🗂️ Conf. Matrix",
    "🗺️ By Scene",
])

# ── Overview ────────────────────────────────────────────────────────────────

with tab_overview:
    st.subheader("Experiment Table")
    st.dataframe(
        df_filtered,
        use_container_width=True,
        column_config={"Date": st.column_config.DateColumn("Date", format="DD-MM-YYYY")},
    )

    st.subheader("Bar Chart")
    col1, col2 = st.columns([1, 3])
    with col1:
        ov_metric = st.radio("Metric", ["mIoU", "mAcc"], key="ov_metric")
    with col2:
        if df_filtered.empty:
            st.warning("No experiments match the current filters.")
        else:
            fig = plot_bar_metrics(df_filtered, metric=ov_metric)
            st.pyplot(fig, use_container_width=True)
            with st.expander("💾 Save figure"):
                path_in = st.text_input("Path", "overview_bar.png", key="ov_save_path")
                dpi_in = st.slider("DPI", 72, 300, 150, key="ov_dpi")
                if st.button("Save", key="ov_save_btn"):
                    save_figure(fig, Path(path_in), dpi=dpi_in)
                    st.success(f"Saved to {path_in}")

# ── Noise Analysis ──────────────────────────────────────────────────────────

with tab_noise:
    st.subheader("Metrics vs Translational Noise")
    col1, col2 = st.columns([1, 3])
    with col1:
        noise_metrics = st.multiselect(
            "Metrics to display",
            options=["mIoU", "mAcc"],
            default=["mIoU", "mAcc"],
            key="noise_metrics",
        )
    with col2:
        if df_filtered.empty:
            st.warning("No experiments match the current filters.")
        elif not noise_metrics:
            st.warning("Select at least one metric.")
        else:
            fig = plot_line_metrics(df_filtered, metrics=noise_metrics)
            st.pyplot(fig, use_container_width=True)
            with st.expander("💾 Save figure"):
                path_in = st.text_input("Path", "noise_line.png", key="noise_save_path")
                dpi_in = st.slider("DPI", 72, 300, 150, key="noise_dpi")
                if st.button("Save", key="noise_save_btn"):
                    save_figure(fig, Path(path_in), dpi=dpi_in)
                    st.success(f"Saved to {path_in}")

# ── Per-class ────────────────────────────────────────────────────────────────

with tab_class:
    st.subheader("Per-class Heatmap")
    col1, col2 = st.columns([1, 3])
    with col1:
        cls_value = st.radio("Value", ["IoU", "Acc"], key="cls_value")
    with col2:
        if df_class_filtered.empty:
            st.warning("No per-class data available for the current filters.")
        else:
            fig = plot_class_heatmap(df_class_filtered, value=cls_value)
            st.pyplot(fig, use_container_width=True)
            with st.expander("💾 Save figure"):
                path_in = st.text_input("Path", "class_heatmap.png", key="cls_save_path")
                dpi_in = st.slider("DPI", 72, 300, 150, key="cls_dpi")
                if st.button("Save", key="cls_save_btn"):
                    save_figure(fig, Path(path_in), dpi=dpi_in)
                    st.success(f"Saved to {path_in}")

# ── Instances ────────────────────────────────────────────────────────────────

with tab_instances:
    st.subheader("Instance Counts")
    if df_filtered.empty:
        st.warning("No experiments match the current filters.")
    else:
        fig = plot_instance_counts(df_filtered)
        st.pyplot(fig, use_container_width=True)
        with st.expander("💾 Save figure"):
            path_in = st.text_input("Path", "instances.png", key="inst_save_path")
            dpi_in = st.slider("DPI", 72, 300, 150, key="inst_dpi")
            if st.button("Save", key="inst_save_btn"):
                save_figure(fig, Path(path_in), dpi=dpi_in)
                st.success(f"Saved to {path_in}")

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
                st.pyplot(fig, use_container_width=True)
                with st.expander("💾 Save figure"):
                    path_in = st.text_input(
                        "Path", f"confmat_{chosen}.png", key="cm_save_path"
                    )
                    dpi_in = st.slider("DPI", 72, 300, 150, key="cm_dpi")
                    if st.button("Save", key="cm_save_btn"):
                        save_figure(fig, Path(path_in), dpi=dpi_in)
                        st.success(f"Saved to {path_in}")

# ── By Scene ─────────────────────────────────────────────────────────────────

with tab_scene:
    if df_scene_all.empty:
        st.info(
            "No per-scene statistics found. "
            "Re-run experiments to generate `statistics_{scene}.txt` files."
        )
    else:
        st.subheader("Scene Table")
        st.dataframe(df_scene_filtered, use_container_width=True)

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
                st.pyplot(fig, use_container_width=True)
                with st.expander("💾 Save figure"):
                    path_in = st.text_input(
                        "Path", "scene_bar.png", key="sc_bar_save_path"
                    )
                    dpi_in = st.slider("DPI", 72, 300, 150, key="sc_bar_dpi")
                    if st.button("Save", key="sc_bar_save_btn"):
                        save_figure(fig, Path(path_in), dpi=dpi_in)
                        st.success(f"Saved to {path_in}")

        st.subheader("Per-class Heatmap")
        col1, col2 = st.columns([1, 3])
        with col1:
            sc_cls_value = st.radio("Value", ["IoU", "Acc"], key="sc_cls_value")
        with col2:
            if df_scene_class_filtered.empty:
                st.warning("No per-class scene data available for the current filters.")
            else:
                fig = plot_class_heatmap(df_scene_class_filtered, value=sc_cls_value)
                st.pyplot(fig, use_container_width=True)
                with st.expander("💾 Save figure"):
                    path_in = st.text_input(
                        "Path", "scene_class_heatmap.png", key="sc_cls_save_path"
                    )
                    dpi_in = st.slider("DPI", 72, 300, 150, key="sc_cls_dpi")
                    if st.button("Save", key="sc_cls_save_btn"):
                        save_figure(fig, Path(path_in), dpi=dpi_in)
                        st.success(f"Saved to {path_in}")
