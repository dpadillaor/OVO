"""Comparison: cov-off vs cov-on under GT jump drift (J2-T0.02-R0.01).

Mirrors the structure of compare_cooc.py but adds a timing block to verify
the covisibility filter does not degrade runtime.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from ovo.utils.results_utils import load_experiments, load_scene_results

OUTPUT_DIR = ROOT / "data" / "output"
DATE = "20260508"

VARIANTS = ["cov-off", "cov-on"]
VARIANT_LONG = {"cov-off": "Sin covisibility", "cov-on": "Con covisibility"}
VARIANT_SHORT = {"cov-off": "Sin Cov", "cov-on": "Con Cov"}
VARIANT_COLORS = {"Sin Cov": "#3a7ebf", "Con Cov": "#e07030"}

EXPECTED_CONFIGS: list[tuple[int, float]] = [(2, 0.02)]
COMP_LABELS = {(2, 0.02): "GT Jump · J2 · T0.02 · R0.01 — office0"}

SEMANTIC_METRICS = ["mIoU", "mAcc", "Head_mIoU", "Common_mIoU", "Tail_mIoU"]
TIMING_KEYS = ["spf", "total_time", "avg_fps", "t_cov_add_keyframe", "t_cov_pairs_query"]


# ── Helpers ─────────────────────────────────────────────────────────────────
def _read_log(path: Path) -> np.ndarray:
    if not path.is_file():
        return np.array([])
    raw = path.read_text().splitlines()
    out: list[float] = []
    for x in raw:
        x = x.strip()
        if not x:
            continue
        try:
            out.append(float(x))
        except ValueError:
            continue
    return np.asarray(out, dtype=float)


def _summarize_log(arr: np.ndarray) -> dict:
    if arr.size == 0:
        return {"n": 0, "total": float("nan"), "mean": float("nan"),
                "max": float("nan"), "min": float("nan")}
    return {
        "n": int(arr.size),
        "total": float(arr.sum()),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
        "min": float(arr.min()),
    }


@st.cache_data(ttl=60)
def _load_timings(experiment_id: str, scene: str = "office0") -> dict[str, dict]:
    log_dir = OUTPUT_DIR / "Replica" / experiment_id / scene / "logger"
    return {key: _summarize_log(_read_log(log_dir / f"{key}.log")) for key in TIMING_KEYS}


@st.cache_data(ttl=60)
def _load_spf_series(experiment_id: str, scene: str = "office0") -> np.ndarray:
    log_dir = OUTPUT_DIR / "Replica" / experiment_id / scene / "logger"
    return _read_log(log_dir / "spf.log")


@st.cache_data(ttl=60)
def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_exp, df_class = load_experiments(OUTPUT_DIR, date_filter=DATE)
    df_scene, _ = load_scene_results(OUTPUT_DIR)
    if not df_scene.empty and "Date" in df_scene.columns:
        df_scene = df_scene[df_scene["Date"] == DATE]
    return df_exp, df_class, df_scene


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Variant"] = df["Label"].map(VARIANT_SHORT)
    return df


# ── Page setup ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Cov off vs on", layout="wide")
st.title("Covisibility filter — overhead y métricas")
st.caption(f"GT con saltos · 2 jumps · T=0.02m · R=0.01° · seed=42 · escena office0 · fecha {DATE}")

if st.button("Refresh"):
    st.cache_data.clear()
    st.rerun()

df_exp, df_class, df_scene = _load()
df = _enrich(df_exp[df_exp["Label"].isin(VARIANTS)]) if not df_exp.empty else pd.DataFrame()


# ── Estado ──────────────────────────────────────────────────────────────────
st.header("Estado de experimentos")
rows = []
for label in VARIANTS:
    match = df[df["Label"] == label] if not df.empty else pd.DataFrame()
    done = not match.empty
    r = match.iloc[0] if done else None
    rows.append({
        "Variante": VARIANT_LONG[label],
        "Estado": "✓ Listo" if done else "⏳ Falta",
        "mIoU":   f"{r['mIoU']:.4f}"        if done else "—",
        "mAcc":   f"{r['mAcc']:.4f}"        if done else "—",
        "AP_agn": f"{r['AP_agnostic']:.4f}" if done and pd.notna(r.get("AP_agnostic")) else "—",
        "F_accept": f"{r['Fusion_Accept_Rate']:.4f}" if done and pd.notna(r.get("Fusion_Accept_Rate")) else "—",
        "Experiment_ID": r["Experiment_ID"] if done else "—",
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

if df.empty or df["Label"].nunique() < 2:
    st.warning("Faltan experimentos para comparar.")
    st.stop()


# ── Timing block ────────────────────────────────────────────────────────────
st.header("Tiempos — overhead del filtro de covisibilidad")
st.caption(
    "**Run stage únicamente.** `add_keyframe` se llama por cada keyframe (O(n) por llamada). "
    "`pairs_query` se llama una vez por fusión global (al cierre de bucle)."
)

timings = {label: _load_timings(df[df["Label"] == label].iloc[0]["Experiment_ID"]) for label in VARIANTS}

# Headline: total runtime delta
t_off = timings["cov-off"]["total_time"]["total"]
t_on  = timings["cov-on"]["total_time"]["total"]
fps_off = timings["cov-off"]["avg_fps"]["total"]
fps_on  = timings["cov-on"]["avg_fps"]["total"]
spf_off = timings["cov-off"]["spf"]["mean"] * 1000
spf_on  = timings["cov-on"]["spf"]["mean"] * 1000

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total runtime (s)", f"{t_on:.2f}", delta=f"{t_on - t_off:+.2f} vs sin",
            delta_color="inverse")
col2.metric("Avg FPS", f"{fps_on:.2f}", delta=f"{fps_on - fps_off:+.2f}",
            delta_color="normal")
col3.metric("Mean SPF (ms)", f"{spf_on:.1f}", delta=f"{spf_on - spf_off:+.1f}",
            delta_color="inverse")
overhead_pct = (t_on - t_off) / t_off * 100 if t_off > 0 else 0.0
col4.metric("Overhead vs sin (%)", f"{overhead_pct:+.2f}%",
            delta=("OK" if overhead_pct <= 5 else "ATENCIÓN"),
            delta_color=("normal" if overhead_pct <= 5 else "inverse"))

st.subheader("Detalle por operación de covisibilidad")
add_t = timings["cov-on"]["t_cov_add_keyframe"]
qry_t = timings["cov-on"]["t_cov_pairs_query"]
detail_rows = [
    {"Op": "add_keyframe", "Llamadas": add_t["n"],
     "Total (s)": f"{add_t['total']:.4f}",
     "Mean (ms)": f"{add_t['mean'] * 1000:.3f}" if add_t["n"] else "—",
     "Max (ms)":  f"{add_t['max']  * 1000:.3f}" if add_t["n"] else "—"},
    {"Op": "pairs_query", "Llamadas": qry_t["n"],
     "Total (s)": f"{qry_t['total']:.4f}",
     "Mean (ms)": f"{qry_t['mean'] * 1000:.3f}" if qry_t["n"] else "—",
     "Max (ms)":  f"{qry_t['max']  * 1000:.3f}" if qry_t["n"] else "—"},
]
total_cov = add_t["total"] + qry_t["total"]
detail_rows.append({"Op": "TOTAL covisibility", "Llamadas": "—",
                    "Total (s)": f"{total_cov:.4f}",
                    "Mean (ms)": "—", "Max (ms)": "—"})
detail_rows.append({"Op": "Diferencia runtime",
                    "Llamadas": "—",
                    "Total (s)": f"{t_on - t_off:+.4f}",
                    "Mean (ms)": "—", "Max (ms)": "—"})
st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)
st.caption(
    f"Total covisibility ≈ {total_cov:.2f}s; diferencia runtime real {t_on - t_off:+.2f}s. "
    "Si los dos se aproximan, el overhead se explica por completo por el filtro."
)


# ── SPF curve ───────────────────────────────────────────────────────────────
st.subheader("SPF por frame")
fig, ax = plt.subplots(figsize=(10, 3.5))
for label in VARIANTS:
    arr = _load_spf_series(df[df["Label"] == label].iloc[0]["Experiment_ID"])
    if arr.size:
        ax.plot(arr * 1000, label=VARIANT_SHORT[label],
                color=VARIANT_COLORS[VARIANT_SHORT[label]], linewidth=0.9)
ax.set_xlabel("frame")
ax.set_ylabel("ms / frame")
ax.set_title("Tiempo por frame")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)


# ── Add_keyframe per-call series ────────────────────────────────────────────
st.subheader("t_cov_add_keyframe por llamada")
arr_add = _read_log(
    OUTPUT_DIR / "Replica" / df[df["Label"] == "cov-on"].iloc[0]["Experiment_ID"]
    / "office0" / "logger" / "t_cov_add_keyframe.log"
)
if arr_add.size:
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(arr_add * 1000, color="#e07030", linewidth=0.9, label="t_cov_add_keyframe")
    ax.set_xlabel("KF index")
    ax.set_ylabel("ms")
    ax.set_title("Coste de añadir KF al grafo (crece con n KFs ya en grafo)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── Semantic metrics delta ──────────────────────────────────────────────────
st.header("Métricas semánticas — Δ (Con − Sin)")

_DELTA_METRICS = [
    ("mIoU", True), ("mAcc", True),
    ("Head_mIoU", True), ("Common_mIoU", True), ("Tail_mIoU", True),
    ("AP_agnostic", True), ("AP_agnostic_50", True), ("AP_agnostic_25", True),
    ("AP", True), ("AP_50", True), ("AP_25", True),
    ("Fusion_Accept_Rate", True),
    ("Fusion_Reject_Centroid", False),
    ("Fusion_Reject_CosSim",   False),
    ("Fusion_Reject_Overlap",  False),
]


def _color(delta: float, higher_is_better: bool) -> str:
    if pd.isna(delta):
        return ""
    if delta > 0:
        return "color:green;font-weight:bold" if higher_is_better else "color:red;font-weight:bold"
    if delta < 0:
        return "color:red;font-weight:bold" if higher_is_better else "color:green;font-weight:bold"
    return ""


b = df[df["Label"] == "cov-off"].iloc[0]
c = df[df["Label"] == "cov-on"].iloc[0]
table_rows = []
for col, hib in _DELTA_METRICS:
    if col not in df.columns:
        continue
    bv = b.get(col, float("nan"))
    cv = c.get(col, float("nan"))
    delta = cv - bv if (pd.notna(bv) and pd.notna(cv)) else float("nan")
    is_int = col in ("Fusion_Reject_Centroid", "Fusion_Reject_CosSim",
                     "Fusion_Reject_Overlap", "Fusion_Total", "Fusion_Accepted")
    fmt = (lambda x: f"{int(x)}" if not pd.isna(x) else "—") if is_int else (lambda x: f"{x:.4f}" if not pd.isna(x) else "—")
    delta_str = "—" if pd.isna(delta) else (f"{int(delta):+d}" if is_int else f"{delta:+.4f}")
    table_rows.append({
        "Métrica": col, "Sin Cov": fmt(bv), "Con Cov": fmt(cv),
        "Δ": delta_str, "_color": _color(delta, hib),
    })

html_rows = ""
for r in table_rows:
    html_rows += (
        f"<tr><td>{r['Métrica']}</td>"
        f"<td>{r['Sin Cov']}</td>"
        f"<td>{r['Con Cov']}</td>"
        f"<td style='{r['_color']}'>{r['Δ']}</td></tr>"
    )
st.markdown(
    f"""<table style="font-size:13px;width:100%;max-width:700px">
    <thead><tr><th>Métrica</th><th>Sin Cov</th><th>Con Cov</th><th>Δ</th></tr></thead>
    <tbody>{html_rows}</tbody></table>""",
    unsafe_allow_html=True,
)


# ── Per-class IoU heatmap ───────────────────────────────────────────────────
st.header("IoU por clase")

if not df_class.empty:
    df_cl = df_class[df_class["Label"].isin(VARIANTS) & (df_class["Date"] == DATE)].copy()
    df_cl["Variant"] = df_cl["Label"].map(VARIANT_SHORT)
    pivot = df_cl.pivot_table(index="Class", columns="Variant", values="IoU", aggfunc="mean")
    active = pivot.index[pivot.max(axis=1).fillna(0) > 0]
    pivot = pivot.loc[active]
    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(6, max(3, len(active) * 0.22)))
        sns.heatmap(pivot, ax=ax, cmap="RdYlGn", vmin=0, vmax=1,
                    linewidths=0.3, linecolor="grey",
                    annot=True, fmt=".2f", annot_kws={"size": 7}, cbar=False)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=7)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)


# ── Raw timings ─────────────────────────────────────────────────────────────
with st.expander("Datos brutos timings"):
    raw = []
    for label in VARIANTS:
        for key, summ in timings[label].items():
            raw.append({
                "Variante": VARIANT_SHORT[label],
                "Métrica":  key,
                "n":        summ["n"],
                "total_s":  None if pd.isna(summ["total"]) else round(summ["total"], 4),
                "mean_ms":  None if pd.isna(summ["mean"]) else round(summ["mean"] * 1000, 3),
                "max_ms":   None if pd.isna(summ["max"]) else round(summ["max"] * 1000, 3),
            })
    st.dataframe(pd.DataFrame(raw), use_container_width=True, hide_index=True)
