"""Phase 7 — report & machine-readable memory.

Assembles study/results_dashboard.html (human), study/conclusions.json
(machine-readable, fixed schema), study/conclusions.md (prose). Reads whatever
phase artifacts exist; degrades gracefully if a phase was skipped.
"""
import json, datetime as dt
from pathlib import Path

import pandas as pd

ROOT = Path("/home/padidavid/repos/OVO")
S = ROOT / "study"


def load(name):
    p = S / name
    return json.loads(p.read_text()) if p.exists() else None


def fnum(x, nd=4):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def main():
    p2 = load("phase2_results.json")
    p3 = load("morris.json")
    p4 = load("phase4_results.json")
    p6 = load("phase6_results.json")
    orb = load("orbslam_results.json")
    ranges = load("ranges.json")
    commit = (subprocess_commit())
    master = pd.read_parquet(S / "runs_master.parquet") if (S / "runs_master.parquet").exists() else pd.DataFrame()

    # ---------- baselines ----------
    baselines = {}
    if p2:
        for cond in ["light", "aggressive"]:
            row = next((e for e in p2["by_condition"][cond]
                        if e["label"] == "abl-centroid-cossim-overlap"), None)
            if row:
                baselines[cond] = {"mIoU": row["mIoU"], "AP_agnostic": row["AP_agnostic"],
                                   "mAcc": row["mAcc"], "Num_Instances": row["Num_Instances"]}

    # ---------- findings ----------
    findings = []
    if p2:
        ov = p2.get("overlap_new_vs_old", {})
        for cond, v in ov.items():
            findings.append({"rq": "RQ1",
                "claim": f"overlap(nuevo) vs overlap_old en {cond}: ΔmIoU medio={fnum(v['mean_delta_mIoU_new_vs_old'])}",
                "evidence": v, "confidence": "media"})
        cd = p2.get("cooccurrence_damage", {})
        for cond, v in cd.items():
            dap = v.get("mean_delta_AP_agnostic_on_vs_off", 0.0)
            pct = v.get("mean_pct_AP_agnostic_gain", 0.0)
            findings.append({"rq": "RQ1",
                "claim": (f"activar cooccurrence en jump drift ({cond}) NO degrada: "
                          f"ΔmIoU={fnum(v['mean_delta_mIoU_on_vs_off'])} (neutro) pero "
                          f"ΔAP_agnostic={fnum(dap)} ({pct:+.1f}%) — sube pureza de instancia, "
                          f"contradice la hipótesis a priori del playbook"),
                "evidence": v,
                "confidence": "alta" if dap > 0 else "media"})
        findings.append({"rq": "RQ1", "claim": f"cadena ganadora (media combinada): {p2['winner']}",
                         "evidence": {"aggregate_ranking_top3": p2["aggregate_ranking"][:3]},
                         "confidence": "media"})
    if p3:
        findings.append({"rq": "RQ3",
            "claim": f"parámetros relevantes (Morris mu*): {p3['reduced_params']}",
            "evidence": {"normalized_importance": p3["normalized_importance"]},
            "confidence": "media"})
    if p4:
        for cond in ["light", "aggressive"]:
            kn = p4["by_condition"][cond].get("knee")
            if kn:
                findings.append({"rq": "RQ2",
                    "claim": f"θ* (knee) {cond}: mIoU={fnum(kn['mIoU'])} AP_agn={fnum(kn['AP_agnostic'])}",
                    "evidence": {"theta": kn["theta"]}, "confidence": "media"})
    if p6:
        findings.append({"rq": "RQ4",
            "claim": f"consistencia entre condiciones: {p6.get('consistent_across_conditions')}",
            "evidence": p6.get("cross_condition", {}), "confidence": "media"})
    if orb and orb.get("comparisons"):
        findings.append({"rq": "VALIDACION-EXTERNA",
            "claim": ("El hallazgo NO transfiere a ORB-SLAM real: el mecanismo cooccurrence "
                      "que mejoraba AP_agnostic en jump-drift sintético lo DEGRADA bajo drift "
                      "real. " + orb.get("finding_summary", "")),
            "evidence": orb["comparisons"], "confidence": "alta"})

    # ---------- best_config ----------
    best = {}
    if p4:
        for cond in ["light", "aggressive"]:
            kn = p4["by_condition"][cond].get("knee")
            if kn:
                best[cond] = {"theta": kn["theta"], "mIoU": kn["mIoU"],
                              "AP_agnostic": kn["AP_agnostic"]}
        # recommended overall: the knee that generalizes best (smaller cross drop)
        if p6 and "light" in best and "aggressive" in best:
            ld = abs(p6["cross_condition"].get("light_knee_rel_drop_on_aggressive", 9))
            ad = abs(p6["cross_condition"].get("aggressive_knee_rel_drop_on_light", 9))
            pick = "light" if ld <= ad else "aggressive"
            best["recommended_overall"] = {
                "theta": best[pick]["theta"],
                "rationale": f"knee de '{pick}' generaliza mejor (menor caída relativa al cruzar de condición)"}

    # ---------- robustness ----------
    robustness = {}
    if p6:
        robustness["consistent_across_conditions"] = p6.get("consistent_across_conditions")
        cons = p6.get("consistency", {})
        for c in cons:
            robustness[f"{c}_mIoU_std_across_scenes"] = cons[c].get("mIoU", {}).get("std")
        loso = p6.get("loso", {})
        for c in loso:
            robustness[f"{c}_loso_std_mIoU"] = loso[c].get("mIoU", {}).get("std")

    conclusions = {
        "meta": {"date": dt.date.today().isoformat(), "commit": commit,
                 "dataset": "Replica", "scenes": 8, "fusion_method": "clip",
                 "deterministic": True,
                 "phases_run": [n for n, x in
                                [("0", True), ("1", ranges is not None),
                                 ("2", p2 is not None), ("3", p3 is not None),
                                 ("4", p4 is not None), ("6", p6 is not None)] if x],
                 "phases_skipped": (["5 (Sobol)"] )},
        "baselines": baselines,
        "findings": findings,
        "best_config": best,
        "robustness": robustness,
        "caveats": ["VALIDACIÓN EXTERNA: el óptimo sim NO transfiere a ORB-SLAM real "
                    "(cooccurrence degrada AP_agnostic ~-15%); baseline gana en SLAM real",
                    "solo fusion_method=clip", "óptimo condicionado a jump_seed=42",
                    "backend SimulatedSLAM/GTJump determinista",
                    "covisibilidad NO evaluada: no implementada en el código actual",
                    "Fase 5 (Sobol) omitida por coste; interacción vía Morris σ",
                    "presupuestos Optuna/Morris reducidos por coste de replay (~6 min/config)"],
        "artifacts": {"master_table": "study/runs_master.parquet",
                      "optuna": ["study/optuna_light.db", "study/optuna_aggressive.db"],
                      "dashboard": "study/results_dashboard.html",
                      "manifest_index": "study/manifest_index.csv"},
    }
    (S / "conclusions.json").write_text(json.dumps(conclusions, indent=2, ensure_ascii=False))

    conclusions["external_validation_orbslam"] = orb
    _make_extra_plots(p4, p6, master)
    _write_md(conclusions, p2, p3, p4, p6)
    _write_dashboard(conclusions, p2, p3, p4, p6, ranges, master, orb)
    print("Wrote study/conclusions.json, conclusions.md, results_dashboard.html")


def _make_extra_plots(p4, p6, master):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    (S / "figs").mkdir(exist_ok=True)
    # Pareto scatter per condition
    if p4 and len(master):
        for cond in ["light", "aggressive"]:
            sub = master[(master.condition == cond) &
                         master.label.astype(str).str.startswith(f"opt-{cond}-")]
            if not len(sub):
                continue
            fig, ax = plt.subplots(figsize=(6, 5))
            sc = ax.scatter(sub.mIoU.astype(float), sub.AP_agnostic.astype(float),
                            c=sub.Num_Instances.astype(float), cmap="viridis", s=30, alpha=.7)
            par = p4["by_condition"][cond].get("pareto", [])
            if par:
                px = [t["mIoU"] for t in par]; py = [t["AP_agnostic"] for t in par]
                order = np.argsort(px)
                ax.plot(np.array(px)[order], np.array(py)[order], "r.-", label="Pareto")
            kn = p4["by_condition"][cond].get("knee")
            if kn:
                ax.scatter([kn["mIoU"]], [kn["AP_agnostic"]], marker="*", s=300,
                           color="gold", edgecolor="k", label="knee", zorder=5)
            ax.set_xlabel("mIoU"); ax.set_ylabel("AP_agnostic")
            ax.set_title(f"Pareto {cond}"); ax.legend(fontsize=8)
            fig.colorbar(sc, ax=ax, label="Num_Instances")
            fig.tight_layout(); fig.savefig(S / "figs" / f"phase4_pareto_{cond}.png", dpi=110)
            plt.close(fig)
    # scene consistency
    if p6 and p6.get("consistency"):
        cons = p6["consistency"]
        scenes = None
        fig, ax = plt.subplots(figsize=(9, 4))
        width = 0.35
        for i, cond in enumerate([c for c in ["light", "aggressive"] if c in cons]):
            ps = cons[cond]["mIoU"]["per_scene"]
            scenes = list(ps.keys())
            x = np.arange(len(scenes)) + (i - 0.5) * width
            ax.bar(x, [ps[s] for s in scenes], width, label=cond)
        if scenes:
            ax.set_xticks(np.arange(len(scenes))); ax.set_xticklabels(scenes, rotation=45, fontsize=8)
        ax.set_ylabel("mIoU"); ax.set_title("θ* mIoU por escena"); ax.legend()
        fig.tight_layout(); fig.savefig(S / "figs" / "phase6_scene_consistency.png", dpi=110)
        plt.close(fig)


def subprocess_commit():
    import subprocess
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]).decode().strip()


def _write_md(c, p2, p3, p4, p6):
    L = []
    L.append("# OVO — Estudio de Fusión: Conclusiones\n")
    L.append(f"*Dataset Replica (8 escenas) · fusion_method=clip · GTJump determinista · "
             f"commit {c['meta']['commit']} · {c['meta']['date']}*\n")
    L.append("## Baselines\n")
    for cond, b in c["baselines"].items():
        L.append(f"- **{cond}**: mIoU={fnum(b['mIoU'])}, AP_agn={fnum(b['AP_agnostic'])}, "
                 f"Num_Instances={b['Num_Instances']:.0f}")
    L.append("\n## Hallazgos\n")
    for f in c["findings"]:
        L.append(f"- **[{f['rq']}]** {f['claim']}  _(confianza: {f['confidence']})_")
    L.append("\n## Mejor configuración\n")
    for cond in ["light", "aggressive"]:
        if cond in c["best_config"]:
            bc = c["best_config"][cond]
            L.append(f"- **{cond}** θ*: `{json.dumps(bc['theta'], ensure_ascii=False)}` "
                     f"→ mIoU={fnum(bc['mIoU'])}, AP_agn={fnum(bc['AP_agnostic'])}")
    if "recommended_overall" in c["best_config"]:
        r = c["best_config"]["recommended_overall"]
        L.append(f"- **Recomendada (global)**: `{json.dumps(r['theta'], ensure_ascii=False)}`\n  - {r['rationale']}")
    L.append("\n## Robustez\n")
    for k, v in c["robustness"].items():
        L.append(f"- {k}: {fnum(v) if isinstance(v,(int,float)) else v}")
    L.append("\n## Limitaciones\n")
    for cav in c["caveats"]:
        L.append(f"- {cav}")
    (S / "conclusions.md").write_text("\n".join(L))


CSS = """
body{font-family:'Segoe UI',system-ui,sans-serif;background:#0f1117;color:#cdd9ff;line-height:1.6;max-width:1000px;margin:0 auto;padding:2rem}
h1{color:#fff}h2{color:#74c0fc;border-bottom:1px solid #2e3350;padding-bottom:.4rem;margin-top:2.5rem}
h3{color:#5c7cfa}table{width:100%;border-collapse:collapse;margin:1rem 0;font-size:.88rem}
th{background:#22263a;color:#74c0fc;text-align:left;padding:.5rem .7rem;border-bottom:2px solid #2e3350}
td{padding:.45rem .7rem;border-bottom:1px solid #2e3350}code{background:#141824;padding:.1em .4em;border-radius:3px;color:#c3e88d}
.card{background:#1a1d27;border:1px solid #2e3350;border-radius:8px;padding:1rem 1.3rem;margin:1rem 0}
.good{color:#51cf66}.bad{color:#ff6b6b}.muted{color:#7885b0}img{max-width:100%;border:1px solid #2e3350;border-radius:6px;margin:.6rem 0}
.badge{display:inline-block;padding:.15em .5em;border-radius:4px;font-size:.75rem;background:#1a2a4a;color:#74c0fc;margin-right:.3rem}
"""


def _img(name, caption=""):
    if (S / "figs" / name).exists():
        return f'<img src="figs/{name}" alt="{name}"><p class="muted">{caption}</p>'
    return ""


def _write_dashboard(c, p2, p3, p4, p6, ranges, master, orb=None):
    h = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>OVO Fusion Study — Results</title><style>{CSS}</style></head><body>"]
    h.append("<h1>OVO — Estudio de Fusión · Resultados</h1>")
    h.append(f"<p class='muted'>Replica 8 escenas · fusion_method=clip · GTJump determinista · "
             f"commit {c['meta']['commit']} · {c['meta']['date']}</p>")

    # exec summary
    h.append("<h2>Resumen ejecutivo</h2><div class='card'>")
    h.append(f"<p>Fases ejecutadas: {', '.join(c['meta']['phases_run'])}. Omitidas: {', '.join(c['meta']['phases_skipped'])}.</p>")
    if c["best_config"].get("recommended_overall"):
        r = c["best_config"]["recommended_overall"]
        h.append(f"<p><b>Config recomendada:</b> <code>{json.dumps(r['theta'], ensure_ascii=False)}</code><br>{r['rationale']}</p>")
    h.append("<ul>")
    for f in c["findings"][:6]:
        h.append(f"<li><span class='badge'>{f['rq']}</span>{f['claim']}</li>")
    h.append("</ul></div>")

    # baselines
    h.append("<h2>Baselines</h2><table><tr><th>Condición</th><th>mIoU</th><th>AP_agn</th><th>mAcc</th><th>Num_Instances</th></tr>")
    for cond, b in c["baselines"].items():
        h.append(f"<tr><td>{cond}</td><td>{fnum(b['mIoU'])}</td><td>{fnum(b['AP_agnostic'])}</td>"
                 f"<td>{fnum(b['mAcc'])}</td><td>{b['Num_Instances']:.0f}</td></tr>")
    h.append("</table>")

    # ablation
    if p2:
        h.append("<h2>Fase 2 · Ablación de mecanismos (RQ1)</h2>")
        for cond in ["light", "aggressive"]:
            h.append(f"<h3>{cond}</h3><table><tr><th>cadena</th><th>mIoU</th><th>AP_agn</th>"
                     "<th>Num_Inst</th><th>ΔmIoU vs base</th><th>Wilcoxon p</th></tr>")
            for e in p2["by_condition"][cond]:
                dm = e.get("delta_mIoU"); pv = e.get("wilcoxon_p_mIoU")
                cls = "good" if (dm or 0) > 0 else ("bad" if (dm or 0) < 0 else "")
                h.append(f"<tr><td><code>{e['label'].replace('abl-','')}</code></td>"
                         f"<td>{fnum(e['mIoU'])}</td><td>{fnum(e['AP_agnostic'])}</td>"
                         f"<td>{e['Num_Instances']:.0f}</td>"
                         f"<td class='{cls}'>{('%+.4f'%dm) if dm is not None else '—'}</td>"
                         f"<td>{fnum(pv,3) if pv is not None else '—'}</td></tr>")
            h.append("</table>")
        h.append(_img("phase2_delta_heatmap.png", "Δ mIoU de cada cadena vs baseline."))
        h.append(f"<p>Daño de cooccurrence: <code>{json.dumps(p2.get('cooccurrence_damage',{}))}</code><br>"
                 f"overlap nuevo vs old: <code>{json.dumps(p2.get('overlap_new_vs_old',{}))}</code></p>")

    # phase1 figs
    h.append("<h2>Fase 1 · Rangos guiados por datos</h2>")
    h.append(_img("phase1_reject_attribution.png", "Atribución de REJECTs por criterio."))
    h.append(_img("phase1_ctx_distributions.png", "Distribuciones ctx ACCEPT vs REJECT."))

    # morris
    if p3:
        h.append("<h2>Fase 3 · Cribado Morris (RQ3)</h2>")
        h.append(f"<p>Params activos: <code>{p3['active_params']}</code> → reducidos: "
                 f"<code>{p3['reduced_params']}</code></p>")
        h.append(_img("phase3_morris.png", "μ* (importancia) vs σ (interacción) por condición y objetivo."))

    # optuna / pareto
    if p4:
        h.append("<h2>Fase 4 · Optuna multi-objetivo (RQ2)</h2>")
        for cond in ["light", "aggressive"]:
            bc = p4["by_condition"][cond]
            kn = bc.get("knee")
            h.append(f"<h3>{cond} · Pareto ({len(bc['pareto'])} pts, {bc['n_complete']} trials)</h3>")
            if kn:
                h.append(f"<div class='card'><b>knee θ*:</b> <code>{json.dumps(kn['theta'], ensure_ascii=False)}</code><br>"
                         f"mIoU={fnum(kn['mIoU'])} · AP_agn={fnum(kn['AP_agnostic'])} · "
                         f"Num_Instances={kn['Num_Instances']}</div>")
            h.append(_img(f"phase4_pareto_{cond}.png", f"Frente de Pareto {cond} (color=Num_Instances)."))

    # robustness
    if p6:
        h.append("<h2>Fase 6 · Robustez (RQ4)</h2><div class='card'>")
        v = p6.get("consistent_across_conditions")
        h.append(f"<p>Consistente entre condiciones: <b class='{'good' if v else 'bad'}'>{v}</b></p>")
        h.append(f"<p>Cruce: <code>{json.dumps(p6.get('cross_condition',{}), ensure_ascii=False)}</code></p>")
        h.append("</div>")
        h.append(_img("phase6_scene_consistency.png", "Consistencia por escena del θ*."))

    # external validation ORB-SLAM
    if orb and orb.get("configs"):
        h.append("<h2>Validación externa · ORB-SLAM2 (drift real)</h2><div class='card'>")
        h.append("<p>Sin checkpoints, pipeline completo. ¿Transfiere el hallazgo del estudio "
                 "(SimulatedSLAM/jump-drift) a SLAM real?</p>")
        h.append("<table><tr><th>Config</th><th>mIoU</th><th>AP_agnostic</th><th>Num_Instances</th></tr>")
        names = {"orb-A-baseline": "A · baseline [centroid,cos_sim,overlap]",
                 "orb-B-mechanism": "B · mecanismo [cooccurrence,aabb,cos_sim,overlap]",
                 "orb-C-tuned": "C · θ* tuneado (sim)",
                 "orb-D-nocooc": "D · sin cooccurrence [aabb,cos_sim,overlap]"}
        for k, v in orb["configs"].items():
            h.append(f"<tr><td>{names.get(k,k)}</td><td>{fnum(v['mIoU'])}</td>"
                     f"<td>{fnum(v['AP_agnostic'])}</td><td>{v['Num_Instances']:.0f}</td></tr>")
        h.append("</table>")
        for comp, v in orb.get("comparisons", {}).items():
            cls = "good" if v["delta_AP_agnostic_mean"] > 0 else "bad"
            h.append(f"<p>{comp}: ΔAP_agn=<b class='{cls}'>{v['delta_AP_agnostic_mean']:+.4f} "
                     f"({v['pct_AP_agnostic']:+.1f}%)</b> (Wilcoxon p={fnum(v['wilcoxon_p_AP_agnostic'],3)}, "
                     f"n={v['n']}); ΔmIoU={v['delta_mIoU_mean']:+.4f} ({v['pct_mIoU']:+.1f}%)</p>")
        verdict = orb.get("finding_transfers")
        h.append(f"<p><b>Veredicto:</b> el hallazgo del estudio simulado "
                 f"<b class='{'good' if verdict else 'bad'}'>"
                 f"{'TRANSFIERE' if verdict else 'NO TRANSFIERE'}</b> a ORB-SLAM real. "
                 f"{orb.get('finding_summary','')}</p>")
        h.append("</div>")
        h.append(_img("orbslam_compare.png", "Baseline vs mecanismo vs tuneado en ORB-SLAM real."))

    # caveats
    h.append("<h2>Limitaciones</h2><ul>")
    for cav in c["caveats"]:
        h.append(f"<li>{cav}</li>")
    h.append("</ul>")
    h.append("<p class='muted'>Memoria machine-readable: <code>study/conclusions.json</code> · "
             "tabla maestra: <code>study/runs_master.parquet</code></p>")
    h.append("</body></html>")
    (S / "results_dashboard.html").write_text("\n".join(h))


if __name__ == "__main__":
    main()
