"""CLI único del módulo contest_metrics. Dominios: query (mecanismo) y viz (telemetría Tier2).

  # qué decide classify entre dos instancias, y por qué (en vivo, sobre pre_fusion.ckpt)
  python -m studies.contest_metrics query pair   --exp <id> --scene office4 27 2
  python -m studies.contest_metrics query list   --exp <id> --scene office4 --by persistence -n 20
  python -m studies.contest_metrics query branch --exp <id> --scene office4 --decision SPLIT
  python -m studies.contest_metrics query point  --exp <id> --scene office4 123456
  # qué DECIDIÓ a lo largo del run (histórico, verdicts.csv)
  python -m studies.contest_metrics query history --exp <id> --scene office4 27 2
  # figuras Tier2 por señal
  python -m studies.contest_metrics viz scene --exp <id> --scene office4 [--derivative]
  python -m studies.contest_metrics viz exp   --exp <id>
  # TUI interactiva
  python -m studies.contest_metrics tui --exp <id> --scene office4
"""

from __future__ import annotations

import argparse
import json
import sys

from .common.resolve import resolve, resolve_exp_path

_INT_KEYS = {"min_grabs", "min_mass"}


# ---- helpers -----------------------------------------------------------
def _parse_overrides(pairs) -> dict:
    """'min_grabs=10' -> {'min_grabs': 10}. Int donde toca, luego float, si no string (p.ej. contest_split_mode=all)."""
    out: dict = {}
    for item in pairs or []:
        key, _, val = item.partition("=")
        key = key.strip()
        if key in _INT_KEYS:
            out[key] = int(val)
        else:
            try:
                out[key] = float(val)
            except ValueError:
                out[key] = val
    return out


def _pair_dict(f) -> dict | None:
    if f is None:
        return None
    return {
        "defender": f.defender, "challenger": f.challenger,
        "containment": f.containment, "reverse_containment": f.reverse_containment,
        "firm_points": f.firm_points, "total_grabs": f.total_grabs,
        "persistence": f.persistence, "focus": f.focus, "exclusivity": f.exclusivity,
        "n_split_points": len(f.split_points),
    }


def _verdict_dict(v) -> dict | None:
    if v is None:
        return None
    return {"decision": v.decision.name, "defender": v.defender,
            "challenger": v.challenger, "reason": v.reason}


def _emit(obj_for_json, text: str, as_json: bool) -> None:
    print(json.dumps(obj_for_json, indent=2, ensure_ascii=False) if as_json else text)


# ---- query -------------------------------------------------------------
def _probe(args):
    from .query.engine import ContestProbe
    return ContestProbe.from_experiment(args.exp, args.scene, _parse_overrides(args.overrides))


def cmd_query_pair(args) -> None:
    from .query.report import format_report
    r = _probe(args).explain(args.pair[0], args.pair[1])
    payload = {
        "a": r.a, "b": r.b, "size_a": r.size_a, "size_b": r.size_b,
        "ab": _pair_dict(r.ab), "ba": _pair_dict(r.ba),
        "verdict_a": _verdict_dict(r.verdict_a), "verdict_b": _verdict_dict(r.verdict_b),
        "resolved": _verdict_dict(r.resolved), "thresholds": r.thresholds,
    }
    _emit(payload, format_report(r), args.json)


def cmd_query_list(args) -> None:
    from .query.report import format_pairs
    pairs = _probe(args).top_pairs(args.n, args.by)
    _emit([_pair_dict(f) for f in pairs], format_pairs(pairs, args.by), args.json)


def cmd_query_branch(args) -> None:
    from .query.report import format_branch
    buckets = _probe(args).branch(args.decision)
    payload = {k: [_verdict_dict(v) for v in vs] for k, vs in buckets.items()}
    _emit(payload, format_branch(buckets), args.json)


def cmd_query_point(args) -> None:
    from dataclasses import asdict

    from .query.report import format_point
    r = _probe(args).point(args.point)
    _emit(asdict(r), format_point(r), args.json)


def cmd_query_history(args) -> None:
    from .query.verdicts import filter_rows, format_history, load_verdicts
    ctx = resolve(args.exp, args.scene)
    rows = load_verdicts(ctx.verdicts_path) if ctx.verdicts_path else []
    pair = tuple(args.pair) if args.pair else None
    rows = filter_rows(rows, pair=pair, decision=args.decision)
    _emit(rows, format_history(rows), args.json)


def cmd_query_resolve(args) -> None:
    """Diagnóstico: qué sustrato se resolvió y si existe."""
    ctx = resolve(args.exp, args.scene)
    payload = {
        "exp_id": ctx.exp_id, "scene": ctx.scene, "kind": ctx.kind, "ok": ctx.ok,
        "ckpt_path": str(ctx.ckpt_path) if ctx.ckpt_path else None,
        "verdicts_path": str(ctx.verdicts_path) if ctx.verdicts_path else None,
        "errors": ctx.errors,
    }
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"exp={ctx.exp_id} scene={ctx.scene} kind={ctx.kind} ok={ctx.ok}")
        print(f"  ckpt:     {ctx.ckpt_path}")
        print(f"  verdicts: {ctx.verdicts_path}")
        for e in ctx.errors:
            print(f"  ⚠ {e}")


# ---- viz ---------------------------------------------------------------
def _figures_dir(exp_path, scene):
    return exp_path / scene / "fusion" / "contest" / "figures"


def _run_viz_scene(exp_path, scene, args) -> None:
    from .viz.figures_mpl import save_scene_figures
    from .viz.loader import load_scene
    data = load_scene(exp_path, scene)
    out_dir = args.out_dir if getattr(args, "out_dir", None) else _figures_dir(exp_path, scene)
    saved = save_scene_figures(data, out_dir, ext=args.ext, derived=not args.no_derived,
                               derivative=args.derivative)
    print(f"[{scene}] {len(saved)} figuras -> {out_dir}")


def cmd_viz_scene(args) -> None:
    _run_viz_scene(resolve_exp_path(args.exp), args.scene, args)


def cmd_viz_exp(args) -> None:
    from .viz.loader import discover_scenes
    exp_path = resolve_exp_path(args.exp)
    scenes = discover_scenes(exp_path)
    if not scenes:
        print(f"sin escenas con logs de contest en {exp_path}", file=sys.stderr)
        sys.exit(1)
    for scene in scenes:
        _run_viz_scene(exp_path, scene, args)


# ---- eval --------------------------------------------------------------
def _resolve_scenes(spec: str) -> list[str]:
    from .eval.pipeline import SCENE_GROUPS
    return SCENE_GROUPS.get(spec, [s.strip() for s in spec.split(",") if s.strip()])


def cmd_eval(args) -> None:
    from .eval.pipeline import grade_scenes
    from .eval.report import format_summary, to_payload
    scenes = [args.scene] if getattr(args, "scene", None) else _resolve_scenes(args.scenes)
    evals = grade_scenes(args.exp, scenes, _parse_overrides(args.overrides))
    title = f"{args.exp} · {args.scenes if not getattr(args, 'scene', None) else args.scene}"
    _emit(to_payload(evals), format_summary(evals, title), args.json)


# ---- tui ---------------------------------------------------------------
def cmd_tui(args) -> None:
    from .tui.app import run_tui
    run_tui(args.exp, args.scene)


def cmd_inspect(args) -> None:
    from .viz.inspect3d import run
    run(args.exp, args.scene, args.z_max, args.point_size, _parse_overrides(args.overrides))


# ---- parser ------------------------------------------------------------
def _add_exp_scene(p) -> None:
    p.add_argument("--exp", required=True, help="ID, ruta o prefijo del experimento")
    p.add_argument("--scene", required=True, help="escena (p.ej. office4)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="contest_metrics", description="Métricas y debug del mecanismo contest")
    dom = ap.add_subparsers(dest="domain", required=True)

    # query
    q = dom.add_parser("query", help="interroga el mecanismo (classify/features/histórico)")
    qsub = q.add_subparsers(dest="qcmd", required=True)

    p = qsub.add_parser("pair", help="qué decide classify entre A y B, y por qué")
    _add_exp_scene(p); p.add_argument("pair", nargs=2, type=int, metavar="A B")
    p.add_argument("--set", dest="overrides", action="append", metavar="K=V", help="override de umbral (repetible)")
    p.add_argument("--json", action="store_true"); p.set_defaults(func=cmd_query_pair)

    p = qsub.add_parser("list", help="pares más disputados")
    _add_exp_scene(p)
    p.add_argument("--by", default="total_grabs",
                   choices=["total_grabs", "containment", "firm_points", "focus", "persistence", "exclusivity"])
    p.add_argument("-n", type=int, default=20)
    p.add_argument("--set", dest="overrides", action="append", metavar="K=V")
    p.add_argument("--json", action="store_true"); p.set_defaults(func=cmd_query_list)

    p = qsub.add_parser("branch", help="veredictos agrupados por rama (en vivo)")
    _add_exp_scene(p)
    p.add_argument("--decision", help="filtra una rama (SPLIT/MERGE_CONTAINMENT/DEFER_TO_FUSION/NO_ACTION)")
    p.add_argument("--set", dest="overrides", action="append", metavar="K=V")
    p.add_argument("--json", action="store_true"); p.set_defaults(func=cmd_query_branch)

    p = qsub.add_parser("point", help="counters crudos de un punto (store, sin mapa)")
    _add_exp_scene(p); p.add_argument("point", type=int, metavar="POINT_ID")
    p.add_argument("--set", dest="overrides", action="append", metavar="K=V")
    p.add_argument("--json", action="store_true"); p.set_defaults(func=cmd_query_point)

    p = qsub.add_parser("history", help="timeline de decisiones (verdicts.csv)")
    _add_exp_scene(p); p.add_argument("pair", nargs="*", type=int, metavar="A B")
    p.add_argument("--decision", help="filtra una rama")
    p.add_argument("--json", action="store_true"); p.set_defaults(func=cmd_query_history)

    p = qsub.add_parser("resolve", help="diagnóstico: qué ckpt se resolvió y si existe")
    _add_exp_scene(p); p.add_argument("--json", action="store_true"); p.set_defaults(func=cmd_query_resolve)

    # viz
    v = dom.add_parser("viz", help="figuras de telemetría Tier2 (per-KF)")
    vsub = v.add_subparsers(dest="vcmd", required=True)
    for name, fn, help_ in (("scene", cmd_viz_scene, "figuras de una escena"),
                            ("exp", cmd_viz_exp, "figuras de todas las escenas")):
        p = vsub.add_parser(name, help=help_)
        p.add_argument("--exp", required=True)
        if name == "scene":
            p.add_argument("--scene", required=True)
            p.add_argument("--out-dir", dest="out_dir")
        p.add_argument("--ext", default="svg")
        p.add_argument("--no-derived", action="store_true")
        p.add_argument("--derivative", action="store_true")
        p.set_defaults(func=fn)

    # eval
    e = dom.add_parser("eval", help="califica las decisiones del contest contra GT (TP/FP/FN/TN por gate)")
    e.add_argument("--exp", required=True, help="ID, ruta o prefijo del experimento")
    grp = e.add_mutually_exclusive_group(required=True)
    grp.add_argument("--scene", help="una escena (p.ej. office4)")
    grp.add_argument("--scenes", help="grupo (tuning/held/all) o lista coma-separada")
    e.add_argument("--set", dest="overrides", action="append", metavar="K=V",
                   help="override de umbral (repetible), p.ej. --set firm_tau=0.30")
    e.add_argument("--json", action="store_true"); e.set_defaults(func=cmd_eval)

    # tui
    t = dom.add_parser("tui", help="TUI interactiva de consulta")
    _add_exp_scene(t); t.set_defaults(func=cmd_tui)

    # inspect (visor 3D interactivo Open3D)
    i = dom.add_parser("inspect", help="visor 3D: escena gris + query por terminal de un par (def→ch)")
    _add_exp_scene(i)
    i.add_argument("--z-max", dest="z_max", type=float, default=None,
                   help="corta el techo: renderiza z <= z_max (Replica office ~1.5)")
    i.add_argument("--point-size", dest="point_size", type=float, default=3.0)
    i.add_argument("--set", dest="overrides", action="append", metavar="K=V",
                   help="override de umbral (repetible)")
    i.set_defaults(func=cmd_inspect)

    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
