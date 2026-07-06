#!/usr/bin/env python3
"""Lector/analizador de contest.json (store crudo del mecanismo contest).

El JSON es el volcado del ContestStore: por cada punto en disputa, cuantos KFs
lo vieron bajo cada instancia ganadora.

    estructura:  {"store": {point_id: {winner_ins_id: nº_KFs}}}

Sin dependencias (solo stdlib). Subcomandos:

    summary   JSON                 totales + distribucion ganadores/punto + top ganadores
    winner    JSON  ID [ID...]     puntos que reclama cada ganador + stats de conteo
    point     JSON  PID [PID...]   ganadores de cada punto concreto
    hot       JSON  [--by winners|count] [--top N]   puntos mas disputados / mas vistos
    pairmass  JSON  A W            masa cruda y nº puntos del par (loser A visto bajo W)

Ejemplos:
    python Study_seg/contest_json.py summary  contest.json
    python Study_seg/contest_json.py winner   contest.json 63 151
    python Study_seg/contest_json.py hot      contest.json --by winners --top 15
    python Study_seg/contest_json.py pairmass contest.json 151 63
"""
import argparse
import json
import statistics as st
from collections import Counter, defaultdict


def load(path):
    d = json.load(open(path))
    store = d.get("grabs", d)
    # normaliza a {int: {int: int}}
    return {int(p): {int(w): int(c) for w, c in ws.items()} for p, ws in store.items()}


def _winner_index(store):
    """winner -> lista de (point, count)."""
    idx = defaultdict(list)
    for p, ws in store.items():
        for w, c in ws.items():
            idx[w].append((p, c))
    return idx


# --------------------------------------------------------------------------- #
def cmd_summary(args):
    store = load(args.json)
    npts = len(store)
    per_point = [len(ws) for ws in store.values()]
    all_counts = [c for ws in store.values() for c in ws.values()]
    idx = _winner_index(store)

    print(f"puntos en disputa: {npts}")
    print(f"ganadores distintos: {len(idx)}")
    print(f"observaciones totales (suma KFs): {sum(all_counts)}\n")

    print("GANADORES POR PUNTO (cuantos puntos tienen N reclamantes):")
    for n, cnt in sorted(Counter(per_point).items()):
        marca = "  <- disputa real" if n >= 2 else ""
        print(f"  {n} ganador(es): {cnt:>8} puntos{marca}")

    print(f"\nCONTEO KF POR (punto,ganador): mediana={st.median(all_counts)} "
          f"min={min(all_counts)} max={max(all_counts)}")

    print("\nTOP GANADORES (por nº de puntos reclamados):")
    top = sorted(idx.items(), key=lambda kv: -len(kv[1]))[: args.top]
    for w, pts in top:
        counts = [c for _, c in pts]
        print(f"  ins {w:>5}: {len(pts):>7} puntos  | masa={sum(counts):>9}  "
              f"KF medio={st.mean(counts):.1f}")


def cmd_winner(args):
    store = load(args.json)
    idx = _winner_index(store)
    for w in args.ids:
        pts = idx.get(w, [])
        print(f"--- ganador {w} ---")
        if not pts:
            print("  (no reclama ningun punto)")
            continue
        counts = [c for _, c in pts]
        print(f"  puntos reclamados: {len(pts)}")
        print(f"  masa (suma KFs):   {sum(counts)}")
        print(f"  KF/punto: mediana={st.median(counts)} min={min(counts)} max={max(counts)}")


def cmd_point(args):
    store = load(args.json)
    for pid in args.ids:
        ws = store.get(pid)
        print(f"--- punto {pid} ---")
        if not ws:
            print("  (no esta en disputa)")
            continue
        for w, c in sorted(ws.items(), key=lambda kv: -kv[1]):
            print(f"  visto bajo ins {w:>5} en {c} KFs")


def cmd_hot(args):
    store = load(args.json)
    if args.by == "winners":
        ranked = sorted(store.items(), key=lambda kv: -len(kv[1]))
        head = "nº ganadores"
        val = lambda ws: len(ws)
    else:  # count
        ranked = sorted(store.items(), key=lambda kv: -sum(kv[1].values()))
        head = "obs totales"
        val = lambda ws: sum(ws.values())
    print(f"PUNTOS MAS CALIENTES (por {head}):")
    for pid, ws in ranked[: args.top]:
        winners = " ".join(f"{w}:{c}" for w, c in sorted(ws.items(), key=lambda kv: -kv[1]))
        print(f"  punto {pid:>8}  {head}={val(ws):>4}  | {winners}")


def cmd_seen(args):
    """Proxy rapido de 'veces vista' una instancia: max KF-count entre sus puntos.

    Si un punto se vio bajo la instancia en N KFs, su mascara estuvo activa >= N
    KFs. El max sobre sus puntos es una cota inferior de los KFs en que aparecio.
    (Dato exacto = nº de KFs distintos -> solo en el checkpoint del mapa.)
    """
    store = load(args.json)
    idx = _winner_index(store)
    print(f"{'ins':>6} {'puntos':>8} {'KF_max(proxy)':>14} {'KF_mediana':>11}")
    for w in args.ids:
        pts = idx.get(w, [])
        if not pts:
            print(f"{w:>6} {'0':>8}   (no reclama puntos en el store)")
            continue
        counts = [c for _, c in pts]
        print(f"{w:>6} {len(pts):>8} {max(counts):>14} {st.median(counts):>11.1f}")


def cmd_pairmass(args):
    """Masa cruda del par (A visto bajo W): suma de KFs y nº de puntos."""
    store = load(args.json)
    a, w = args.a, args.w
    pts = [(p, ws[w]) for p, ws in store.items() if a in ws and w in ws and a != w]
    # nota: 'a in ws' no marca propiedad (eso lo da el mapa vivo); aqui solo
    # contamos puntos donde A y W coexisten como reclamantes en el store.
    only_w = [(p, ws[w]) for p, ws in store.items() if w in ws]
    print(f"par {a} (loser) <- {w} (winner) [datos crudos del store]")
    print(f"  puntos donde W={w} es reclamante: {len(only_w)}  masa={sum(c for _,c in only_w)}")
    print(f"  puntos donde A={a} y W={w} coexisten: {len(pts)}  masa={sum(c for _,c in pts)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("summary"); s.add_argument("json"); s.add_argument("--top", type=int, default=15); s.set_defaults(fn=cmd_summary)
    w = sub.add_parser("winner"); w.add_argument("json"); w.add_argument("ids", type=int, nargs="+"); w.set_defaults(fn=cmd_winner)
    pt = sub.add_parser("point"); pt.add_argument("json"); pt.add_argument("ids", type=int, nargs="+"); pt.set_defaults(fn=cmd_point)
    h = sub.add_parser("hot"); h.add_argument("json"); h.add_argument("--by", choices=["winners", "count"], default="winners"); h.add_argument("--top", type=int, default=15); h.set_defaults(fn=cmd_hot)
    sn = sub.add_parser("seen"); sn.add_argument("json"); sn.add_argument("ids", type=int, nargs="+"); sn.set_defaults(fn=cmd_seen)
    pm = sub.add_parser("pairmass"); pm.add_argument("json"); pm.add_argument("a", type=int); pm.add_argument("w", type=int); pm.set_defaults(fn=cmd_pairmass)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
