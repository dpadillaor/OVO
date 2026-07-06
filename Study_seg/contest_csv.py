#!/usr/bin/env python3
"""Lector/analizador de contest_verdicts.csv (mecanismo contest de OVO).

Sin dependencias (solo stdlib). Subcomandos:

    summary  CSV                 conteos por decision + familias de razon + stats por feature
    pair     CSV  ID [ID...]      todas las filas que tocan esas instancias (loser o winner)
    filter   CSV  --decision X    filas de una decision, ordenadas por una feature
    compare  CSV_A CSV_B          diff de decision por (loser,winner) entre dos runs

Ejemplos:
    python Study_seg/contest_csv.py summary data/output/.../office3/contest_verdicts.csv
    python Study_seg/contest_csv.py pair  CSV 63 151
    python Study_seg/contest_csv.py filter CSV --decision SPLIT --sort containment --top 10
    python Study_seg/contest_csv.py compare baseline.csv nuevo.csv
"""
import argparse
import csv
import statistics as st
from collections import Counter, defaultdict

NUM = ["containment", "reverse_containment", "firm_points", "total_grabs", "persistence", "focus"]


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def num(row, key):
    v = row.get(key, "")
    if v in (None, ""):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def reason_family(reason):
    """Primer trozo de la razon, antes de '(' o '->'."""
    return reason.split("->")[0].split("(")[0].strip()


# --------------------------------------------------------------------------- #
def cmd_summary(args):
    rows = load(args.csv)
    print(f"filas: {len(rows)}   reports: {sorted({int(r['report']) for r in rows})}\n")

    print("DECISIONES:")
    for d, n in Counter(r["decision"] for r in rows).most_common():
        print(f"  {n:4d}  {d}")

    print("\nFAMILIAS DE RAZON:")
    for fam, n in Counter(reason_family(r["reason"]) for r in rows).most_common():
        print(f"  {n:4d}  {fam}")

    print("\nFEATURES POR DECISION (mediana / min / max):")
    for dec in sorted({r["decision"] for r in rows}):
        sub = [r for r in rows if r["decision"] == dec]
        print(f"  {dec}  (n={len(sub)})")
        for k in NUM:
            vals = [num(r, k) for r in sub if num(r, k) is not None]
            if not vals:
                continue
            print(f"      {k:20} {st.median(vals):10.3f} {min(vals):10.3f} {max(vals):10.3f}")


def cmd_pair(args):
    rows = load(args.csv)
    targets = set(args.ids)
    for tgt in args.ids:
        print(f"--- instancia {tgt} ---")
        hits = [
            r for r in rows
            if int(r["defender"]) == tgt
            or (r["challenger"] not in ("", None) and int(r["challenger"]) == tgt)
        ]
        if not hits:
            print("  (sin filas)")
        for r in hits:
            role = "LOSER " if int(r["defender"]) == tgt else f"win<-{r['defender']}"
            feats = "  ".join(
                f"{k.split('_')[0][:4]}={num(r,k):.3f}" if num(r, k) is not None else f"{k[:4]}=-"
                for k in ("containment", "reverse_containment", "persistence", "focus")
            )
            print(f"  {role:10} {r['decision']:18} {feats}  mass={r['total_grabs']:>8} | {r['reason']}")


def cmd_filter(args):
    rows = [r for r in load(args.csv) if r["decision"] == args.decision]
    rows.sort(key=lambda r: (num(r, args.sort) or 0), reverse=not args.asc)
    rows = rows[: args.top] if args.top else rows
    print(f"{args.decision}: {len(rows)} filas (orden {args.sort}{' asc' if args.asc else ' desc'})")
    for r in rows:
        feats = "  ".join(
            f"{k.split('_')[0][:4]}={num(r,k):.3f}" if num(r, k) is not None else f"{k[:4]}=-"
            for k in NUM
        )
        print(f"  {r['defender']:>4}->{str(r['challenger']):>4}  {feats} | {r['reason']}")


def cmd_compare(args):
    a = {(int(r["defender"]), r["challenger"]): r for r in load(args.csv_a)}
    b = {(int(r["defender"]), r["challenger"]): r for r in load(args.csv_b)}
    keys = sorted(set(a) | set(b))
    ca, cb = Counter(r["decision"] for r in a.values()), Counter(r["decision"] for r in b.values())
    print(f"A={args.csv_a}\nB={args.csv_b}\n")
    print(f"{'decision':20} {'A':>6} {'B':>6} {'delta':>7}")
    for d in sorted(set(ca) | set(cb)):
        print(f"{d:20} {ca[d]:>6} {cb[d]:>6} {cb[d]-ca[d]:>+7}")

    changed = [(k, a[k]["decision"], b[k]["decision"]) for k in keys
               if k in a and k in b and a[k]["decision"] != b[k]["decision"]]
    print(f"\nVEREDICTOS CAMBIADOS ({len(changed)}):")
    for (loser, win), da, db in changed:
        print(f"  {loser:>4}->{str(win):>4}   {da:18} -> {db}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("summary"); s.add_argument("csv"); s.set_defaults(fn=cmd_summary)
    pr = sub.add_parser("pair"); pr.add_argument("csv"); pr.add_argument("ids", type=int, nargs="+"); pr.set_defaults(fn=cmd_pair)
    fl = sub.add_parser("filter"); fl.add_argument("csv")
    fl.add_argument("--decision", required=True)
    fl.add_argument("--sort", default="containment", choices=NUM)
    fl.add_argument("--top", type=int, default=0)
    fl.add_argument("--asc", action="store_true")
    fl.set_defaults(fn=cmd_filter)
    cp = sub.add_parser("compare"); cp.add_argument("csv_a"); cp.add_argument("csv_b"); cp.set_defaults(fn=cmd_compare)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
