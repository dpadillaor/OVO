"""Presentación pura: estructura y formatea las trazas de consulta. No calcula heurística."""

from __future__ import annotations

from dataclasses import dataclass

from ovo.entities.contest.types import PairFeatures, Verdict


@dataclass
class ProbeReport:
    """Resultado de interrogar el par {a, b}: features de ambos sentidos + veredictos + resolución."""

    a: int
    b: int
    size_a: int
    size_b: int
    ab: PairFeatures | None       # a como defender, b como challenger
    ba: PairFeatures | None       # b como defender, a como challenger
    verdict_a: Verdict | None     # decisión real del discriminator con defender=a
    verdict_b: Verdict | None     # decisión real del discriminator con defender=b
    resolved: Verdict | None      # veredicto del par tras juntar ambos sentidos (o None si nadie lo dirige)
    thresholds: dict


@dataclass
class PointReport:
    """Counters crudos de un punto (store), sin mapa."""

    point: int
    claims: int
    sightings: int
    grabbers: dict[int, int]                 # grabber -> nº KFs
    persistence_per_grabber: dict[int, float]
    p4: int                                  # sightings - claims (lealtad invisible)
    loyalty: int                             # claims - Σ grabs


def _fmt_features(f: PairFeatures | None) -> str:
    if f is None:
        return "    (sin disputa registrada en este sentido)"
    return (
        f"    containment={f.containment:.3f}  reverse={f.reverse_containment:.3f}\n"
        f"    firm_points={f.firm_points}  total_grabs={f.total_grabs}\n"
        f"    persistence={f.persistence:.3f}  focus={f.focus:.3f}  "
        f"exclusivity={f.exclusivity:.3f}  |costura|={len(f.split_points)} pts"
    )


def _fmt_verdict(v: Verdict | None, defender: int, challenger: int) -> str:
    if v is None:
        return f"    defender {defender}: sin pares -> sin veredicto"
    drives = "" if v.challenger == challenger else f"  ⚠ challenger real = {v.challenger} (no {challenger})"
    return f"    {v.decision.name}{drives}\n    reason: {v.reason}"


def format_report(r: ProbeReport) -> str:
    """Traza legible del par: features por sentido, veredicto dirigido de cada lado, y el resuelto."""
    th = "  ".join(f"{k}={v}" for k, v in r.thresholds.items())
    resolved = (
        f"{r.resolved.decision.name} (challenger={r.resolved.challenger}) — {r.resolved.reason}"
        if r.resolved is not None
        else "ningún sentido dirige el par -> cada defender resuelve por su cuenta (ver arriba)"
    )
    return (
        f"═══ contest probe · {r.a}  vs  {r.b} ═══\n"
        f"tamaños: |{r.a}|={r.size_a} pts   |{r.b}|={r.size_b} pts\n"
        f"\n▸ {r.a} (defender)  ←  {r.b} (challenger)\n"
        f"{_fmt_features(r.ab)}\n"
        f"{_fmt_verdict(r.verdict_a, r.a, r.b)}\n"
        f"\n▸ {r.b} (defender)  ←  {r.a} (challenger)\n"
        f"{_fmt_features(r.ba)}\n"
        f"{_fmt_verdict(r.verdict_b, r.b, r.a)}\n"
        f"\n▸ resuelto (lo que aplicaría el actuador):\n    {resolved}\n"
        f"\numbrales: {th}"
    )


def format_pairs(pairs: list[PairFeatures], by: str) -> str:
    """Tabla de los pares más disputados, para descubrir qué interrogar."""
    head = (f"top {len(pairs)} pares por {by}:\n"
            f"  {'def':>5} {'chall':>5} {'contain':>8} {'firm':>6} {'grabs':>7} "
            f"{'persist':>8} {'focus':>7} {'excl':>6}")
    rows = [
        f"  {f.defender:>5} {f.challenger:>5} {f.containment:>8.3f} {f.firm_points:>6} "
        f"{f.total_grabs:>7} {f.persistence:>8.3f} {f.focus:>7.3f} {f.exclusivity:>6.3f}"
        for f in pairs
    ]
    return "\n".join([head, *rows])


def format_branch(buckets: dict[str, list[Verdict]]) -> str:
    """Veredictos agrupados por rama de decisión (en vivo)."""
    out: list[str] = []
    for decision in sorted(buckets):
        verdicts = buckets[decision]
        out.append(f"═══ {decision} · {len(verdicts)} defenders ═══")
        for v in verdicts:
            ch = f"←{v.challenger}" if v.challenger is not None else ""
            out.append(f"  def {v.defender:>5} {ch:<8}  {v.reason}")
        out.append("")
    return "\n".join(out).rstrip() or "(sin veredictos)"


def format_point(r: PointReport) -> str:
    """Counters crudos de un punto: claims/sightings/grabs, P4 y persistencia por grabber."""
    lines = [
        f"═══ punto {r.point} ═══",
        f"  sightings={r.sightings}  claims={r.claims}  lealtad(claims-Σgrabs)={r.loyalty}",
        f"  P4 (sightings-claims, lealtad invisible)={r.p4}",
        f"  grabbers ({len(r.grabbers)}):",
    ]
    for ch in sorted(r.grabbers, key=lambda k: r.grabbers[k], reverse=True):
        lines.append(f"    ins {ch:>5}: grabs={r.grabbers[ch]:>4}  persist={r.persistence_per_grabber[ch]:.3f}")
    if not r.grabbers:
        lines.append("    (ningún grabber: punto no disputado)")
    return "\n".join(lines)
