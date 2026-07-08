"""TUI de consulta del contest. Carga el sustrato UNA vez y consulta en caliente.

Sintaxis de la barra (una línea):
  27 2            -> par: qué decide classify entre 27 y 2
  /SPLIT          -> rama: veredictos en vivo de esa decisión (o /all)
  #123456         -> punto: counters crudos del store
  top persistence -> pares más disputados por ese criterio
  hist 27 2       -> histórico (verdicts.csv) del par
  set k=v         -> reajusta un umbral y recarga el discriminator
  /clear          -> limpia pantalla (igual que ctrl+l)
La TUI es un front-end fino: toda la lógica vive en query/engine.py (idéntica a producción).
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Footer, Header, Input, RichLog, Static

from ..common.resolve import resolve
from ..query.engine import ContestProbe
from ..query.loader import load_substrate
from ..query.report import format_branch, format_pairs, format_point, format_report
from ..query.verdicts import filter_rows, format_history, load_verdicts

_HELP = ("consultas:  [b]A B[/b] par · [b]/SPLIT[/b] rama · [b]#punto[/b] · "
         "[b]top persistence[/b] · [b]hist A B[/b] · [b]set k=v[/b] · [b]/clear[/b] limpiar")


class ContestTUI(App):
    CSS = """
    #status { height: auto; padding: 0 1; color: $text-muted; }
    #out { height: 1fr; border: round $primary; padding: 0 1; }
    Input { dock: bottom; }
    """
    BINDINGS = [("ctrl+c", "quit", "salir"), ("ctrl+l", "clear", "limpiar")]

    def __init__(self, exp: str, scene: str) -> None:
        super().__init__()
        self._exp, self._scene = exp, scene
        self._probe: ContestProbe | None = None
        self._overrides: dict = {}
        self._verdicts: list[dict] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield Static(_HELP, id="status")
            yield RichLog(id="out", highlight=True, markup=True, wrap=False)
        yield Input(placeholder="cargando sustrato…", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        self.title = "contest query"
        self.sub_title = f"{self._exp} · {self._scene}"
        self._load()

    # ---- carga (worker: torch.load ~1s) --------------------------------
    def _load(self) -> None:
        self.run_worker(self._load_worker, thread=True, exclusive=True)

    def _load_worker(self) -> None:
        log = self.query_one("#out", RichLog)
        ctx = resolve(self._exp, self._scene)
        if not ctx.ok:
            self.call_from_thread(log.write, "[red]no analizable:[/red]\n  " + "\n  ".join(ctx.errors))
            return
        self.call_from_thread(log.write, f"[dim]cargando {ctx.kind} ckpt {ctx.ckpt_path}…[/dim]")
        self._verdicts = load_verdicts(ctx.verdicts_path) if ctx.verdicts_path else []
        self._sub = load_substrate(ctx)   # cacheado: `set` reusa sin recargar el ckpt
        self._probe = ContestProbe.from_substrate(self._sub, self._overrides)
        self._ctx = ctx
        self.call_from_thread(self._show_ready, ctx)

    def _show_ready(self, ctx) -> None:
        inp = self.query_one(Input)
        inp.disabled = False
        inp.placeholder = "consulta (ej: 27 2)"
        inp.focus()
        self.query_one("#out", RichLog).write(
            f"[green]listo[/green] · {ctx.kind} · {len(self._verdicts)} filas histórico · {_HELP}")

    # ---- acciones ------------------------------------------------------
    def action_clear(self) -> None:
        self.query_one("#out", RichLog).clear()

    def on_input_submitted(self, ev: Input.Submitted) -> None:
        q = ev.value.strip()
        self.query_one(Input).value = ""
        if q:
            self._dispatch(q)

    def _dispatch(self, q: str) -> None:
        log = self.query_one("#out", RichLog)
        if q.strip().lower() == "/clear":
            log.clear()
            return
        if self._probe is None:
            log.write("[yellow]aún cargando…[/yellow]")
            return
        log.write(f"[b cyan]› {q}[/b cyan]")
        try:
            log.write(self._run(q))
        except Exception as e:  # noqa: BLE001 — TUI no debe crashear por una query
            log.write(f"[red]error:[/red] {e}")

    def _run(self, q: str) -> str:
        p = self._probe
        if q.startswith("#"):
            return format_point(p.point(int(q[1:])))
        if q.startswith("/"):
            dec = q[1:].strip()
            return format_branch(p.branch(None if dec.lower() in ("", "all") else dec))
        toks = q.split()
        if toks[0] == "top":
            by = toks[1] if len(toks) > 1 else "total_grabs"
            return format_pairs(p.top_pairs(20, by), by)
        if toks[0] == "hist":
            pair = tuple(int(x) for x in toks[1:3]) if len(toks) >= 3 else None
            return format_history(filter_rows(self._verdicts, pair=pair))
        if toks[0] == "set":
            self._apply_set(toks[1:])
            return f"[green]umbrales:[/green] {self._overrides} — recargado"
        if len(toks) == 2:
            return format_report(p.explain(int(toks[0]), int(toks[1])))
        return "[yellow]no entiendo. " + _HELP + "[/yellow]"

    def _apply_set(self, kvs) -> None:
        int_keys = {"min_grabs", "min_mass"}
        for kv in kvs:
            k, _, v = kv.partition("=")
            k = k.strip()
            self._overrides[k] = int(v) if k in int_keys else float(v)
        # recargar solo el discriminator (barato): reusar el sustrato ya en RAM
        self._probe = ContestProbe.from_substrate(self._sub, self._overrides)


def run_tui(exp: str, scene: str) -> None:
    ContestTUI(exp, scene).run()
