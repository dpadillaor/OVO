"""The decisions of one fusion step, read back from the decisions log (ovo.entities.fusion.log),
and the merges they produced.

    table = decisions.parse(csv_text)
    rows = table.at_frame(1990)        # the pairs judged by the fusion of frame 1990
    groups(merges(rows, "greedy", order))   # {survivor: [survivor, absorbed, ...]}

Each row keeps the CSV cells as read, so they can be written back unchanged, next to the fields
the evaluation needs. The merges are recomputed with the run's own apply policy over the rows, in
log order: the same calls the pipeline made, so the same survivors. The pipeline only logs pairs
the policy let it judge, so a row the policy would have skipped means the log is not this fusion's.
"""

import csv
import io
from dataclasses import dataclass
from typing import Optional

from ovo.entities.fusion.apply import POLICIES


@dataclass(frozen=True)
class Row:
    cells: dict[str, str]  # the CSV row as read
    frame_id: int
    first: int
    second: int
    accepted: bool
    reason: str
    mode: Optional[str]


@dataclass(frozen=True)
class Table:
    columns: tuple[str, ...]
    rows: list[Row]

    def frames(self) -> list[int]:
        """Frames with at least one judged pair, ascending."""
        return sorted({r.frame_id for r in self.rows})

    def at_frame(self, frame_id: int) -> list[Row]:
        return [r for r in self.rows if r.frame_id == frame_id]


def parse(text: str) -> Table:
    reader = csv.DictReader(io.StringIO(text))
    rows = [_row(cells) for cells in reader]
    return Table(tuple(reader.fieldnames or ()), rows)


def _row(cells: dict[str, str]) -> Row:
    if cells["accepted"] not in ("True", "False"):
        raise ValueError(f"accepted must be True or False, got {cells['accepted']!r}")
    return Row(
        cells=dict(cells),
        frame_id=int(cells["frame_id"]),
        first=int(cells["first"]),
        second=int(cells["second"]),
        accepted=cells["accepted"] == "True",
        reason=cells["reason"],
        mode=cells["mode"] or None,
    )


def merges(rows: list[Row], apply: str, order: list[int]) -> list[tuple[int, int]]:
    """(survivor, absorbed) as the fusion applied them. `order` is the instance order the pipeline
    ran with (its survivor under `chained` is the earliest instance). Raises ValueError when the
    rows could not come from that fusion: an instance outside `order`, or a pair the policy would
    not have judged (e.g. under `greedy`, an instance absorbed twice)."""
    known = set(order)
    policy = POLICIES[apply](order)
    for r in rows:
        missing = sorted({r.first, r.second} - known)
        if missing:
            raise ValueError(f"frame {r.frame_id}: pair ({r.first}, {r.second}) has {missing}, not in the "
                             f"pre-fusion map: the decisions do not belong to this checkpoint")
        if not policy.should_judge(r.first, r.second):
            raise ValueError(f"frame {r.frame_id}: pair ({r.first}, {r.second}) would not have been judged "
                             f"by '{apply}': the decisions do not belong to this fusion")
        if r.accepted:
            policy.on_accept(r.first, r.second)
    return policy.merges()


def groups(merges: list[tuple[int, int]]) -> dict[int, list[int]]:
    """{final survivor: [survivor, absorbed...]} for every instance that took part in a merge.
    Merges apply in order, so an instance that absorbed others may be absorbed later."""
    into = {absorbed: survivor for survivor, absorbed in merges}

    def final(i: int) -> int:
        while i in into:
            i = into[i]
        return i

    out: dict[int, list[int]] = {}
    for survivor, absorbed in merges:
        for i in (survivor, absorbed):
            members = out.setdefault(final(i), [final(i)])
            if i not in members:
                members.append(i)
    return out
