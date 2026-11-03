"""Instance predictions on the GT mesh: the on-disk format, written by ``run_eval --segment``.

Layout inside an ``instance_pred/`` directory, per scene::

    {scene}.txt                      one line per instance: ``<mask file> <class index> <score>``
    {scene}.classes.json             {"benchmark": ..., "class_names": [...]}
    predicted_masks/{scene}_{i}.json run-length encoded mask over the GT mesh vertices

The class index is a position in ``class_names`` (-1 = unclassified). Storing the list next to
the predictions lets the evaluator check that it is the one its GT expects.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_MASKS_DIR = "predicted_masks"


@dataclass(frozen=True)
class InstancePredictions:
    masks: np.ndarray  # (N, V) bool, one row per instance over the V GT mesh vertices
    classes: np.ndarray  # (N,) int, index into class_names, -1 = unclassified
    scores: np.ndarray  # (N,) float
    class_names: tuple[str, ...]

    def __post_init__(self):
        n = len(self.masks)
        if not (len(self.classes) == len(self.scores) == n):
            raise ValueError(f"{n} masks, {len(self.classes)} classes, {len(self.scores)} scores")


def write(directory: Path, scene: str, preds: InstancePredictions, benchmark: str) -> None:
    directory = Path(directory)
    (directory / _MASKS_DIR).mkdir(parents=True, exist_ok=True)
    width = len(str(max(len(preds.masks) - 1, 0)))
    lines = []
    for i, (mask, cls, score) in enumerate(zip(preds.masks, preds.classes, preds.scores)):
        mask_file = f"./{_MASKS_DIR}/{scene}_{str(i).zfill(width)}.json"
        (directory / mask_file).write_text(json.dumps(rle_encode(mask)))
        lines.append(f"{mask_file} {int(cls)} {float(score):.4f}")
    (directory / f"{scene}.txt").write_text("\n".join(lines))
    meta = {"benchmark": benchmark, "class_names": list(preds.class_names)}
    (directory / f"{scene}.classes.json").write_text(json.dumps(meta, indent=1))


def read(directory: Path, scene: str) -> InstancePredictions:
    directory = Path(directory)
    meta = json.loads((directory / f"{scene}.classes.json").read_text())
    masks, classes, scores = [], [], []
    for line in (directory / f"{scene}.txt").read_text().splitlines():
        mask_file, cls, score = line.split()
        masks.append(rle_decode(json.loads((directory / mask_file).read_text())).astype(bool))
        classes.append(int(cls))
        scores.append(float(score))
    return InstancePredictions(
        masks=np.stack(masks) if masks else np.zeros((0, 0), dtype=bool),
        classes=np.asarray(classes, dtype=np.int64),
        scores=np.asarray(scores, dtype=np.float64),
        class_names=tuple(meta["class_names"]),
    )


def rle_encode(mask: np.ndarray) -> dict:
    """Run-length encode a 1D binary mask as ``{length, counts}`` (1-based start, run length)."""
    mask = np.asarray(mask).astype(np.uint8)
    padded = np.concatenate([[0], mask, [0]])
    runs = np.where(padded[1:] != padded[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return {"length": int(mask.shape[0]), "counts": " ".join(str(x) for x in runs)}


def rle_decode(rle: dict) -> np.ndarray:
    values = np.asarray(rle["counts"].split(), dtype=np.int64)
    starts, lengths = values[0::2] - 1, values[1::2]
    mask = np.zeros(rle["length"], dtype=np.uint8)
    for lo, n in zip(starts, lengths):
        mask[lo : lo + n] = 1
    return mask
