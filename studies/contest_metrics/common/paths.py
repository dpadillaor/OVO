"""Rutas raíz del proyecto. Fuente única (mata la dep cruzada probe<->viz de antes)."""

from __future__ import annotations

import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
OUTPUT_ROOT = REPO_ROOT / "data" / "output" / "Replica"
CHECKPOINT_ROOT = REPO_ROOT / "data" / "checkpoints"
# GT para calificar decisiones contra la verdad (malla + etiquetas por vértice).
MESH_ROOT = REPO_ROOT / "data" / "input" / "Datasets" / "Replica"
GT_ROOT = MESH_ROOT / "instance_gt"
