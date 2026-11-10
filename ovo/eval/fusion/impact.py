"""What the fusion did to the map: class-agnostic AP and per-GT-instance quality, before vs after.

Both sides come from the same pre-fusion masks: after = the masks of each merged group OR-ed into
one, under the id of its survivor. Same vertices, same projection, so any difference is the
fusion's. Matching and the set of GT instances are those of ovo.eval.instance_ap.

    pre = side(masks, ids, gt_ids, class_ids)
    post = side(*post_masks(masks, ids, groups), gt_ids, class_ids)
    agnostic_ap(gt_ids, pre, post, class_ids)
    {"pre": {"AP": .., "AP_50": .., "AP_25": .., "matched_50": 7, "spurious_50": 3, "missed_50": 2},
     "post": {...}, "delta": {...}}            # delta = post - pre
"""

from dataclasses import dataclass

import numpy as np

from ovo.eval import instance_ap

MATCH_IOU = 0.5  # threshold of the matched/spurious/missed counts and of match_50
TOLERANCE = 1e-6  # IoU changes below this are "unchanged"


@dataclass(frozen=True)
class Side:
    """The predictions on one side of the fusion and how they match the GT."""
    masks: np.ndarray  # (N, V) bool
    ids: np.ndarray  # (N,) instance id of each mask
    matching: instance_ap.Matching  # at MATCH_IOU

    def matched_id(self, gt_id: int):
        index = self.matching.pred_of_gt[gt_id]
        return None if index is None else int(self.ids[index])


def side(masks: np.ndarray, ids: np.ndarray, gt_ids: np.ndarray, class_ids: tuple[int, ...]) -> Side:
    return Side(masks, ids, instance_ap.match_agnostic(_scan(gt_ids, masks), class_ids, MATCH_IOU))


def post_masks(masks: np.ndarray, ids: np.ndarray, groups: dict[int, list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Masks after the merges: each group's masks OR-ed under its survivor's id. ids ascending."""
    survivor_of = {member: survivor for survivor, members in groups.items() for member in members}
    merged: dict[int, np.ndarray] = {}
    for ins_id, mask in zip(ids.tolist(), masks):
        key = survivor_of.get(ins_id, ins_id)
        merged[key] = merged[key] | mask if key in merged else mask.copy()
    out_ids = sorted(merged)
    return np.array([merged[i] for i in out_ids], dtype=bool).reshape(len(out_ids), masks.shape[1]), np.array(out_ids, dtype=np.int64)


def agnostic_ap(gt_ids: np.ndarray, pre: Side, post: Side, class_ids: tuple[int, ...]) -> dict:
    out = {name: _ap(gt_ids, s, class_ids) for name, s in (("pre", pre), ("post", post))}
    out["delta"] = {k: out["post"][k] - out["pre"][k] for k in out["pre"]}
    return out


def _ap(gt_ids: np.ndarray, s: Side, class_ids: tuple[int, ...]) -> dict:
    ap = instance_ap.evaluate_agnostic({"scene": _scan(gt_ids, s.masks)}, class_ids)
    m = s.matching
    return {**ap, "matched_50": m.matched, "spurious_50": m.spurious, "missed_50": m.missed}


def instance_stats(gt_ids: np.ndarray, pre: Side, post: Side, class_ids: tuple[int, ...], class_names: tuple[str, ...]) -> list[dict]:
    """One row per GT instance the AP counts (instance_stats.csv):

    iou_*        best IoU of any prediction with it (0 if none touches it)
    coverage_*   share of it covered by that best-IoU prediction
    change       improved | worsened | unchanged (iou_post vs iou_pre)
    match_50     KEPT | LOST | GAINED | UNMATCHED: claimed at IoU 0.5 before x after
    pred_*       id of the instance that claims it at 0.5 (after: the group's survivor)
    fragments_pre  ';'-joined ids of pre-fusion predictions that claim no GT and overlap this
                 one the most
    """
    gts = list(pre.matching.pred_of_gt)
    column = np.full(gt_ids.shape, -1)
    for g, gt_id in enumerate(gts):
        column[gt_ids == gt_id] = g
    sizes = np.bincount(column[column >= 0], minlength=len(gts))
    inter_pre, inter_post = _intersections(pre.masks, column, len(gts)), _intersections(post.masks, column, len(gts))
    fragments = _fragments(pre, inter_pre)

    name_of = dict(zip(class_ids, class_names))
    rows = []
    for g, gt_id in enumerate(gts):
        iou_pre, cov_pre = _best(inter_pre[:, g], pre.masks.sum(1), sizes[g])
        iou_post, cov_post = _best(inter_post[:, g], post.masks.sum(1), sizes[g])
        pred_pre, pred_post = pre.matched_id(gt_id), post.matched_id(gt_id)
        rows.append({
            "gt_id": gt_id,
            "class_name": name_of[gt_id // 1000],
            "iou_pre": iou_pre,
            "iou_post": iou_post,
            "coverage_pre": cov_pre,
            "coverage_post": cov_post,
            "change": _change(iou_pre, iou_post),
            "match_50": _match_status(pred_pre is not None, pred_post is not None),
            "pred_pre": "" if pred_pre is None else pred_pre,
            "pred_post": "" if pred_post is None else pred_post,
            "fragments_pre": ";".join(str(i) for i in fragments.get(g, [])),
        })
    return rows


def _scan(gt_ids: np.ndarray, masks: np.ndarray) -> instance_ap.Scan:
    n = len(masks)
    return instance_ap.Scan(gt_ids=gt_ids, masks=masks, classes=np.zeros(n, dtype=np.int64), scores=np.ones(n))


def _intersections(masks: np.ndarray, column: np.ndarray, n_gt: int) -> np.ndarray:
    """(N, G) vertices shared by each mask and each counted GT instance."""
    out = np.zeros((len(masks), n_gt), dtype=np.int64)
    for i, mask in enumerate(masks):
        under = column[mask]
        out[i] = np.bincount(under[under >= 0], minlength=n_gt)
    return out


def _best(inter: np.ndarray, pred_sizes: np.ndarray, gt_size: int) -> tuple[float, float]:
    """(IoU, coverage) of the prediction with the highest IoU; (0, 0) when none touches it."""
    if not inter.any():
        return 0.0, 0.0
    iou = inter / (pred_sizes + gt_size - inter)
    best = int(np.argmax(iou))
    return float(iou[best]), float(inter[best] / gt_size)


def _fragments(pre: Side, inter: np.ndarray) -> dict[int, list[int]]:
    """{GT column: ids of the unclaiming predictions whose largest intersection is that GT}."""
    claiming = {i for i in pre.matching.pred_of_gt.values() if i is not None}
    out: dict[int, list[int]] = {}
    for i in range(len(pre.masks)):
        if i in claiming or not inter[i].any():
            continue
        out.setdefault(int(np.argmax(inter[i])), []).append(int(pre.ids[i]))
    return out


def _change(pre: float, post: float) -> str:
    if post > pre + TOLERANCE:
        return "improved"
    if post < pre - TOLERANCE:
        return "worsened"
    return "unchanged"


def _match_status(pre: bool, post: bool) -> str:
    if pre and post:
        return "KEPT"
    if pre:
        return "LOST"
    return "GAINED" if post else "UNMATCHED"
