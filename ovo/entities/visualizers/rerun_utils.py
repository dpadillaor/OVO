import queue
import numpy as np
import rerun as rr
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


QUEUE_EMPTY = object()


def get_instance_cmap(n_colors=40):
    """Return an instance colormap compatible with existing Open3D colors."""
    colours = mcolors.ListedColormap(plt.cm.tab20b.colors + plt.cm.tab20c.colors)
    return (colours(np.arange(n_colors))[:, :3] * 255).astype(np.uint8)


def resolve_instance_ids(obj_ids, n_points):
    """Resolve finest-level instance ids and ensure a 1D int32 array."""
    if obj_ids.ndim > 1 and obj_ids.shape[1] > 0:
        return obj_ids[:, -1].astype(np.int32)
    if obj_ids.ndim == 1:
        return obj_ids.astype(np.int32)
    return np.zeros(n_points, dtype=np.int32)


def ceiling_mask(points, margin=0.2):
    """Return a z-based ceiling mask with an empty-safe fallback."""
    if points.size == 0:
        return np.zeros(0, dtype=bool)
    ceiling_z = points[:, -1].max() - margin
    return points[:, -1] < ceiling_z


def get_nowait_or_empty(mpqueue):
    """Fetch queue item without polling races; return sentinel when empty."""
    try:
        return mpqueue.get_nowait()
    except queue.Empty:
        return QUEUE_EMPTY


def drain_until_sentinel(mpqueue):
    """Drain queue during error handling and stop on sentinel if seen."""
    while True:
        item = get_nowait_or_empty(mpqueue)
        if item is QUEUE_EMPTY:
            return False
        if item is None:
            return True


def log_instances(prefix, points, instance_ids, instance_colors):
    """Log one entity per instance id to keep per-instance toggles in the UI."""
    for ins_id in np.unique(instance_ids[instance_ids >= 0]):
        mask = instance_ids == ins_id
        if mask.sum() == 0:
            continue
        ins_color = instance_colors[mask][0]
        rr.log(
            f"{prefix}/instances/ins_{ins_id}",
            rr.Points3D(
                points[mask],
                colors=np.tile(ins_color, (mask.sum(), 1)),
                radii=np.full(mask.sum(), 0.008, dtype=np.float32),
            ),
            static=True,
        )
