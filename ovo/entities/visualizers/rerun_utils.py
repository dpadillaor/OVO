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


def get_instance_colors(obj_ids, cmap):
    """Map instance ids to RGB colors."""
    mapped_ids = obj_ids.copy()
    mapped_ids[mapped_ids > -1] = mapped_ids[mapped_ids > -1] % cmap.shape[0]
    instance_colors = np.take(cmap, np.clip(mapped_ids, 0, cmap.shape[0] - 1), axis=0)
    instance_colors[mapped_ids == -1] = 40
    return instance_colors


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


def subsample_consistent(arrays, max_points):
    """Subsample multiple aligned arrays with the same random indices."""
    if len(arrays) == 0:
        return arrays

    n_points = len(arrays[0])
    if n_points <= max_points:
        return arrays

    idx = np.random.choice(n_points, max_points, replace=False)
    idx.sort()
    return tuple(arr[idx] for arr in arrays)


def trajectory_positions(traj_dict):
    """Build trajectory xyz positions sorted by frame id."""
    return np.array([v[:3, 3] for _, v in sorted(traj_dict.items())], dtype=np.float32)


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


def log_loop_closure_pair(
    path_prefix,
    pts_before,
    ids_before,
    pts_after,
    ids_after,
    traj_before,
    traj_after,
    cmap,
    step,
    traj_before_path=None,
    traj_after_path=None,
):
    """Log before/after loop-closure states on consecutive timeline steps."""
    colors_before = get_instance_colors(ids_before, cmap)
    colors_after = get_instance_colors(ids_after, cmap)

    traj_before_pos = trajectory_positions(traj_before)
    traj_after_pos = trajectory_positions(traj_after)
    if traj_before_path is None:
        traj_before_path = f"{path_prefix}/traj_before"
    if traj_after_path is None:
        traj_after_path = f"{path_prefix}/traj_after"

    rr.set_time("lc_event", sequence=2 * step)
    rr.log(
        f"{path_prefix}/points",
        rr.Points3D(
            pts_before,
            colors=colors_before,
            radii=np.full(len(pts_before), 0.008, dtype=np.float32),
        ),
    )
    if len(traj_before_pos) >= 2:
        rr.log(
            traj_before_path,
            rr.LineStrips3D([traj_before_pos], colors=[[255, 165, 0]], radii=[0.007]),
        )

    rr.set_time("lc_event", sequence=2 * step + 1)
    rr.log(
        f"{path_prefix}/points",
        rr.Points3D(
            pts_after,
            colors=colors_after,
            radii=np.full(len(pts_after), 0.008, dtype=np.float32),
        ),
    )
    if len(traj_after_pos) >= 2:
        rr.log(
            traj_after_path,
            rr.LineStrips3D([traj_after_pos], colors=[[0, 255, 0]], radii=[0.007]),
        )


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
