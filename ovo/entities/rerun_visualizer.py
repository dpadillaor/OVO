import rerun as rr
import rerun.blueprint as rrb
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import time
import queue
from pathlib import Path


def _get_instance_cmap(n_colors=40):
    """
    Returns a colormap for instance visualization, matching the Open3D visualizer.
    Uses the same tab20b + tab20c combination as vis_utils.get_cmap().
    """
    colours = mcolors.ListedColormap(plt.cm.tab20b.colors + plt.cm.tab20c.colors)
    return (colours(np.arange(n_colors))[:, :3] * 255).astype(np.uint8)


def _get_instance_colors(obj_ids, cmap):
    """
    Maps object instance IDs to RGB colors using the colormap.
    Points with id == -1 get dark grey so the structure is visible.
    """
    mapped_ids = obj_ids.copy()
    mapped_ids[mapped_ids > -1] = mapped_ids[mapped_ids > -1] % cmap.shape[0]
    instance_colors = np.take(cmap, np.clip(mapped_ids, 0, cmap.shape[0] - 1), axis=0)
    instance_colors[mapped_ids == -1] = 40  # dark grey instead of black
    return instance_colors


_QUEUE_EMPTY = object()


def _resolve_instance_ids(obj_ids, n_points):
    """Resolve finest-level instance ids and ensure a 1D int32 array."""
    if obj_ids.ndim > 1 and obj_ids.shape[1] > 0:
        return obj_ids[:, -1].astype(np.int32)
    if obj_ids.ndim == 1:
        return obj_ids.astype(np.int32)
    return np.zeros(n_points, dtype=np.int32)


def _ceiling_mask(points, margin=0.2):
    """Return a z-based ceiling mask with an empty-safe fallback."""
    if points.size == 0:
        return np.zeros(0, dtype=bool)
    ceiling_z = points[:, -1].max() - margin
    return points[:, -1] < ceiling_z


def _subsample_consistent(arrays, max_points):
    """Subsample multiple aligned arrays with the same random indices."""
    if len(arrays) == 0:
        return arrays

    n_points = len(arrays[0])
    if n_points <= max_points:
        return arrays

    idx = np.random.choice(n_points, max_points, replace=False)
    idx.sort()
    return tuple(arr[idx] for arr in arrays)


def _trajectory_positions(traj_dict):
    """Build trajectory xyz positions sorted by frame id."""
    return np.array([v[:3, 3] for _, v in sorted(traj_dict.items())], dtype=np.float32)


def _get_nowait_or_empty(mpqueue):
    """Fetch queue item without polling races; return sentinel when empty."""
    try:
        return mpqueue.get_nowait()
    except queue.Empty:
        return _QUEUE_EMPTY


def _drain_until_sentinel(mpqueue):
    """Drain queue during error handling and stop on sentinel if seen."""
    while True:
        item = _get_nowait_or_empty(mpqueue)
        if item is _QUEUE_EMPTY:
            return False
        if item is None:
            return True


def _log_loop_closure_pair(
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
    colors_before = _get_instance_colors(ids_before, cmap)
    colors_after = _get_instance_colors(ids_after, cmap)

    traj_before_pos = _trajectory_positions(traj_before)
    traj_after_pos = _trajectory_positions(traj_after)
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


def _log_instances(prefix, points, instance_ids, instance_colors):
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


def stream_rerun(semantic_module, mpqueu    e, query_data, cam_intrinsic, scene_name, output_path, show, save_rrd=False):
    """
    Streams SLAM data to Rerun visualizer — instance-colored point cloud only.

    Each point is colored by its instance ID using the same colormap as Open3D.
    Unassigned points (id == -1) are shown in dark grey.
    The camera trajectory is drawn as a temporal trace.

    Point clouds are logged as static (overwritten each step) to keep memory low.
    """
    rr.init(f"OVO_{scene_name}", spawn=show)
    if save_rrd:
        rr.save(str(Path(output_path) / "rerun.rrd"))

    # ── Blueprint: single 3D view ────────────────────────────────────────
    blueprint = rrb.Blueprint(
        rrb.Spatial3DView(name="Instances", contents="world/**"),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint)

    # ── Camera intrinsics (static, logged once) ──────────────────────────
    width = cam_intrinsic["width"]
    height = cam_intrinsic["height"]
    K = cam_intrinsic["intrinsic"]

    rr.log(
        "world/camera",
        rr.Pinhole(
            resolution=[width, height],
            image_from_camera=K,
        ),
        static=True,
    )

    # ── Instance colormap ─────────────────────────────────────────────────
    cmap = _get_instance_cmap()

    step = 0
    trajectory = []  # accumulates camera positions for trajectory line

    while True:
        try:
            data = _get_nowait_or_empty(mpqueue)
            if data is _QUEUE_EMPTY:
                time.sleep(0.01)
                continue

            if data is None:
                break

            points, obj_ids, colors, c2w = data

            points = points.astype(np.float32)
            c2w = c2w.astype(np.float32)

            instance_ids = _resolve_instance_ids(obj_ids, points.shape[0])

            mask = _ceiling_mask(points)
            points = points[mask]
            instance_ids = instance_ids[mask]

            instance_colors = _get_instance_colors(instance_ids, cmap)

            unique_ids = np.unique(instance_ids[instance_ids >= 0])
            if len(unique_ids) > 0:
                annotations = [
                    rr.AnnotationInfo(
                        id=int(uid),
                        label=f"obj_{uid}",
                        color=tuple(int(x) for x in cmap[int(uid) % len(cmap)]),
                    )
                    for uid in unique_ids
                ]
                rr.log("world/points", rr.AnnotationContext(annotations), static=True)

            rr.set_time("step", sequence=step)

            rr.log(
                "world/camera",
                rr.Transform3D(
                    translation=c2w[:3, 3],
                    mat3x3=c2w[:3, :3],
                ),
            )

            trajectory.append(c2w[:3, 3].tolist())
            if len(trajectory) >= 2:
                rr.log(
                    "world/trajectory",
                    rr.LineStrips3D(
                        [trajectory],
                        colors=[[0, 255, 255]],  # cyan
                        radii=[0.005],
                    ),
                    static=True,
                )

            class_ids = instance_ids.copy().astype(np.uint16)
            class_ids[instance_ids < 0] = 0

            rr.log(
                "world/points",
                rr.Points3D(
                    points,
                    colors=instance_colors,
                    class_ids=class_ids,
                    radii=np.full(len(points), 0.008, dtype=np.float32),
                ),
                static=True,
            )

            step += 1

        except Exception as e:
            print(f"[RerunVis] Warning: {e}")
            if _drain_until_sentinel(mpqueue):
                return


def stream_rerun_fusion(semantic_module, mpqueue, query_data, cam_intrinsic, scene_name, output_path, show, save_rrd=False):
    """
    Rerun visualizer for fusion comparison — shows side-by-side Before / After
    point clouds every time instance fusion occurs in update_map().

    Only receives data when fusion actually happens (not every segment step).
    Each message is a dict with: points, before_ids, after_ids, frame_id.

    Includes diff visualization layers:
    - diff/changed_points: Points whose instance ID changed (yellow)
    - diff/disappeared/ins_X: Instances that were merged away (red points)
    - diff/merge_arrows: 3D arrows showing fusion direction (magenta, origin → survivor)
    - diff/boxes/ins_X: Bounding boxes around merged instances (red)

    Also handles loop_closure messages, showing before/after geometric correction
    in a dedicated view with lc_event timeline.
    """
    rr.init(f"OVO_{scene_name}_fusion", spawn=show)
    if save_rrd:
        rr.save(str(Path(output_path) / "rerun.rrd"))

    # ── Blueprint: fusion side-by-side + LC section + stats ───────────────
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    name="Before Fusion",
                    contents=["before/instances/**", "diff/**"],
                    origin="/",
                ),
                rrb.Spatial3DView(
                    name="After Fusion",
                    contents=["after/instances/**", "diff/**"],
                    origin="/",
                ),
            ),
            rrb.Spatial3DView(
                name="Loop Closure",
                contents=["lc/**"],
                origin="/",
            ),
            rrb.TextDocumentView(
                name="Fusion Stats",
                contents="stats/**",
            ),
        ),
        collapse_panels=False,
        auto_layout=True,
    )
    rr.send_blueprint(blueprint)

    cmap = _get_instance_cmap()
    step = 0
    MAX_FUSION_POINTS = 400_000

    while True:
        try:
            data = _get_nowait_or_empty(mpqueue)
            if data is _QUEUE_EMPTY:
                time.sleep(0.05)
                continue

            if data is None:
                break

            if not isinstance(data, dict):
                continue

            # ── Loop Closure message ──────────────────────────────────
            if data.get("type") == "loop_closure":
                pcd_before = data["pcd_before"].astype(np.float32)
                pcd_after = data["pcd_after"].astype(np.float32)
                ids = data["ids"].astype(np.int32)
                traj_before = data["traj_before"]
                traj_after = data["traj_after"]

                if pcd_after.size == 0:
                    continue

                ceiling_z = pcd_after[:, -1].max() - 0.2
                mask_before = pcd_before[:, -1] < ceiling_z if pcd_before.size > 0 else np.zeros(0, dtype=bool)
                mask_after = pcd_after[:, -1] < ceiling_z

                pts_b = pcd_before[mask_before]
                ids_b = ids[mask_before]
                pts_a = pcd_after[mask_after]
                ids_a = ids[mask_after]

                pts_b, ids_b = _subsample_consistent((pts_b, ids_b), MAX_FUSION_POINTS)
                pts_a, ids_a = _subsample_consistent((pts_a, ids_a), MAX_FUSION_POINTS)

                _log_loop_closure_pair(
                    path_prefix="lc",
                    pts_before=pts_b,
                    ids_before=ids_b,
                    pts_after=pts_a,
                    ids_after=ids_a,
                    traj_before=traj_before,
                    traj_after=traj_after,
                    cmap=cmap,
                    step=0,
                )
                continue

            # ── Fusion message ────────────────────────────────────────
            if data.get("type") != "fusion":
                continue

            points = data["points"].astype(np.float32)
            before_ids = _resolve_instance_ids(data["before_ids"], points.shape[0])
            after_ids = _resolve_instance_ids(data["after_ids"], points.shape[0])
            frame_id = data["frame_id"]

            mask = _ceiling_mask(points)
            pts = points[mask]
            b_ids = before_ids[mask]
            a_ids = after_ids[mask]

            if np.array_equal(b_ids, a_ids):
                continue

            pts, b_ids, a_ids = _subsample_consistent((pts, b_ids, a_ids), MAX_FUSION_POINTS)

            before_colors = _get_instance_colors(b_ids, cmap)
            after_colors = _get_instance_colors(a_ids, cmap)

            unique_before = np.unique(b_ids[b_ids >= 0])
            unique_after = np.unique(a_ids[a_ids >= 0])
            n_before = len(unique_before)
            n_after = len(unique_after)
            fused = n_before - n_after

            changed_mask = b_ids != a_ids
            disappeared_ids = set(unique_before) - set(unique_after)

            fusion_map = {}
            fusion_details = []
            for after_id in unique_after:
                mask_after = a_ids == after_id
                before_ids_in_region = np.unique(b_ids[mask_after])
                before_ids_in_region = before_ids_in_region[before_ids_in_region >= 0]

                if len(before_ids_in_region) <= 1:
                    continue

                contributors = []
                for bid in before_ids_in_region:
                    if bid != after_id:
                        fusion_map[bid] = after_id
                    contributors.append(int(bid))
                fusion_details.append((int(after_id), contributors))

            arrow_origins = []
            arrow_vectors = []
            for deleted_id, survivor_id in fusion_map.items():
                mask_deleted = b_ids == deleted_id
                mask_survivor = a_ids == survivor_id

                if mask_deleted.sum() == 0 or mask_survivor.sum() == 0:
                    continue

                centroid_deleted = pts[mask_deleted].mean(axis=0)
                centroid_survivor = pts[mask_survivor].mean(axis=0)
                arrow_origins.append(centroid_deleted)
                arrow_vectors.append(centroid_survivor - centroid_deleted)

            rr.set_time("fusion_event", sequence=step)

            _log_instances("before", pts, b_ids, before_colors)
            rr.log("before/info", rr.TextLog(f"Frame {frame_id} | {n_before} instances"), static=True)

            _log_instances("after", pts, a_ids, after_colors)
            rr.log("after/info", rr.TextLog(f"Frame {frame_id} | {n_after} instances ({fused} fused)"), static=True)

            if changed_mask.sum() > 0:
                rr.log(
                    "diff/changed_points",
                    rr.Points3D(
                        pts[changed_mask],
                        colors=[255, 255, 0],
                        radii=np.full(changed_mask.sum(), 0.012, dtype=np.float32),
                    ),
                    static=True,
                )

            for dis_id in disappeared_ids:
                mask_dis = b_ids == dis_id
                if mask_dis.sum() == 0:
                    continue

                rr.log(
                    f"diff/disappeared/ins_{dis_id}",
                    rr.Points3D(
                        pts[mask_dis],
                        colors=[255, 0, 0],
                        radii=np.full(mask_dis.sum(), 0.01, dtype=np.float32),
                    ),
                    static=True,
                )

                pts_ins = pts[mask_dis]
                bbox_min = pts_ins.min(axis=0)
                bbox_max = pts_ins.max(axis=0)
                center = (bbox_min + bbox_max) / 2
                half_sizes = (bbox_max - bbox_min) / 2
                rr.log(
                    f"diff/boxes/ins_{dis_id}",
                    rr.Boxes3D(
                        centers=[center],
                        half_sizes=[half_sizes],
                        labels=[f"Fused: {dis_id}"],
                        colors=[[255, 0, 0]],
                    ),
                    static=True,
                )

            if len(arrow_origins) > 0:
                rr.log(
                    "diff/merge_arrows",
                    rr.Arrows3D(
                        origins=np.array(arrow_origins),
                        vectors=np.array(arrow_vectors),
                        colors=[255, 0, 255],
                        radii=[0.02],
                    ),
                    static=True,
                )

            stats_text = f"""# Frame {frame_id} Fusion Event
                            ## Summary
                            - **Before:** {n_before} instances
                            - **After:** {n_after} instances
                            - **Fused:** {fused} instances

                            ## Fusion Details
                            """
            if len(fusion_details) > 0:
                for survivor_id, contributors in fusion_details:
                    contributors_str = ", ".join(str(c) for c in contributors)
                    stats_text += f"\n- Instance **{survivor_id}** <- merged from [{contributors_str}]"
            else:
                stats_text += "\n*No multi-instance fusions (only deletions)*"

            stats_text += "\n\n## Disappeared Instances\n"
            if len(disappeared_ids) > 0:
                stats_text += ", ".join(str(d) for d in sorted(disappeared_ids))
            else:
                stats_text += "*None*"

            rr.log(
                "stats/fusion_info",
                rr.TextDocument(stats_text, media_type=rr.MediaType.MARKDOWN),
                static=True,
            )

            step += 1

        except Exception as e:
            print(f"[FusionVis] Warning: {e}")
            if _drain_until_sentinel(mpqueue):
                return
            time.sleep(1.0)


def stream_rerun_loopclosure(semantic_module, mpqueue, query_data, cam_intrinsic, scene_name, output_path, show, save_rrd=False):
    """
    Rerun visualizer for loop closure comparison.

    Waits for a single {"type": "loop_closure"} message and then logs two timesteps:
      - lc_event=0: point cloud + trajectory BEFORE global correction (orange trajectory)
      - lc_event=1: point cloud + trajectory AFTER global correction (green trajectory)

    Scrubbing between the two steps in Rerun shows the geometric shift clearly.
    Both point clouds are colored by instance ID using the same colormap as other visualizers.
    """
    rr.init(f"OVO_{scene_name}_loopclosure", spawn=show)
    if save_rrd:
        rr.save(str(Path(output_path) / "rerun.rrd"))

    blueprint = rrb.Blueprint(
        rrb.Spatial3DView(name="Loop Closure", contents="world/**"),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint)

    cmap = _get_instance_cmap()
    MAX_POINTS = 400_000
    step = 0

    while True:
        try:
            data = _get_nowait_or_empty(mpqueue)
            if data is _QUEUE_EMPTY:
                time.sleep(0.05)
                continue

            if data is None:
                break

            if not isinstance(data, dict) or data.get("type") != "loop_closure":
                continue

            pcd_before = data["pcd_before"].astype(np.float32)
            pcd_after = data["pcd_after"].astype(np.float32)
            ids = data["ids"].astype(np.int32)
            traj_before = data["traj_before"]
            traj_after = data["traj_after"]

            if pcd_after.size == 0:
                continue

            ceiling_z = pcd_after[:, -1].max() - 0.2
            mask_before = pcd_before[:, -1] < ceiling_z if pcd_before.size > 0 else np.zeros(0, dtype=bool)
            mask_after = pcd_after[:, -1] < ceiling_z

            pts_before = pcd_before[mask_before]
            pts_after = pcd_after[mask_after]
            ids_before = ids[mask_before]
            ids_after = ids[mask_after]

            pts_before, ids_before = _subsample_consistent((pts_before, ids_before), MAX_POINTS)
            pts_after, ids_after = _subsample_consistent((pts_after, ids_after), MAX_POINTS)

            _log_loop_closure_pair(
                path_prefix="world",
                pts_before=pts_before,
                ids_before=ids_before,
                pts_after=pts_after,
                ids_after=ids_after,
                traj_before=traj_before,
                traj_after=traj_after,
                cmap=cmap,
                step=step,
                traj_before_path="world/trajectory",
                traj_after_path="world/trajectory",
            )

            step += 1

        except Exception as e:
            print(f"[LCVis] Warning: {e}")
            if _drain_until_sentinel(mpqueue):
                return
            time.sleep(1.0)
