import rerun as rr
import rerun.blueprint as rrb
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import time
import multiprocessing
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


def stream_rerun(semantic_module, mpqueue, query_data, cam_intrinsic, scene_name, output_path, show, save_rrd=False):
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
            if not mpqueue.empty():
                data = mpqueue.get()
                if data is None:
                    break

                points, obj_ids, colors, c2w = data

                points = points.astype(np.float32)
                c2w = c2w.astype(np.float32)

                # ── Resolve finest instance IDs ───────────────────────────
                if obj_ids.ndim > 1 and obj_ids.shape[1] > 0:
                    instance_ids = obj_ids[:, -1].astype(np.int32)
                elif obj_ids.ndim == 1:
                    instance_ids = obj_ids.astype(np.int32)
                else:
                    instance_ids = np.zeros(points.shape[0], dtype=np.int32)

                # ── Ceiling filter (same as Open3D visualizer) ────────────
                ceiling_z = points[:, -1].max() - 0.2
                mask = points[:, -1] < ceiling_z
                points = points[mask]
                instance_ids = instance_ids[mask]

                # ── Compute instance colors ───────────────────────────────
                instance_colors = _get_instance_colors(instance_ids, cmap)

                # ── AnnotationContext (labels in the UI) ──────────────────
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

                # ── Camera pose (temporal — keeps trajectory history) ─────
                rr.log(
                    "world/camera",
                    rr.Transform3D(
                        translation=c2w[:3, 3],
                        mat3x3=c2w[:3, :3],
                    ),
                )

                # ── Camera trajectory (line through all past positions) ───
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

                # ── Instance-colored point cloud (static — overwrites) ────
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
            else:
                time.sleep(0.01)

        except Exception as e:
            print(f"[RerunVis] Warning: {e}")
            while not mpqueue.empty():
                d = mpqueue.get()
                if d is None:
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
            if not mpqueue.empty():
                data = mpqueue.get()
                if data is None:
                    break

                if not isinstance(data, dict):
                    continue

                # ── Loop Closure message ──────────────────────────────────
                if data.get("type") == "loop_closure":
                    pcd_before = data["pcd_before"].astype(np.float32)
                    pcd_after  = data["pcd_after"].astype(np.float32)
                    ids        = data["ids"].astype(np.int32)
                    traj_before = data["traj_before"]
                    traj_after  = data["traj_after"]

                    ceiling_z = pcd_after[:, -1].max() - 0.2
                    pts_b = pcd_before[pcd_before[:, -1] < ceiling_z]
                    ids_b = ids[pcd_before[:, -1] < ceiling_z]
                    pts_a = pcd_after[pcd_after[:, -1] < ceiling_z]
                    ids_a = ids[pcd_after[:, -1] < ceiling_z]

                    if len(pts_b) > MAX_FUSION_POINTS:
                        idx = np.random.choice(len(pts_b), MAX_FUSION_POINTS, replace=False)
                        idx.sort(); pts_b = pts_b[idx]; ids_b = ids_b[idx]
                    if len(pts_a) > MAX_FUSION_POINTS:
                        idx = np.random.choice(len(pts_a), MAX_FUSION_POINTS, replace=False)
                        idx.sort(); pts_a = pts_a[idx]; ids_a = ids_a[idx]

                    traj_b_pos = np.array([v[:3, 3] for _, v in sorted(traj_before.items())], dtype=np.float32)
                    traj_a_pos = np.array([v[:3, 3] for _, v in sorted(traj_after.items())],  dtype=np.float32)

                    # step 0 — drifted cloud + drifted trajectory only
                    rr.set_time("lc_event", sequence=0)
                    rr.log("lc/points", rr.Points3D(pts_b, colors=_get_instance_colors(ids_b, cmap), radii=np.full(len(pts_b), 0.008, dtype=np.float32)))
                    if len(traj_b_pos) >= 2:
                        rr.log("lc/traj_before", rr.LineStrips3D([traj_b_pos], colors=[[255, 165, 0]], radii=[0.007]))

                    # step 1 — corrected cloud + both trajectories (traj_before persists from step 0)
                    rr.set_time("lc_event", sequence=1)
                    rr.log("lc/points", rr.Points3D(pts_a, colors=_get_instance_colors(ids_a, cmap), radii=np.full(len(pts_a), 0.008, dtype=np.float32)))
                    if len(traj_a_pos) >= 2:
                        rr.log("lc/traj_after", rr.LineStrips3D([traj_a_pos], colors=[[0, 255, 0]], radii=[0.007]))

                    continue

                # ── Fusion message ────────────────────────────────────────
                if data.get("type") != "fusion":
                    continue

                points = data["points"].astype(np.float32)
                before_ids = data["before_ids"]
                after_ids = data["after_ids"]
                frame_id = data["frame_id"]

                # Resolve finest level if multi-column
                if before_ids.ndim > 1 and before_ids.shape[1] > 0:
                    before_ids = before_ids[:, -1].astype(np.int32)
                else:
                    before_ids = before_ids.astype(np.int32)

                if after_ids.ndim > 1 and after_ids.shape[1] > 0:
                    after_ids = after_ids[:, -1].astype(np.int32)
                else:
                    after_ids = after_ids.astype(np.int32)

                # ── Ceiling filter ────────────────────────────────────────
                ceiling_z = points[:, -1].max() - 0.2
                mask = points[:, -1] < ceiling_z
                pts = points[mask]
                b_ids = before_ids[mask]
                a_ids = after_ids[mask]

                # ── Check if anything actually changed ────────────────────
                if np.array_equal(b_ids, a_ids):
                    continue  # no fusion happened, skip

                # ── Subsample if too large ────────────────────────────────
                if len(pts) > MAX_FUSION_POINTS:
                    idx = np.random.choice(len(pts), MAX_FUSION_POINTS, replace=False)
                    idx.sort()
                    pts, b_ids, a_ids = pts[idx], b_ids[idx], a_ids[idx]

                before_colors = _get_instance_colors(b_ids, cmap)
                after_colors = _get_instance_colors(a_ids, cmap)

                unique_before = np.unique(b_ids[b_ids >= 0])
                unique_after = np.unique(a_ids[a_ids >= 0])
                n_before = len(unique_before)
                n_after = len(unique_after)
                fused = n_before - n_after

                # ── Diff Analysis ─────────────────────────────────────────
                changed_mask = b_ids != a_ids
                disappeared_ids = set(unique_before) - set(unique_after)

                # ── Build fusion map (which instances merged into which) ───
                fusion_map = {}  # {deleted_id: survivor_id}
                fusion_details = []
                for after_id in unique_after:
                    mask_after = a_ids == after_id
                    before_ids_in_region = np.unique(b_ids[mask_after])
                    before_ids_in_region = before_ids_in_region[before_ids_in_region >= 0]

                    if len(before_ids_in_region) > 1:
                        contributors = []
                        for bid in before_ids_in_region:
                            if bid != after_id:
                                fusion_map[bid] = after_id
                            contributors.append(int(bid))
                        fusion_details.append((int(after_id), contributors))

                # ── Merge arrows (centroids) ───────────────────────────────
                arrow_origins = []
                arrow_vectors = []

                if len(fusion_map) > 0:
                    for deleted_id, survivor_id in fusion_map.items():
                        mask_deleted = b_ids == deleted_id
                        mask_survivor = a_ids == survivor_id

                        if mask_deleted.sum() > 0 and mask_survivor.sum() > 0:
                            centroid_deleted = pts[mask_deleted].mean(axis=0)
                            centroid_survivor = pts[mask_survivor].mean(axis=0)
                            arrow_origins.append(centroid_deleted)
                            arrow_vectors.append(centroid_survivor - centroid_deleted)

                rr.set_time("fusion_event", sequence=step)

                # ── Before Fusion - Individual Instances ──────────────────
                for ins_id in unique_before:
                    mask = b_ids == ins_id
                    if mask.sum() > 0:
                        ins_color = before_colors[mask][0]
                        rr.log(
                            f"before/instances/ins_{ins_id}",
                            rr.Points3D(
                                pts[mask],
                                colors=np.tile(ins_color, (mask.sum(), 1)),
                                radii=np.full(mask.sum(), 0.008, dtype=np.float32),
                            ),
                            static=True,
                        )

                rr.log("before/info",
                       rr.TextLog(f"Frame {frame_id} | {n_before} instances"),
                       static=True)

                # ── After Fusion - Individual Instances ───────────────────
                for ins_id in unique_after:
                    mask = a_ids == ins_id
                    if mask.sum() > 0:
                        ins_color = after_colors[mask][0]
                        rr.log(
                            f"after/instances/ins_{ins_id}",
                            rr.Points3D(
                                pts[mask],
                                colors=np.tile(ins_color, (mask.sum(), 1)),
                                radii=np.full(mask.sum(), 0.008, dtype=np.float32),
                            ),
                            static=True,
                        )

                rr.log("after/info",
                       rr.TextLog(f"Frame {frame_id} | {n_after} instances ({fused} fused)"),
                       static=True)

                # ── Diff Visualization Layers ─────────────────────────────
                # Layer 1: Changed points (yellow)
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

                # Layer 2: Disappeared instances (red)
                for dis_id in disappeared_ids:
                    mask = b_ids == dis_id
                    if mask.sum() > 0:
                        rr.log(
                            f"diff/disappeared/ins_{dis_id}",
                            rr.Points3D(
                                pts[mask],
                                colors=[255, 0, 0],
                                radii=np.full(mask.sum(), 0.01, dtype=np.float32),
                            ),
                            static=True,
                        )

                # Layer 3: Merge arrows
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

                # Layer 4: Bounding boxes for disappeared instances
                for dis_id in disappeared_ids:
                    mask = b_ids == dis_id
                    if mask.sum() > 0:
                        pts_ins = pts[mask]
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

                # ── Stats Panel (Markdown) ────────────────────────────────
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
                        stats_text += f"\n- Instance **{survivor_id}** ← merged from [{contributors_str}]"
                else:
                    stats_text += "\n*No multi-instance fusions (only deletions)*"

                stats_text += f"\n\n## Disappeared Instances\n"
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
            else:
                time.sleep(0.05)

        except Exception as e:
            print(f"[FusionVis] Warning: {e}")
            while not mpqueue.empty():
                d = mpqueue.get()
                if d is None:
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
            if not mpqueue.empty():
                data = mpqueue.get()
                if data is None:
                    break

                if not isinstance(data, dict) or data.get("type") != "loop_closure":
                    continue

                pcd_before = data["pcd_before"].astype(np.float32)
                pcd_after = data["pcd_after"].astype(np.float32)
                ids = data["ids"].astype(np.int32)
                traj_before = data["traj_before"]
                traj_after = data["traj_after"]

                # ── Ceiling filter ────────────────────────────────────────
                ceiling_z = pcd_after[:, -1].max() - 0.2
                mask_before = pcd_before[:, -1] < ceiling_z
                mask_after = pcd_after[:, -1] < ceiling_z

                pts_before = pcd_before[mask_before]
                pts_after = pcd_after[mask_after]
                ids_before = ids[mask_before]
                ids_after = ids[mask_after]

                # ── Subsample if too large ────────────────────────────────
                if len(pts_before) > MAX_POINTS:
                    idx = np.random.choice(len(pts_before), MAX_POINTS, replace=False)
                    idx.sort()
                    pts_before = pts_before[idx]
                    ids_before = ids_before[idx]

                if len(pts_after) > MAX_POINTS:
                    idx = np.random.choice(len(pts_after), MAX_POINTS, replace=False)
                    idx.sort()
                    pts_after = pts_after[idx]
                    ids_after = ids_after[idx]

                colors_before = _get_instance_colors(ids_before, cmap)
                colors_after = _get_instance_colors(ids_after, cmap)

                # ── Trajectory positions (sorted by frame id) ─────────────
                traj_before_pos = np.array(
                    [v[:3, 3] for _, v in sorted(traj_before.items())], dtype=np.float32
                )
                traj_after_pos = np.array(
                    [v[:3, 3] for _, v in sorted(traj_after.items())], dtype=np.float32
                )

                # ── BEFORE: timestep 2*step ───────────────────────────────
                rr.set_time("lc_event", sequence=2 * step)

                rr.log(
                    "world/points",
                    rr.Points3D(
                        pts_before,
                        colors=colors_before,
                        radii=np.full(len(pts_before), 0.008, dtype=np.float32),
                    ),
                )

                if len(traj_before_pos) >= 2:
                    rr.log(
                        "world/trajectory",
                        rr.LineStrips3D(
                            [traj_before_pos],
                            colors=[[255, 165, 0]],  # orange
                            radii=[0.007],
                        ),
                    )

                # ── AFTER: timestep 2*step + 1 ────────────────────────────
                rr.set_time("lc_event", sequence=2 * step + 1)

                rr.log(
                    "world/points",
                    rr.Points3D(
                        pts_after,
                        colors=colors_after,
                        radii=np.full(len(pts_after), 0.008, dtype=np.float32),
                    ),
                )

                if len(traj_after_pos) >= 2:
                    rr.log(
                        "world/trajectory",
                        rr.LineStrips3D(
                            [traj_after_pos],
                            colors=[[0, 255, 0]],  # green
                            radii=[0.007],
                        ),
                    )

                step += 1
            else:
                time.sleep(0.05)

        except Exception as e:
            print(f"[LCVis] Warning: {e}")
            while not mpqueue.empty():
                d = mpqueue.get()
                if d is None:
                    return
            time.sleep(1.0)
