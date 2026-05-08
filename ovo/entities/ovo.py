from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from collections import deque
import numpy as np
import pprint
import torch
import time

from ..utils import geometry_utils, instance_utils
from .generator_pipeline import GeneratorPipeline
from .instance3d import Instance3D
from .logger import Logger
from .fusion import create_fusion_strategy, InstanceGeometry
from .fusion_encoders import FusionEncoderAdapter, FusionFrameInput, PEFusionAdapter, SAM3FusionAdapter
from .keyframe_store import KeyframeStore
from .run_config import RunConfig
from .semantic_config import SemanticConfig


@dataclass
class MapData:
    """Typed container for SLAM map state passed into the semantic pipeline."""
    points_3d: torch.Tensor
    points_ids: torch.Tensor
    points_ins_ids: torch.Tensor

    @classmethod
    def from_tuple(cls, t: Tuple) -> "MapData":
        return cls(points_3d=t[0], points_ids=t[1], points_ins_ids=t[2])


@dataclass
class FusionResult:
    objects: Dict[int, Any]
    fused_objects: Dict[int, int]
    points_ins_ids: torch.Tensor
    decisions: list


def _profil(func):
    """Profile function runtime when logging enabled. Keyed by function name."""
    def wrapper(self, *args, **kwargs):
        if self.semantic_config.log:
            torch.cuda.synchronize()
            t0 = time.time()
            out = func(self, *args, **kwargs)
            torch.cuda.synchronize()
            self._time_cache[func.__name__] = time.time() - t0
            return out
        return func(self, *args, **kwargs)
    return wrapper


class OVO:
    """Semantic pipeline: instance lifecycle (segment → track → fuse → classify).

    Args:
        - semantic_config: Typed configuration for the semantic pipeline.
        - run_config: Runtime configuration (device, scene_name, eval flag, etc.).
        - logger: Logger instance.
        - cam_intrinsics: Camera intrinsic matrix. Required when run_config.eval=False.
    """
    def __init__(self, semantic_config: SemanticConfig, run_config: RunConfig, logger: Logger, cam_intrinsics: torch.Tensor | None = None) -> None:
        if not run_config.eval:
            assert cam_intrinsics is not None, "Camera intrinsics required for reconstruction!"

        self.semantic_config = semantic_config
        self.run_config = run_config
        self.cam_intrinsics = cam_intrinsics
        self.logger = logger
        self.device = run_config.device
        Instance3D.n_top_kf = semantic_config.n_top_views

        self.generators = GeneratorPipeline(semantic_config, run_config.device, scene_name=run_config.scene_name, eval=run_config.eval)

        self.kf_store = KeyframeStore()
        self.keyframes_queue = deque([])
        self.objects: Dict[int, Instance3D] = dict()
        self._time_cache: Dict[str, float] = {}
        self._last_visual_snapshot = None

        self.next_ins_id = 0
        self.kf_id = 0

        self.fusion_strategy = create_fusion_strategy({
            "fusion_method": semantic_config.fusion_method,
            "th_centroid": semantic_config.th_centroid,
            "th_cossim": semantic_config.th_cossim,
            "th_points": semantic_config.th_points,
        })
        self.fusion_encoder = self._get_fusion_encoder()
        self._validate_fusion_config()

        if semantic_config.verbose:
            print('Semantic config')
            pprint.PrettyPrinter().pprint(semantic_config)


    def _get_fusion_encoder(self) -> FusionEncoderAdapter | None:
        """Get the fusion encoder adapter based on config, or None if CLIP-only."""
        fusion_method = self.semantic_config.fusion_method.lower()

        if fusion_method == "pe":
            return PEFusionAdapter(self.generators.pe)
        elif fusion_method == "sam3":
            return SAM3FusionAdapter(self.generators.sam3)

        # CLIP fusion uses CLIP features directly, no extra encoder needed
        return None

    def _validate_fusion_config(self):
        """Validate that fusion_method has required generator available."""
        method = self.semantic_config.fusion_method.lower()

        validation_map = {
            "pe": (self.generators.pe, "PE generator"),
            "sam3": (self.generators.sam3, "SAM3 generator"),
        }

        if method in validation_map:
            generator, name = validation_map[method]
            if generator is None:
                raise ValueError(
                    f"fusion_method='{method}' requires {name} to be configured. "
                    f"Add '{method}' key to config or change fusion_method."
                )

    def to(self, device: str) -> None:
        """
        Move predictor model to either 'cpu' or 'cuda' device.
        Args:
            device (str): device to mode the model to.
        """
        if "cuda" in device:
            return self.cuda()
        else:
            return self.cpu()

    def cpu(self) -> None:
        """
        Move predictor model to cpu device.
        """
        self.device = "cpu"
        self.generators.cpu()

    def cuda(self) -> None:
        """
        Move predictor model to cuda default device.
        """
        self.device = "cuda"
        self.generators.cuda()

    def detect_and_track_objects(self, frame_data: Tuple[int, np.ndarray, np.ndarray, Tuple[float, float, int]], map_data: "MapData", c2w: torch.Tensor) -> torch.Tensor:
        r""" For the current frame (1) computes using SAM for each level i \in M, a set of segmentation maps; (2) track segmentation maps between frames projecting 3D points and associating the map to 3D instances, if 3D points don't have an associated 3D instance, create a new; (3) associate 3D points without an instance id to matched instances; (4) fuse 2D segments associated to the same 3D instance.

        Args:
            - frame_data (tuple): current frame data.
                - frame_id (int): current frame id.
                - image (np.ndarray): RGB image with shape (H, W, 3).
                - depth (np.ndarray): Frame depth with shape (h, w).
                - rgb_depth_ratio (tuple): If H == h and W == w, tuple is empty, otherwise stores (r_h, r_w, crop_edge), such that H = (h+2*crop_edge)*r_h, and W = (w+2*crop_edge)*r_w
            - map_data (MapData): typed SLAM map state.
            - c2w: (torch.Tensor): camera to world 3D transform.
        Update:
            - map_data.points_ins_ids (torch.Tensor): updated ids of 3D instances associated to each 3d point after current keyframe segmentation.
        """
        frame_id, image = frame_data[:2]
        self._last_visual_snapshot = None

        seg_maps, binary_maps = self._get_masks(image, frame_id)
        if len(seg_maps) == 0:
            print(f"No mask segmented in {frame_id}!")
            return None

        last_id = self.next_ins_id
        matched_ins_ids, binary_maps, n_matched_points, updated_points_ins_ids = self._match_and_track_instances(frame_data[1:], map_data, c2w, seg_maps, binary_maps)

        # Keep lightweight visual data aligned with the current segmented frame.
        ins_map = np.full(seg_maps.shape, -1, dtype=np.int32)
        for idx, ins_id in enumerate(matched_ins_ids):
            mask_np = binary_maps[idx].detach().cpu().numpy().astype(bool, copy=False)
            ins_map[mask_np] = int(ins_id)

        self._last_visual_snapshot = {
            "frame_id": int(frame_id),
            "rgb": np.asarray(image).copy(),
            "ins_map": ins_map,
            "sam_map": seg_maps.cpu().numpy().astype(np.int16),
        }

        # Save keyframe information
        self.keyframes_queue.append([matched_ins_ids, binary_maps, image, self.kf_id])
        self.kf_id += 1

        if self.semantic_config.log:
            self.kf_store.register_frame(frame_id)
            self.logger.log_ovo_stats(
                {
                    "frame_id": frame_id,
                    "n_obj": [self.next_ins_id - last_id],
                    "n_matches": n_matched_points,
                    "t_sam": round(self._time_cache.get("_get_masks", 0), 2),
                    "t_obj": round(self._time_cache.get("_match_and_track_instances", 0), 3),
                },
                print_output=True
                )
            self._time_cache = {}

        return updated_points_ins_ids

    def get_last_visual_snapshot(self) -> Dict[str, Any] | None:
        """Return latest segmented frame visual data for external stream consumers."""
        return self._last_visual_snapshot

    @_profil
    def _get_masks(self, image: np.ndarray, frame_id: int):
        """ Profiled call to mask_generator to either compute segmentation maps for image, or load precomputed segments.
        Args:
            - frame_id (int): current frame id.
            - image (np.ndarray): RGB image with shape (H, W, 3).

        Returns:
            - seg_map (torch.Tensor): The segmentation maps on self.device with shape (H, W).
            - binary_maps (torch.Tensor): The binary maps on self.device with shape (N, H, W).
        """
        return self.generators.mask.get_masks(image, frame_id)

    @_profil
    def _match_and_track_instances(self, frame_data: Tuple[np.ndarray, np.ndarray, Tuple[float, float, int]], map_data: "MapData", c2w: torch.Tensor, seg_map: torch.Tensor, binary_maps: torch.Tensor) -> Tuple[List[int], torch.Tensor, int]:
        """ For the current frame (1) computes using SAM for each level i \\in M, a set of segmentation maps; (2) track segmentation maps between frames projecting 3D points and associating the map to 3D instances, if 3D points don't have an associated 3D instance, create a new; (3) associate 3D points without an instance id to matched instances; (4) fuse 2D segments associated to the same 3D instance.

        Args:
            - frame_data (tuple): current frame data.
                - image (np.ndarray): RGB image with shape (H, W, 3).
                - depth (np.ndarray): Frame depth with shape (h, w).
                - rgb_depth_ratio (tuple): If H == h and W == w, tuple is empty, otherwise stores (r_h, r_w, crop_edge), such that H = (h+2*crop_edge)*r_h, and W = (w+2*crop_edge)*r_w
            - map_data (MapData): typed SLAM map state.
            - c2w: (torch.Tensor): camera to world 3D transform.
            - seg_map (torch.Tensor): The segmentation maps on self.device with shape (H, W).
            - binary_maps (torch.Tensor): Tensor of shape (N, H, W) on self.device, where each pixel will have a value of 1 if it belongs to the nth segmentation mask, or 0 otherwise.
        Update:
            - map_data.points_ins_ids (torch.Tensor): updated ids of 3D instances associated to each 3d point with current keyframe segmentation.
        Return:
            - matched_ins_ids (List): Ids of 3D instances matched in current frame
            - binary_maps (torch.Tensor): The matched binary maps on self.device with shape (M, H, W).
            - n_matched_points (int): numbed of 3D points matched with 3D instances in current keyframe.
        """
        kf_id = self.kf_id
        image, depth, rgb_depth_ratio = frame_data
        points_3d, points_ids, points_ins_ids = map_data.points_3d, map_data.points_ids, map_data.points_ins_ids

        depth = torch.from_numpy(depth).to(self.device)
        camera_frustum_corners = geometry_utils.compute_camera_frustum_corners(depth, c2w, self.cam_intrinsics)
        frustum_mask = geometry_utils.compute_frustum_point_ids(points_3d, camera_frustum_corners, device=self.device)
        frustum_points_3d = points_3d[frustum_mask]

        if self.semantic_config.depth_filter:
            depth = geometry_utils.depth_filter(depth)

        matched_points_idxs, matches = geometry_utils.match_3d_points_to_2d_pixels(depth, torch.linalg.inv(c2w), frustum_points_3d, self.cam_intrinsics, self.semantic_config.match_distance_th)

        if len(rgb_depth_ratio) > 0:
            matches += rgb_depth_ratio[-1]
            matches[:,1] = (matches[:,1]*rgb_depth_ratio[0]).int()
            matches[:,0] = (matches[:,0]*rgb_depth_ratio[1]).int()
        matched_seg_idxs = seg_map[matches[:,1], matches[:,0]]

        frustum_points_ids, frustum_points_ins_ids = points_ids[frustum_mask], points_ins_ids[frustum_mask]
        frustum_points_ins_ids, matched_ins_info = self._track_objects(frustum_points_ids, frustum_points_ins_ids, matched_points_idxs, matched_seg_idxs, seg_map, self.semantic_config.track_th, kf_id)
        matched_ins_ids, binary_maps = self._fuse_masks_with_same_ins_id(binary_maps, matched_ins_info, kf_id)

        updated_points_ins_ids = points_ins_ids.clone()
        updated_points_ins_ids[frustum_mask] = frustum_points_ins_ids

        if self.semantic_config.debug_info:
            ins_maps = torch.ones(image.shape[:2], dtype=torch.int, device=self.device)*-1
            for ins_id, matches_info in matched_ins_info.items():
                for map_idx, _ in matches_info:
                    ins_maps[binary_maps[map_idx]] = ins_id
            self.kf_store.ins_maps.append(ins_maps.cpu().numpy())

        return matched_ins_ids, binary_maps, len(matched_points_idxs), updated_points_ins_ids

    def _track_objects(self, points_ids: torch.Tensor, points_ins_ids: torch.Tensor, matched_points_idxs: torch.Tensor, matched_seg_idxs: torch.Tensor, seg_map: torch.Tensor, track_th: float, kf_id: int) -> tuple[torch.Tensor, Dict[int, List[Tuple[int, int]]]]:
        """  We project 3D points and match with segmentation maps. Then we assign to each segmentation map the id of the 3D instance associated with the majority of points projected into it. If the set points don't have an object assigned, a new object is created and assigned to them. Points without an object assigned get assigned the segmentation map's instance.
        Args:
            - points_ids (torch.Tensor): ids to identify 3D points in case their order changes, or any of them is pruned, between keyframes.
            - points_ins_ids (torch.Tensor): ids of 3D instances associated to each 3d point.
            - matched_points_idxs (torch.Tensor): idxs in points_3d of N matched points.
            - matched_seg_idxs (torch.Tensor): (N) tensor of the indexes of the segmentation map matched to each of N 3D points.
            - seg_map (torch.Tensor): (H,W) tensor where each pixel stores the idx of the corresponding mask.
            - kf_id (int): current keyframe id.
        Return:
            - points_ins_ids (torch.Tensor): ids of 3D instances associated to each element of points_ids.
            - matched_ins_info (Dict[int, List[Tuple[int, int]]]]): Hash map storing for each observed 3D instance, a list of (matched mask index, mask area).
        """

        matched_ins_info = {}
        for map_idx in range(seg_map.max()+1):
            map_ins_id = -1
            map_points = matched_points_idxs[matched_seg_idxs == map_idx]
            if len(map_points)> track_th:
                mask_area = (seg_map == map_idx).sum().item()
                assigned_mask = points_ins_ids[map_points] > -1
                unassigned_points_ids = points_ids[map_points[~assigned_mask]].cpu().tolist()
                #Assign points to 3D instance, or create a new instance
                if assigned_mask.sum().item() > track_th:
                    map_ins_id = torch.mode(points_ins_ids[map_points[assigned_mask]]).values.item()
                    self.objects[map_ins_id].update(unassigned_points_ids, kf_id, mask_area)
                    if map_ins_id in matched_ins_info.keys():
                        matched_ins_info[map_ins_id].append((map_idx, mask_area))
                    else:
                        matched_ins_info[map_ins_id]=[(map_idx, mask_area)]

                elif len(unassigned_points_ids) > track_th:
                    map_ins_id = self.next_ins_id
                    self.next_ins_id +=1
                    #assigned points do not change obj id
                    self.objects[map_ins_id] = Instance3D(map_ins_id, kf_id=kf_id, points_ids=unassigned_points_ids, mask_area=mask_area)
                    matched_ins_info[map_ins_id]=[(map_idx, mask_area)]

                if map_ins_id > -1:
                    # Assign to matched unassigned points (id==-1) new instance id
                    points_ins_ids[map_points[~assigned_mask]] = map_ins_id

        return points_ins_ids, matched_ins_info

    def _fuse_masks_with_same_ins_id(self, binary_maps: torch.Tensor, matched_ins_info: Dict[int, List[Tuple[int, int]]], kf_id: int) -> Tuple[List[int], torch.Tensor] :
        """ A 3D object can be mapped to more than one 2D mask. We fuse masks that belong to the same ins_id, keeping idx of first occurence. Objects matched to fused masks are updated to the new masks areas.
        Args:
            - binary_maps (torch.Tensor): Tensor of shape (N, H, W) on self.device, where each pixel will have a value of 1 if it belongs to the nth segmentation mask, or 0 otherwise.
            - matched_ins_info (dict): Hash map storing for each observed 3D instance, a list of (matched mask index, mask area).
            - kf_id (int): current keyframe id.
        Return:
            - matched_ins_ids:
            - binary_maps (torch.Tensor): Updated binary maps on self.device with shape (M, H, W).
        """

        matched_ins_ids = []
        maps_idxs=[]
        to_pop = []
        i = 0
        for ins_id, data_list in matched_ins_info.items():
            map_idx = data_list[0][0]
            if len(data_list)>1:
                for j in range(1,len(data_list)):
                    binary_maps[map_idx] = torch.logical_or(binary_maps[map_idx], binary_maps[data_list[j][0]])

                mask = binary_maps[map_idx]
                mask_area = mask.sum().item()

                if self.semantic_config.n_top_views>0:
                    self.objects[ins_id].add_top_kf(kf_id, mask_area)

            if self.semantic_config.n_top_views<=0 or self.objects[ins_id].is_top_kf(kf_id):
                matched_ins_ids.append(ins_id)
                maps_idxs.append(map_idx)
                matched_ins_info[ins_id] = [(i, binary_maps[map_idx].sum().item())]
                i+=1
            else:
                to_pop.append(ins_id)
        for ins_id in to_pop:
            matched_ins_info.pop(ins_id)

        binary_maps = binary_maps[maps_idxs]

        return matched_ins_ids, binary_maps

    def compute_semantic_info(self) -> None:
        if len(self.keyframes_queue)>self.semantic_config.kf_queue_delay:
            self._compute_semantic_info()

    def complete_semantic_info(self) -> None:
        while len(self.keyframes_queue)>0:
            self._compute_semantic_info()

    def _compute_semantic_info(self) -> None:
        """ Compute semantic information of first keyframe in the queue.
        """
        matched_ins_ids, binary_maps, image, kf_id = self.keyframes_queue.popleft()

        if len(matched_ins_ids)>0:
            if self.semantic_config.n_top_views > 0:
                obj_to_compute = []
                for j, ins_id in enumerate(matched_ins_ids):
                    if self.objects[ins_id].is_top_kf(kf_id):
                        obj_to_compute.append(j)
                if len(obj_to_compute) == 0:
                    return
                matched_ins_ids, binary_maps = np.asarray(matched_ins_ids)[obj_to_compute].tolist(), binary_maps[obj_to_compute]

            # 1. CLIP - Always compute (core semantic)
            clip_embeds = self._extract_clip(image, binary_maps).cpu()
            self._update_matched_objects_clip(clip_embeds, matched_ins_ids, kf_id)

            # 2. Fusion Encoder - Conditional (PE, SAM3, etc.)
            if self.fusion_encoder is not None:
                self._compute_fusion_info(image, binary_maps, matched_ins_ids, kf_id)

            if self.semantic_config.log:
                frame_id = self.kf_store.frame_ids[kf_id]
                log_stats = {
                    "frame_id": frame_id,
                    "t_clip": round(self._time_cache.get("_extract_clip", 0), 2),
                    "t_up": round(self._time_cache.get("_update_matched_objects_clip", 0), 3),
                }
                if self.fusion_encoder is not None and "_compute_fusion_info" in self._time_cache:
                    log_stats["t_fusion"] = round(self._time_cache["_compute_fusion_info"], 2)

                self.logger.log_ovo_stats(log_stats, print_output=True)
                self._time_cache = {}

    @_profil
    def _compute_fusion_info(self, image: torch.Tensor, binary_maps: torch.Tensor, matched_ins_ids: List[int], kf_id: int) -> None:
        """Profiled call to fusion encoder to extract embeddings and update instances."""
        self.fusion_encoder.compute_and_update(
            FusionFrameInput(image=image, binary_maps=binary_maps, matched_ins_ids=matched_ins_ids, kf_id=kf_id),
            self.kf_store, self.objects
        )

    def _remove_deleted_keyframes(self, kfs: Dict) -> None:
        """ Remove keyframe information for deleted keyframes.
        Args:
            - kfs (Dict): SLAM keyframe dict keyed by frame_id; frames absent here are deleted.
        """
        for i, frame_id in enumerate(self.kf_store.frame_ids):
            if frame_id not in kfs:
                self.kf_store.cleanup_clip(i)

                if self.fusion_encoder is not None:
                    self.fusion_encoder.cleanup_keyframe(i, self.kf_store)

                self.kf_store.mark_deleted(i)

    def _remove_missing_instances(
        self,
        points_ins_ids: torch.Tensor,
        objects_list: list,
        objects_to_del: list,
    ) -> None:
        """
         Remove 3D instances that are not in existing_ins_ids.
        """
        map_ins_ids = points_ins_ids.unique()
        for ins_id in self.objects.keys():
            if ins_id in map_ins_ids:
                objects_list.append(self.objects[ins_id])
            else:
                objects_to_del.append(self.objects[ins_id])

    def update_map(self, map_data: "MapData", kfs: Dict) -> Tuple[torch.Tensor, list]:
        # 0. clean the queue
        self.complete_semantic_info()

        # 0.1 Remove deleted_kfs:
        self._remove_deleted_keyframes(kfs)

        # 1. Remove 3D instances that are not in pcd_obj_ids
        objects_list = []
        objects_to_del = []
        self._remove_missing_instances(map_data.points_ins_ids, objects_list, objects_to_del)

        # 2. Fuse 3D instances that fulfill a condition.
        result = self._fuse_overlapping_instances(objects_list, map_data.points_3d, map_data)

        print(f"Semantic Map update: removed {len(objects_to_del)}, fused {len(result.fused_objects)} instances")

        # 3. Update saved info
        self._update_descriptors_after_fusion(result.fused_objects)

        self.objects = result.objects

        # 4. Update object descriptors
        self.update_objects_clip()
        if self.fusion_encoder is not None:
            self.fusion_encoder.update_objects(self.objects, self.kf_store)

        return result.points_ins_ids, result.decisions

    def _fuse_overlapping_instances(
        self,
        objects_list: List[Instance3D],
        points_3d: torch.Tensor,
        map_data: "MapData",
    ) -> Tuple[Dict[int, Instance3D], Dict[int, int], torch.Tensor, list]:
        """
        Identify and fuse overlapping instances based on the fusion strategy.
        Returns:
            - objects: Dictionary of updated (surviving) Instance3D objects.
            - fused_objects: Dictionary mapping {deleted_instance_id: survivor_instance_id}.
            - points_ins_ids: Updated tensor of instance IDs for each 3D point.
            - decisions: List of fusion decision records.
        """
        # TODO: optimize brute-force approach (compare all instances to each-other)
        points_ins_ids = map_data.points_ins_ids
        obj_pcds = {}
        for instance in objects_list:
            obj_pcd = points_3d[points_ins_ids == instance.id]
            obj_pcds[instance.id] = InstanceGeometry(points=obj_pcd, centroid=obj_pcd.mean(axis=0))

        objects = {}
        fused_objects = {}
        for i, instance1 in enumerate(objects_list):
            if instance1.id in fused_objects:
                continue
            for instance2 in objects_list[i+1:]:
                if instance2.id in fused_objects:
                    continue
                elif self.fusion_strategy.same_instance(
                    instance1, instance2, obj_pcds[instance1.id], obj_pcds[instance2.id]
                ):
                    instance1, points_ins_ids = instance_utils.fuse_instances(instance1, instance2, (map_data.points_3d, map_data.points_ids, points_ins_ids))
                    fused_objects[instance2.id] = instance1.id
            objects[instance1.id] = instance1

        decisions = self.fusion_strategy.pop_decisions()
        return FusionResult(objects=objects, fused_objects=fused_objects, points_ins_ids=points_ins_ids, decisions=decisions)

    def _update_descriptors_after_fusion(self, fused_objects: Dict[int, int]) -> None:
        """
        Update keyframe descriptors and fusion encoder states after instance fusion.
        Args:
            - fused_objects: Dictionary mapping {deleted_instance_id: survivor_instance_id}.
        """
        for id2, id1 in fused_objects.items():
            for kf in self.objects[id2].kfs_ids:
                # Transfer CLIP descriptor from merged instance to survivor
                if kf in self.kf_store.clip and id2 in self.kf_store.clip[kf]:
                    self.kf_store.transfer_clip(id2, id1, kf)

                # Transfer fusion encoder descriptors (PE, SAM3, etc.)
                if self.fusion_encoder is not None:
                    self.fusion_encoder.transfer_on_merge([id2], id1, self.kf_store)

    @_profil
    def _extract_clip(self, image: torch.Tensor, binary_maps: torch.Tensor) -> List[Any]:
        """Profiled call to generators.clip.extract_clip. Computes a CLIP vector for each mask of the segmented image.
        Args:
            - image (torch.Tensor): Full source RGB image with dimensions (H,W,3) and range 0-255.
            - binary_maps (torch.Tensor): A tensor of (N, H, W) containing N binary maps, one for each segmented instance.
            - return_all: if True returns the three computed descriptors of each image in seg_images instead of merging them.
        Return:
            - climp_embeds: each level/key stores a list of numpy arrays with dim (N, self.clip_dim).
        """
        image = torch.from_numpy(image.transpose((2,0,1))).to(self.device)
        return self.generators.clip.extract_clip(image, binary_maps, self.semantic_config.return_all_clips).cpu()

    @_profil
    def _update_matched_objects_clip(self, clip_embeds: torch.Tensor, matched_ins_ids: List[int], kf_id: int) -> None:
        """
        Store clip_embeds keyframe information, and updates matched 3D instances' clip embeddings.
        Args:
            - clip_embeds (torch.Tensor): A tensor containing the clip embeddings.
            - matched_ins_ids (List[int]): A list of instance IDs that are matched with the clip embeddings.
            - kf_id (int): current keyframe id.
        Updates:
            self.kf_store.clip
        """
        ins_embeds = dict()
        for i, ins_id in enumerate(matched_ins_ids):
            if ins_id != -1:
                ins_embeds[ins_id] = clip_embeds[i]

        self.kf_store.clip[kf_id] = ins_embeds

        for ins_id in matched_ins_ids:
            self.objects[ins_id].update_clip(self.kf_store.clip)
        return

    def update_objects_clip(self, force_update: bool = False) -> None:
        """ Update all 3D instances descriptors
        Args:
            - force_update (bool): if True, recomputed Instance3D descriptors even Instance3D.to_update == False
        """
        for object in self.objects.values():
            object.update_clip(self.kf_store.clip, force_update=force_update)
        return

    @torch.no_grad()
    def classify_instances(self, classes: List[str], template: str | List[str] = "This is a photo of a {}", th: float = 0):
        """
        Classifies 3D instances based on provided classes and templates.
        Args:
            - classes (List[str]): A list of class names to classify the instances into.
            - templates (str | List[str], optional): A template or a list of templates to use for classification. If it's a list, the classes embeddings will be an ensembles of the templates.
            - th (float, optional): Minimum confidence to classify an instance. If highest confidenc is lower than th, the instance remains unclassified (-1). Default 0.
        Returns:
            dict: A dictionary containing:
                - "classes" (numpy.ndarray): An array of classes indices corresponding to each 3D instance.
                - "conf" (numpy.ndarray): An array of confidence scores for each classification.
        """

        sim_map = self.query(classes, template)
        instances_classes = torch.argmax(sim_map,dim=1)
        max_conf = torch.gather(sim_map, -1, instances_classes[:,None]).squeeze()
        instances_classes[max_conf <=th] = -1
        max_conf[max_conf<=th] = 0
        instances_info = {"classes": instances_classes.cpu().numpy(), "conf":max_conf.cpu().numpy()}
        return instances_info

    @torch.no_grad()
    def query(self, queries: List[str], templates: List[str] = ['{}'], ensemble: bool = False) -> torch.Tensor:
        """
        Queries the 3D instances using the provided queries and templates.
        Args:
            - queries (List[str]): A list of query strings to be used for querying the 3D instances.
            - templates (List[str], optional): A list of template strings to format the queries. Defaults to ['{}'].
            - ensemble (bool, optional): A flag indicating whether to use ensemble method for querying. Defaults to False.
        Returns:
            - torch.Tensor: A relevance map tensor of shape (len(queries), n_objs) indicating the similarity between the queries and the 3D instances.
        Raises:
            AssertionError: If there are no 3D instances to query.
        """
        assert len(self.objects) > 0, "No 3D instances to query!"
        obj_clips = self.get_objs_clips()
        relev_map = self.generators.clip.get_embed_txt_similarity(obj_clips.to(self.device), queries, templates=templates)
        return relev_map

    @torch.no_grad()
    def get_objs_clips(self) -> torch.Tensor:
        """ Retrieve all N 3D instances' descriptors.
        Return:
            - torch.Tensor: A tensor with shape (N, generators.clip.clip_dim) on self.device.
        """
        object_clips = torch.zeros((len(self.objects), self.generators.clip.clip_dim), device = self.device)
        for j, obj in enumerate(self.objects.values()):
            if obj.clip_feature is not None:
                object_clips[j] = obj.clip_feature.to(self.device)
            else:
                # This should never happen
                obj.to_update = True
                obj.update_clip(self.kf_store.clip)
                object_clips[j] = obj.clip_feature.to(self.device)
        return object_clips

    def capture_dict(self, debug_info: bool) -> Dict[str, Any]:
        """
        Captures the current state of the scene and returns it as a dictionary.
        Args:
            debug_info (bool): If True, includes additional debug information in the dictionary.
        Returns:
            dict: A dictionary containing the current state of the scene.
        """
        scene_dict = {
            "ins_3d_ids": np.asarray(list(self.objects.keys()))
        }
        for obj in self.objects.values():
            scene_dict.update(obj.export(debug_info))
        if debug_info:
            scene_dict["ins_map"] = np.array(self.kf_store.ins_maps)
            scene_dict.update(self.kf_store.capture(debug_info=True))
        return scene_dict

    def restore_dict(self, scene_dict: Dict[str, Any], debug_info: bool = False):
        """
        Restores the state of the object from a given scene dictionary.
        Args:
            scene_dict (Dict[str, Any]): A dictionary containing the scene data to restore.
            debug_info (bool, optional): If True, additional debug information will be restored. Defaults to False.
        Raises:
            Exception: Catches and ignores any exceptions during the restoration process.
        Notes:
            - Iterates through the keys of the scene_dict to restore Instance3D instances.
            - If debug_info is True, restores keyframes information including frame IDs, instance maps, and instance descriptors.
        """
        for i in scene_dict["ins_3d_ids"]:
            obj = Instance3D(i)
            obj.restore(scene_dict, debug_info)
            self.objects[obj.id] = obj
        if debug_info:
            self.kf_store.ins_maps = [x.squeeze() for x in np.split(scene_dict["ins_map"], len(scene_dict["frame_id"]))]
            self.kf_store.restore(scene_dict, self.objects.keys(), self.device)
