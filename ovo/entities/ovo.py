from typing import Any, Dict, List, Tuple
from collections import deque
import numpy as np
import pprint
import torch
import time

from ..utils import geometry_utils, instance_utils
from .clip_generator import CLIPGenerator
from .mask_generator import MaskGenerator
from .pe_generator import PEGenerator
from .instance3d import Instance3D
from .logger import Logger
from .fusion import create_fusion_strategy
from .fusion_encoders import FusionEncoderAdapter, PEFusionAdapter, DINOFusionAdapter, SAM3FusionAdapter

class OVO:
    """ Initialize CLIP and SAM backbones, with a given configuration, and logger.
    Args:
        - config (Dict[str, Any]): Configuration dictionary specifying hyperparameters and operational settings.
        - logger (Logger): Object for logging the tracking process.
        - scene_name (str, optional): Name of current scene required to load precomputed masks, or save them if precomputing. Default is None.
        - cam_intrinsics (torch.Tensor, optional): Camera intrinsic matrix. Required for scene reconstruction but not for evaluation. Default is None.
        - eval (bool, optional): If True SAM backbone is not loaded and the Camera intrinsic matrix is not required. Default is False.
        - device (str, optional): Device to run CLIP model and SAM. Must be either 'cpu' or 'cuda'. Default is 'cuda'.
    """
    def __init__(self, config: Dict[str, Any], logger: Logger, scene_name: str | None = None, cam_intrinsics: torch.Tensor | None = None, eval: bool = False, device = "cuda") -> None:
        if not eval:
            assert cam_intrinsics is not None, "Camera intrinsics required for reconstruction!"

        config["sam"]["multi_crop"] = False if config["clip"]["embed_type"] == "vanilla" else True
        self.cam_intrinsics = cam_intrinsics
        self.config = config
        self.logger = logger
        self.debug_info = config.get("debug_info", False)
        self.fusion_method = config.get("fusion_method", "CLIP")
        self.device = device
        self.n_top_views = config["clip"].get("k_top_views", 0)
        Instance3D.n_top_kf = self.n_top_views
        
        # For backward compatibility
        if "mask_res" in config["sam"] and "mask_res" not in config["clip"]:
            config["clip"]["mask_res"] = config["sam"]["mask_res"]

        self.clip_generator = CLIPGenerator(config["clip"], device=device)
        self.pe_generator = PEGenerator(config["pe"], device=device) if "pe" in config else None

        # Lazy import SAM3Generator to avoid dependency issues when SAM3 is not used
        if "sam3" in config:
            try:
                from .sam3_generator import SAM3Generator
                self.sam3_generator = SAM3Generator(config["sam3"], device=device)
            except ImportError as e:
                raise ImportError(
                    f"Failed to import SAM3Generator. SAM3 dependencies may not be installed: {e}"
                ) from e
        else:
            self.sam3_generator = None
        if not eval:
            self.mask_generator = MaskGenerator(config["sam"], scene_name, device=device)
        else:
            self.mask_generator = None
        self.keyframes = {
            "ins_descriptors": dict(),
            "ins_pe_descriptors": dict(),
            "ins_sam3_descriptors": dict(),
            "frame_id": list(),
            "ins_maps": list(),
        }
        self.keyframes_queue = deque([])
        self.objects=dict()
        self._time_cache = []
        self._last_visual_snapshot = None
        
        self.next_ins_id = 0
        self.kf_id = 0

        # Sem loop-closure parameters
        self.th_centroid = config.get("th_centroid", 1.5)
        self.th_cossim = config.get("th_cossim", 0.81)
        self.th_points = config.get("th_points", 0.1)

        # Initialize fusion strategy
        self.fusion_strategy = create_fusion_strategy(config)
        
        # Initialize fusion encoder adapter
        self.fusion_encoder = self._get_fusion_encoder()
        self._validate_fusion_config()

        if config.get("verbose", True):
            print('Semantic config')
            pprint.PrettyPrinter().pprint(config)


    def _get_fusion_encoder(self) -> FusionEncoderAdapter | None:
        """Get the fusion encoder adapter based on config, or None if CLIP-only."""
        fusion_method = self.fusion_method.lower()

        if fusion_method == "pe":
            return PEFusionAdapter(self.pe_generator)
        elif fusion_method == "dino":
            return DINOFusionAdapter(self.pe_generator) # DINO generator not implemented yet
        elif fusion_method == "sam3":
            return SAM3FusionAdapter(self.sam3_generator)

        # CLIP fusion uses CLIP features directly, no extra encoder needed
        return None

    def _validate_fusion_config(self):
        """Validate that fusion_method has required generator available."""
        method = self.fusion_method.lower()

        # Map fusion method to (generator_instance, readable_name)
        validation_map = {
            "pe": (self.pe_generator, "PE generator"),
            "sam3": (self.sam3_generator, "SAM3 generator"),
            # "dino": (self.dino_generator, "DINO generator"), # Future
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
        self.clip_generator.cpu()
        if self.pe_generator is not None:
            self.pe_generator.cpu()
        if self.sam3_generator is not None:
            self.sam3_generator.cpu()
        if self.mask_generator is not None:
            self.mask_generator.cpu()

    def cuda(self) -> None:
        """
        Move predictor model to cuda default device.
        """
        self.device = "cuda"
        self.clip_generator.cuda()
        if self.pe_generator is not None:
            self.pe_generator.cuda()
        if self.sam3_generator is not None:
            self.sam3_generator.cuda()
        if self.mask_generator is not None:
            self.mask_generator.cuda()

    def profil(func):
        """A decorator that profiles functions running time if self.config["log"] == True.
        Args:
            - func: The function to be decorated.
        Returns:
            - The wrapper function.
        """
        def wrapper(self, *args, **kwargs):
            if self.config.get("log", False):
                torch.cuda.synchronize()
                start_time = time.time()
                out = func(self, *args, **kwargs)
                torch.cuda.synchronize()
                end_time = time.time()
                self._time_cache.append(end_time-start_time)
                return out
            else:
                return func(self, *args, **kwargs)
        return wrapper    
    
    def detect_and_track_objects(self, frame_data: Tuple[int, np.ndarray, np.ndarray, Tuple[float, float, int]], map_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], c2w: torch.Tensor) -> torch.Tensor:
        r""" For the current frame (1) computes using SAM for each level i \in M, a set of segmentation maps; (2) track segmentation maps between frames projecting 3D points and associating the map to 3D instances, if 3D points don't have an associated 3D instance, create a new; (3) associate 3D points without an instance id to matched instances; (4) fuse 2D segments associated to the same 3D instance. 

        Args:
            - frame_data (tuple): current frame data.
                - frame_id (int): current frame id.
                - image (np.ndarray): RGB image with shape (H, W, 3).
                - depth (np.ndarray): Frame depth with shape (h, w).
                - rgb_depth_ratio (tuple): If H == h and W == w, tuple is empty, otherwise stores (r_h, r_w, crop_edge), such that H = (h+2*crop_edge)*r_h, and W = (w+2*crop_edge)*r_w
            - map_data (tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
                - points_3d (torch.Tensor): set of 3D point coordinates to match to current fram segmentation maps.
                - points_ids (torch.Tensor): ids to identify 3D points in case their order changes, or any of them is pruned, between keyframes.
                - points_ins_ids (torch.Tensor): ids of objects associated to each 3d point for each segmentation level of previous keyframes.
            - c2w: (torch.Tensor): camera to world 3D transform.
        Update: 
            - points_ins_ids (torch.Tensor): updated ids of 3D instances associated to each 3d point after current keyframe segmentation.
        """
        frame_id, image = frame_data[:2]
        self._last_visual_snapshot = None

        seg_maps, binary_maps = self._get_masks(image, frame_id)
        if len(seg_maps) == 0:
            print(f"No mask segmented in {frame_id}!")
            return None

        last_id = self.next_ins_id
        matched_ins_ids, binary_maps, n_matched_points, updated_ponts_ins_ids = self._match_and_track_instances(frame_data[1:], map_data, c2w, seg_maps, binary_maps)

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
        self.kf_id +=1

        if self.config.get("log", False):
            self.keyframes["frame_id"].append(frame_id)
            self.logger.log_ovo_stats(
                {
                    "frame_id":frame_id,
                    "n_obj":[self.next_ins_id-last_id],
                    "n_matches":n_matched_points, 
                    "t_sam":round(self._time_cache[0],2),
                    "t_obj":round(self._time_cache[1],3),
                },
                print_output=True
                )
            self._time_cache = []

        return updated_ponts_ins_ids

    def get_last_visual_snapshot(self) -> Dict[str, Any] | None:
        """Return latest segmented frame visual data for external stream consumers."""
        return self._last_visual_snapshot
    
    @profil
    def _get_masks(self, image: np.ndarray, frame_id: int):
        """ Profiled call to mask_generator to either compute segmentation maps for image, or load precomputed segments.
        Args:
            - frame_id (int): current frame id.
            - image (np.ndarray): RGB image with shape (H, W, 3).

        Returns:
            - seg_map (torch.Tensor): The segmentation maps on self.device with shape (H, W).
            - binary_maps (torch.Tensor): The binary maps on self.device with shape (N, H, W).
        """
        return self.mask_generator.get_masks(image, frame_id)
    
    @profil
    def _match_and_track_instances(self, frame_data: Tuple[int, np.ndarray, np.ndarray, Tuple[float, float, int]], map_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], c2w: torch.Tensor, seg_map: torch.Tensor, binary_maps: torch.Tensor) -> Tuple[List[int], torch.Tensor, int]:
        """ For the current frame (1) computes using SAM for each level i \\in M, a set of segmentation maps; (2) track segmentation maps between frames projecting 3D points and associating the map to 3D instances, if 3D points don't have an associated 3D instance, create a new; (3) associate 3D points without an instance id to matched instances; (4) fuse 2D segments associated to the same 3D instance. 

        Args:
            - frame_data (tuple): current frame data.
                - image (np.ndarray): RGB image with shape (H, W, 3).
                - depth (np.ndarray): Frame depth with shape (h, w).
                - rgb_depth_ratio (tuple): If H == h and W == w, tuple is empty, otherwise stores (r_h, r_w, crop_edge), such that H = (h+2*crop_edge)*r_h, and W = (w+2*crop_edge)*r_w
            - map_data (tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
                - points_3d (torch.Tensor): set of 3D point coordinates to match to current fram segmentation maps.
                - points_ids (torch.Tensor): ids to identify 3D points in case their order changes, or any of them is pruned, between keyframes.
                - points_ins_ids (torch.Tensor): ids of objects associated to each 3d point for each segmentation level of previous keyframes.
            - c2w: (torch.Tensor): camera to world 3D transform.
            - seg_map (torch.Tensor): The segmentation maps on self.device with shape (H, W).
            - binary_maps (torch.Tensor): Tensor of shape (N, H, W) on self.device, where each pixel will have a value of 1 if it belongs to the nth segmentation mask, or 0 otherwise.
        Update: 
            - points_ins_ids (torch.Tensor): updated ids of 3D instances associated to each 3d point with current keyframe segmentation.
        Return:
            - matched_ins_ids (List): Ids of 3D instances matched in current frame 
            - binary_maps (torch.Tensor): The matched binary maps on self.device with shape (M, H, W).
            - n_matched_points (int): numbed of 3D points matched with 3D instances in current keyframe.
        """
        kf_id = self.kf_id
        image, depth, rgb_depth_ratio = frame_data
        points_3d, points_ids, points_ins_ids = map_data

        depth = torch.from_numpy(depth).to(self.device)        
        camera_frustum_corners = geometry_utils.compute_camera_frustum_corners(depth, c2w, self.cam_intrinsics)
        frustum_mask = geometry_utils.compute_frustum_point_ids(points_3d, camera_frustum_corners, device=self.device)
        frustum_points_3d = points_3d[frustum_mask]

        if self.config.get("depth_filter", False):
            depth = geometry_utils.depth_filter(depth)

        matched_points_idxs, matches = geometry_utils.match_3d_points_to_2d_pixels(depth, torch.linalg.inv(c2w), frustum_points_3d, self.cam_intrinsics, self.config["match_distance_th"])

        if len(rgb_depth_ratio)>0:
            matches += rgb_depth_ratio[-1]
            matches[:,1] = (matches[:,1]*rgb_depth_ratio[0]).int()
            matches[:,0] = (matches[:,0]*rgb_depth_ratio[1]).int()
        matched_seg_idxs = seg_map[matches[:,1], matches[:,0]]

        frustum_points_ids, frustum_points_ins_ids = points_ids[frustum_mask], points_ins_ids[frustum_mask]
        frustum_points_ins_ids, matched_ins_info = self._track_objects(frustum_points_ids, frustum_points_ins_ids, matched_points_idxs, matched_seg_idxs, seg_map, self.config["track_th"], kf_id)
        matched_ins_ids, binary_maps = self._fuse_masks_with_same_ins_id(binary_maps, matched_ins_info, kf_id)

        updated_ponts_ins_ids = points_ins_ids.clone()
        updated_ponts_ins_ids[frustum_mask] = frustum_points_ins_ids # Updates points_ins_ids

        if self.config.get("debug_info", False):
            ins_maps = torch.ones(image.shape[:2], dtype=torch.int, device=self.device)*-1
            for ins_id, matches_info in matched_ins_info.items():
                for map_idx, _ in matches_info:
                    ins_maps[binary_maps[map_idx]] = ins_id
            self.keyframes["ins_maps"].append(ins_maps.cpu().numpy())

        return matched_ins_ids, binary_maps, len(matched_points_idxs), updated_ponts_ins_ids
            
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
                    # Assignto matched unassigned points (id==-1) new instance id 
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

                if self.n_top_views>0:
                    self.objects[ins_id].add_top_kf(kf_id, mask_area)

            if self.n_top_views<=0 or self.objects[ins_id].is_top_kf(kf_id):
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
        if len(self.keyframes_queue)>self.config.get("kf_queue_delay", 0):
            self._compute_semantic_info() 

    def complete_semantic_info(self) -> None:
        while len(self.keyframes_queue)>0:
            self._compute_semantic_info()
    
    def _compute_semantic_info(self) -> None:
        """ Compute semantic information of first keyframe in the queue.
        """
        matched_ins_ids, binary_maps, image, kf_id = self.keyframes_queue.popleft()

        if len(matched_ins_ids)>0:
            if self.n_top_views > 0:
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

            # 2. Fusion Encoder - Conditional (PE, DINO, SAM3, etc.)
            if self.fusion_encoder is not None:
                self._compute_fusion_info(image, binary_maps, matched_ins_ids, kf_id)

            if self.config.get("log", False):
                frame_id = self.keyframes["frame_id"][kf_id]
                log_stats = {
                    "frame_id":frame_id,
                    "t_clip": round(self._time_cache[0],2),
                    "t_up": round(self._time_cache[1],3)
                }
                # Log fusion stats (PE/DINO/SAM3)
                if self.fusion_encoder is not None and len(self._time_cache) > 2:
                     log_stats["t_fusion"] = round(self._time_cache[2], 2)

                self.logger.log_ovo_stats(log_stats, print_output=True)
                self._time_cache = []

    @profil
    def _compute_fusion_info(self, image: torch.Tensor, binary_maps: torch.Tensor, matched_ins_ids: List[int], kf_id: int) -> None:
        """Profiled call to fusion encoder to extract embeddings and update instances."""
        self.fusion_encoder.compute_and_update(
            image, binary_maps, matched_ins_ids, kf_id,
            self.keyframes, self.objects
        )
    
    def _remove_deleted_keyframes(self, kfs: List[int]) -> None:
        """ Remove keyframe information for deleted keyframes.
        Args:
            - deleted_kfs (List[int]): List of deleted keyframe ids.
        """
        deleted_kfs = []
        for i, kf in enumerate(self.keyframes["frame_id"]):
            if kf not in kfs:
                deleted_kfs.append(kf)
                if kf in self.keyframes["ins_descriptors"]: # Not all Keyframes will have descriptors associated
                    self.keyframes["ins_descriptors"].pop(kf)
                
                # Delegate cleanup to fusion encoder (handles PE/SAM3/DINO descriptors if active)
                if self.fusion_encoder is not None:
                    self.fusion_encoder.cleanup_keyframe(kf, self.keyframes)

                self.keyframes["frame_id"][i] = "Deleted" #deleting from self.keyframes["frame_id"] would require a checkpoint refactor to change it from list to dict
                # self.keyframes["ins_maps"] # This variable is for Debug, better to not remove it

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

    def update_map(self, map_data, kfs):
        # 0. clean the queue
        self.complete_semantic_info()
        points_3d, _, points_ins_ids = map_data

        # 0.1 Remove deleted_kfs :
        self._remove_deleted_keyframes(kfs)

        # 1. remove 3D instances that are not in pcd_obj_ids
        objects_list = []
        objects_to_del = []
        self._remove_missing_instances(points_ins_ids, objects_list, objects_to_del)

        # 2. Fuse 3D instances that fulfill a condition.
        new_objects, fused_objects, points_ins_ids, fusion_decisions = self._fuse_overlapping_instances(
            objects_list, points_3d, map_data
        )

        print(f"Semantic Map update: removed {len(objects_to_del)}, fused {len(fused_objects)} instances")

        # 3. Updated saved info
        self._update_descriptors_after_fusion(fused_objects)

        self.objects = new_objects
        # 4. Update object descriptors
        self.update_objects_clip()
        if self.fusion_encoder is not None:
            self.fusion_encoder.update_objects(self.objects, self.keyframes)

        # Explicitly update PE if generator is available but not using fusion_encoder (for backward compatibility)
        if self.pe_generator is not None and self.fusion_encoder is None:
            self.update_objects_pe()

        return points_ins_ids, fusion_decisions

    def _fuse_overlapping_instances(
        self,
        objects_list: List[Instance3D],
        points_3d: torch.Tensor,
        map_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[Dict[int, Instance3D], Dict[int, int], torch.Tensor]:
        """
        Identify and fuse overlapping instances based on the fusion strategy.
        Returns:
            - objects: Dictionary of updated (surviving) Instance3D objects.
            - fused_objects: Dictionary mapping {deleted_instance_id: survivor_instance_id}.
            - points_ins_ids: Updated tensor of instance IDs for each 3D point.
        """
        # TODO: optimize brute-force approach (compare all instances to each-other)
        # Precompute pointcloud data for efficiency
        _, _, points_ins_ids = map_data
        obj_pcds = {}
        for instance in objects_list:
            obj_pcd = points_3d[points_ins_ids == instance.id]
            obj_pcds[instance.id] = [obj_pcd, obj_pcd.mean(axis=0)]

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
                    instance1, points_ins_ids = instance_utils.fuse_instances(instance1, instance2, map_data)
                    fused_objects[instance2.id] = instance1.id
            objects[instance1.id] = instance1

        decisions = self.fusion_strategy.pop_decisions()
        return objects, fused_objects, points_ins_ids, decisions

    def _update_descriptors_after_fusion(self, fused_objects: Dict[int, int]) -> None:
        """
        Update keyframe descriptors and fusion encoder states after instance fusion.
        Args:
            - fused_objects: Dictionary mapping {deleted_instance_id: survivor_instance_id}.
        """
        for id2, id1 in fused_objects.items():
            # Access old state (self.objects) before it is updated
            for kf in self.objects[id2].kfs_ids:
                # Handle CLIP descriptors
                if kf in self.keyframes["ins_descriptors"] and id2 in self.keyframes["ins_descriptors"][kf]:
                    # If both ins were observed in the same frame, the ins_maps should be fused and descriptors recomputed.
                    # Nevertheless, it is not probable that two instances seen in the same kf will fulfill the distance threshold
                    ins_descriptor2 = self.keyframes["ins_descriptors"][kf].pop(id2)
                    if id1 not in self.keyframes["ins_descriptors"][kf] or True:
                        self.keyframes["ins_descriptors"][kf][id1] = ins_descriptor2

                # Handle Fusion Encoder descriptors (PE, DINO, SAM3, etc.)
                if self.fusion_encoder is not None:
                    self.fusion_encoder.transfer_on_merge([id2], id1, self.keyframes)
    
    @profil
    def _extract_clip(self, image: torch.Tensor, binary_maps: torch.Tensor) -> List[Any]:
        """Profiled call to self.clip_generator.extract_clip. Computes a CLIP vector for each mask of the segmented image.
        Args:
            - image (torch.Tensor): Full source RGB image with dimensions (H,W,3) and range 0-255.
            - binary_maps (torch.Tensor): A tensor of (N, H, W) containing N binary maps, one for each segmented instance.
            - return_all: if True returns the three computed descriptors of each image in seg_images instead of merging them.
        Return:
            - climp_embeds: each level/key stores a list of numpy arrays with dim (N, self.clip_dim).    
        """
        image = torch.from_numpy(image.transpose((2,0,1))).to(self.device)
        return self.clip_generator.extract_clip(image, binary_maps, self.config.get("return_all_clips", False)).cpu()
    
    @profil
    def _update_matched_objects_clip(self, clip_embeds: torch.Tensor, matched_ins_ids: List[int], kf_id: int) -> None:  
        """
        Store clip_embeds keyframe information, and updates matched 3D instances' clip embeddings.
        Args:
            - clip_embeds (torch.Tensor): A tensor containing the clip embeddings.
            - matched_ins_ids (List[int]): A list of instance IDs that are matched with the clip embeddings.
            - kf_id (int): current keyframe id.
        Updates:
            self.keyframes["ins_descriptors"]
        """

        ins_embeds = dict()
        for i, ins_id in enumerate(matched_ins_ids):
            if ins_id != -1:
                ins_embeds[ins_id] = clip_embeds[i]

        # Save keyframe information
        self.keyframes["ins_descriptors"][kf_id] = ins_embeds
        
        for ins_id in matched_ins_ids:
            self.objects[ins_id].update_clip(self.keyframes["ins_descriptors"])
        return
    
    def update_objects_clip(self, force_update: bool = False) -> None:
        """ Update all 3D instances descriptors
        Args:
            - force_update (bool): if True, recomputed Instance3D descriptors even Insatance_3D.to_update == False
        """
        for object in self.objects.values():
            object.update_clip(self.keyframes["ins_descriptors"], force_update=force_update)
        return

    @profil
    def _extract_pe(self, image: torch.Tensor, binary_maps: torch.Tensor) -> torch.Tensor:
        """Profiled call to self.pe_generator.extract_pe. Computes a PE vector for each mask of the segmented image.
        Args:
            - image (torch.Tensor): Full source RGB image with dimensions (H,W,3) and range 0-255.
            - binary_maps (torch.Tensor): A tensor of (N, H, W) containing N binary maps, one for each segmented instance.
        Return:
            - pe_embeds: tensor with dim (N, self.pe_generator.embed_dim).
        """
        image = torch.from_numpy(image.transpose((2,0,1))).to(self.device)
        return self.pe_generator.extract_pe(image, binary_maps).cpu()

    @profil
    def _update_matched_objects_pe(self, pe_embeds: torch.Tensor, matched_ins_ids: List[int], kf_id: int) -> None:
        """
        Store pe_embeds keyframe information, and updates matched 3D instances' PE embeddings.
        Args:
            - pe_embeds (torch.Tensor): A tensor containing the PE embeddings.
            - matched_ins_ids (List[int]): A list of instance IDs that are matched with the PE embeddings.
            - kf_id (int): current keyframe id.
        Updates:
            self.keyframes["ins_pe_descriptors"]
        """
        ins_embeds = dict()
        for i, ins_id in enumerate(matched_ins_ids):
            if ins_id != -1:
                ins_embeds[ins_id] = pe_embeds[i]

        # Save keyframe information
        self.keyframes["ins_pe_descriptors"][kf_id] = ins_embeds

        for ins_id in matched_ins_ids:
            self.objects[ins_id].update_pe(self.keyframes["ins_pe_descriptors"])
        return

    def update_objects_pe(self, force_update: bool = False) -> None:
        """ Update all 3D instances PE descriptors
        Args:
            - force_update (bool): if True, recomputed Instance3D PE descriptors even Instance_3D.to_update == False
        """
        if self.pe_generator is None:
            return
        for object in self.objects.values():
            object.update_pe(self.keyframes["ins_pe_descriptors"], force_update=force_update)
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
        relev_map = self.clip_generator.get_embed_txt_similarity(obj_clips.to(self.device), queries, templates=templates)
        return relev_map

    @torch.no_grad()
    def get_objs_clips(self) -> torch.Tensor:
        """ Retrieve all N 3D instances' descriptors.
        Return:
            - torch.Tensor: A tensor with shape (N, self.clip_generator.clip_dim) on self.device.
        """    
        object_clips = torch.zeros((len(self.objects), self.clip_generator.clip_dim), device = self.device)
        for j, obj in enumerate(self.objects.values()):
            if obj.clip_feature is not None:
                object_clips[j] = obj.clip_feature.to(self.device)
            else:
                # This should never happen
                obj.to_update = True
                obj.update_clip(self.keyframes["ins_descriptors"])
                object_clips[j] = obj.clip_feature.to(self.device)
        return object_clips    

    def capture_dict(self, debug_info: bool) -> Dict[str, Any]:
        """
        Captures the current state of the scene and returns it as a dictionary.
        Args:
            debug_info (bool): If True, includes additional debug information in the dictionary.
        Returns:
            dict: A dictionary containing the current state of the scene. If debug_info is True,
                  the dictionary will also include frame IDs, default object maps, and object descriptors.
        """
        scene_dict = {
            "ins_3d_ids": np.asarray(list(self.objects.keys()))
        }
        for obj in self.objects.values():
            scene_dict.update(obj.export(debug_info))
        if debug_info:
            scene_dict["frame_id"] = np.array(self.keyframes["frame_id"])
            scene_dict["ins_map"] = np.array(self.keyframes["ins_maps"])
            for kf_id, ins_descriptors in self.keyframes["ins_descriptors"].items():
                for ins_id, descriptors in ins_descriptors.items():
                    scene_dict[f"kf_{kf_id}_ins3d_{ins_id}_clips"] = descriptors.cpu().numpy()
            for kf_id, ins_pe_descriptors in self.keyframes["ins_pe_descriptors"].items():
                for ins_id, descriptors in ins_pe_descriptors.items():
                    scene_dict[f"kf_{kf_id}_ins3d_{ins_id}_pe"] = descriptors.cpu().numpy()
            for kf_id, ins_sam3_descriptors in self.keyframes["ins_sam3_descriptors"].items():
                for ins_id, descriptors in ins_sam3_descriptors.items():
                    scene_dict[f"kf_{kf_id}_ins3d_{ins_id}_sam3"] = descriptors.cpu().numpy()
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
            self.keyframes["frame_id"] = list(scene_dict["frame_id"])
            self.keyframes["ins_maps"] = [x.squeeze() for x in np.split(scene_dict["ins_map"], len(self.keyframes["frame_id"]))]
            for i in range(len(self.keyframes["frame_id"])):
                self.keyframes["ins_descriptors"][i] = {}
                self.keyframes["ins_pe_descriptors"][i] = {}
                self.keyframes["ins_sam3_descriptors"][i] = {}
                for ins_id in self.objects.keys():
                    descriptor = scene_dict.get(f"kf_{i}_ins3d_{ins_id}_clips", None)
                    if descriptor is not None:
                        self.keyframes["ins_descriptors"][i][ins_id] = torch.tensor(descriptor, device=self.device)
                    pe_descriptor = scene_dict.get(f"kf_{i}_ins3d_{ins_id}_pe", None)
                    if pe_descriptor is not None:
                        self.keyframes["ins_pe_descriptors"][i][ins_id] = torch.tensor(pe_descriptor, device=self.device)
                    sam3_descriptor = scene_dict.get(f"kf_{i}_ins3d_{ins_id}_sam3", None)
                    if sam3_descriptor is not None:
                        self.keyframes["ins_sam3_descriptors"][i][ins_id] = torch.tensor(sam3_descriptor, device=self.device)