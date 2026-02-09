import rerun as rr
import numpy as np
import time
import multiprocessing

def stream_rerun(semantic_module, mpqueue, query_data, cam_intrinsic, scene_name, output_path, show):
    """
    Streams SLAM data to Rerun visualizer.
    """
    rr.init(f"OVO_{scene_name}", spawn=True)
    
    # Log camera intrinsics
    width = cam_intrinsic["width"]
    height = cam_intrinsic["height"]
    K = cam_intrinsic["intrinsic"]
    
    # Rerun uses Pinhole camera model. 
    # We define a camera at "world/camera"
    rr.log(
        "world/camera",
        rr.Pinhole(
            resolution=[width, height],
            image_from_camera=K,
        ),
        static=True
    )

    step = 0
    while True:
        try:
            # Check if queue has data
            if not mpqueue.empty():
                # Unpack data
                # points: (N, 3) float16
                # obj_ids: (N, L) int16
                # colors: (N, 3) float16 (0-255)
                # c2w: (4, 4) float16
                data = mpqueue.get()
                if data is None:
                    break
                    
                points, obj_ids, colors, c2w = data
                
                points = points.astype(np.float32)
                colors = colors.astype(np.uint8) # Rerun expects uint8 for 0-255 or float 0-1
                c2w = c2w.astype(np.float32)
                
                rr.set_time_sequence("step", step)
                
                # Log camera pose
                # c2w is Camera to World. Rerun Transform3D expects translation and rotation (mat3x3)
                # representing the transform from the child (camera) to the parent (world).
                rr.log(
                    "world/camera",
                    rr.Transform3D(
                        translation=c2w[:3, 3],
                        mat3x3=c2w[:3, :3],
                    )
                )

                # Log point cloud with RGB colors
                # We also log instance IDs as class_ids. 
                # Assuming the last column of obj_ids represents the finest instance segmentation.
                class_ids = None
                if obj_ids.ndim > 1 and obj_ids.shape[1] > 0:
                     class_ids = obj_ids[:, -1].astype(np.uint16)
                elif obj_ids.ndim == 1:
                     class_ids = obj_ids.astype(np.uint16)

                rr.log(
                    "world/points",
                    rr.Points3D(
                        points,
                        colors=colors,
                        class_ids=class_ids,
                    )
                )
                
                step += 1
            else:
                time.sleep(0.01)
                
        except Exception as e:
            print(f"Rerun stream error: {e}")
            break
