
baseline_data = {
    "metadata": {
        "dataset": "...",
        "scene": "...",
        "slam_backend": "...",
        "timestamp": "...",
        "config_hash": "...",
        "intrinsics": ...
    },
    "estimated_c2ws": {                      
        frame_id: c2w_matrix,
        # ...
    },
    "kfs": [                  
        frame_id_1, frame_id_2, ...
    ],
    "loop_closures": [
        {
            "frame_id": ...,         # Frame donde se detecta el loop closure
            "before": {
                "estimated_c2ws": ...,
                "kfs": ...,
                # "points_3d": {       # Nube de puntos antes del loop closure
                #     "pcd": ...,
                #     "pcd_ids": ...,
                #     "pcd_colors": ...
                # } Decidimos no calcularlo porque no hara falta
            },
            "after": {
                "estimated_c2ws": ...,
                "kfs": ...,
                # "points_3d": {       # Nube de puntos después del loop closure
                #     "pcd": ...,
                #     "pcd_ids": ...,
                #     "pcd_colors": ...
                # } Decidimos no calcularlo porque no hara falta
            }
        },
    ],
}