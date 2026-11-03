def fuse_instances(instance1, instance2, map_data):
    points_3d, points_ids, points_ins_ids = map_data
    
    instance1.add_points_ids(instance2.points_ids)
    for kf in instance2.kfs_ids:
        instance1.add_keyframes(kf)
    for (area, kf_id) in instance2.top_kf:
        instance1.add_top_kf(kf_id, area)
    points_ins_ids[points_ins_ids == instance2.id] = instance1.id
    return instance1, points_ins_ids