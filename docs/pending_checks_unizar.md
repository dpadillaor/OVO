# Pendiente de comprobar en unizar

Cosas que solo se pueden verificar en la máquina de unizar (entorno `ovo2`, GPU, datos de
experimentos). Marcar al comprobar y borrar cuando ya no aporten.

## Refactor voxel (`refactor/voxel-set`)

- [ ] **`voxel_origin` nunca se usa.** Buscar en configs y en los configs guardados de los
      experimentos:
      ```bash
      grep -rn "voxel_origin" data/working/configs data/output scripts
      ```
      En local solo aparece leído en `ovo/entities/fusion/factory.py` con default `0.0`.
      Si sale vacío, se puede quitar `origin` de `VoxelGrid`.
- [ ] **Pasar los tests de voxel** antes de mergear la rama a `study/contest-realtime-online`:
      ```bash
      conda run -n ovo2 python -m pytest tests/unit/test_fusion_voxel.py tests/unit/test_fusion_broadphase.py tests/unit/test_fusion_planner.py tests/unit/test_fusion_geometry_prepare.py
      ```
