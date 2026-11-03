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
      **Ojo:** antes era un escalar y ahora `VoxelGrid.from_cfg` espera una lista de 3
      (`[x, y, z]`). Si algún config tiene `voxel_origin: 0.0`, peta al arrancar: pasarlo a lista.
- [ ] **Pasar los tests de voxel** antes de mergear la rama a `study/contest-realtime-online`:
      ```bash
      conda run -n ovo2 python -m pytest tests/unit/test_fusion_voxel.py tests/unit/test_fusion_broadphase.py tests/unit/test_fusion_planner.py tests/unit/test_fusion_geometry_prepare.py
      ```
      En local (Windows, torch CPU, `open3d` simulado) ya pasan los 29. Falta repetirlo en `ovo2`
      con el `open3d` real.
- [ ] **Mismo resultado que antes del refactor, en GPU.** El refactor no debe cambiar ninguna
      decisión de fusión. Repetir un mismo replay de fusión con `overlap_voxel` en la cadena
      (`fusion_criteria`) en `c6ef2f6` (antes del refactor) y en `refactor/voxel-set`, y comparar
      las decisiones (`fusion_decisions_eval.csv` / verdicts): las fusiones aceptadas tienen que
      salir idénticas. Ojo: desde el broadphase en eager (ver abajo) habrá menos filas `REJECTED`,
      porque las parejas sin celda compartida ya no se juzgan.
      En local (CPU) ya se comprobó con nubes aleatorias (varios `size`/`origin`, negativos,
      float64, fuera de rango) que el código viejo y el nuevo dan las mismas celdas y el mismo
      `shared`. Falta CUDA y datos reales.
- [ ] **Eval con `--voxel-overlap`.** `overlapping_pairs` ya no acepta `origin`. Correr
      `studies/fusion_metrics` con `--voxel-overlap` sobre un checkpoint y comprobar que da los
      mismos pares que en `c6ef2f6`. Buscar también si algún script lo llamaba con `origin=`:
      ```bash
      grep -rn "overlapping_pairs" scripts studies
      ```
- [ ] **Broadphase también en eager.** Antes solo `deferred_fusion: true` usaba el broadphase;
      `_fuse_eager` comparaba todas las parejas aunque la cadena fuera `[cos_sim, overlap_voxel]`.
      Comprobar en un replay eager con voxel: mismas fusiones que antes del arreglo, menos
      `n_pairs_evaluated` y menor `t_fusion`. Los tiempos de runs eager+voxel antiguos se midieron
      sin broadphase: no compararlos tal cual con los nuevos. Mirar qué `deferred_fusion` y
      `fusion_criteria` tenían los runs con voxel ya hechos.
- [ ] **Tabla plana de celdas en GPU (decidir antes de abrir la PR de voxel).** Hoy el broadphase
      encuentra las parejas y luego cada una vuelve a calcular su `shared` con un `isin` (una espera
      a la GPU por pareja). La alternativa ordena todas las celdas una vez y saca el `shared` de
      todas las parejas en la misma pasada. Correr:
      ```bash
      conda run -n ovo2 python scripts/bench_voxel_pair_table.py --device cuda
      ```
      El script ya comprueba que los dos métodos dan las mismas parejas, el mismo `shared` y las
      mismas fusiones. En local (CPU) fue así en todos los casos, y la tabla salió 2-9x más rápida
      (más cuanto más instancias: 187 ms frente a 21 ms con 500). Si en GPU también gana, cambiar
      el broadphase del fork a la tabla y llevar esa versión a la PR. Mirar además cuántas parejas
      candidatas por pasada salen en un run real (`n_pairs_evaluated`), que es lo que decide si
      compensa.
- [ ] **`owners` en `shared_cells`: `full_like` + `cat` frente a `repeat_interleave`.** Hoy se
      crea un `full_like` por set y se concatenan (N+1 llamadas a la GPU). La alternativa:
      ```python
      lengths = torch.tensor([len(v) for v in sets.values()], device=cells.device)
      owners = torch.repeat_interleave(torch.arange(n_sets, device=cells.device), lengths)
      ```
      da el mismo tensor con 2-3 llamadas. Medir en GPU con unos cientos de instancias; si no se
      nota, dejar la versión actual, que se lee más directa.
