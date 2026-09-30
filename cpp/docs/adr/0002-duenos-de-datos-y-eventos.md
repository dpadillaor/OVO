# ADR-0002: Geometría y semántica son dueñas de sus datos y se comunican por eventos

- **Estado:** aceptado
- **Fecha:** 2026-09-30

## Contexto
En OVO Python la nube de puntos y la etiqueta punto->instancia (`pcd_obj_ids`) viven juntas dentro
del backbone SLAM. OVO las pide como tupla (`map_data`), las modifica y las devuelve
(`update_pcd_obj_ids`). Funciona en un solo hilo, pero con la semántica en un thread paralelo a
ORB-SLAM3 ambos escribirían los mismos datos.

Hechos del código actual (ver `ovo/slam/orbslam2.py`, `ovo/entities/ovo.py`):
- Un punto nace con un keyframe de SLAM, se mueve con él (transformación rígida en LC/BA) y muere con él.
- `PointId` es estable y nunca se reutiliza; el índice de fila no es identidad.
- Keyframe semántico (cada `segment_every`) y keyframe de SLAM son conceptos distintos, enlazados por `FrameId`.
- Una fusión nunca crea un `InstanceId` nuevo: una instancia absorbe a otra. El linaje solo queda en CSV.

Visión: a largo plazo el SLAM será propio, y habrá submapas semánticos abiertos por un detector de drift.

## Decisión
1. **Geometría** es dueña de puntos y poses. Publica eventos: puntos añadidos, bloque de puntos
   transformado (LC/BA), puntos eliminados (keyframe borrado), submapa abierto, submapas fusionados.
2. **Semántica** es dueña de instancias, descriptores y etiquetas punto->instancia (indexadas por `PointId`).
   Consume los eventos de geometría; nunca escribe en datos geométricos.
3. Puntos agrupados en bloques inmutables por keyframe de SLAM; un LC sustituye bloques (copy-on-write).
4. `SlamKeyframe` y `SemanticKeyframe` son tipos distintos, con ids distintos.
5. Las fusiones de instancias son eventos de dominio (`InstanceMerged`: superviviente, absorbida, motivo, evidencia).
6. **Preparado para submapas desde el día 1:** el mapa semántico se organiza por `SubmapId` (al principio
   uno solo) y la decisión de abrir submapa va detrás de `SubmapPolicy` (primera versión trivial).

## Por qué
- Sin datos compartidos mutables no hay carreras entre threads: cada lado muta solo lo suyo.
- El SLAM queda intercambiable (GT, ORB-SLAM3, Gaussian, SLAM propio): para la semántica es una fuente de eventos.
- Los submapas y el detector de drift se añaden sin rehacer el núcleo.
- El linaje de fusiones queda disponible para telemetría, evaluación y tesis.

## Consecuencias
- La semántica busca etiquetas por `PointId` en lugar de por fila (mitigable: los ids son densos y crecientes).
- Hay latencia entre un cambio geométrico y su reflejo semántico (el tiempo de procesar el evento).
- Pendiente: canal de eventos (colas, orden, backpressure) y cómo lee la semántica la geometría (snapshots).
