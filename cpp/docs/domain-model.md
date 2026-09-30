# Modelo de dominio (borrador)

Deriva de ADR-0002. Vocabulario: **entidad** = identidad que persiste aunque cambien sus datos;
**valor** = se define por sus datos, inmutable, barato de copiar; **agregado** = grupo con una raíz
que protege sus invariantes.

## Idea central
Dos lados, cada uno dueño de sus datos. El link entre ambos es el **`PointId`**:
estable (un LC mueve puntos sin cambiar su id) y nunca reutilizado.

```
GEOMETRÍA (escribe solo el SLAM)        SEMÁNTICA (escribe solo OVO)
PointId | xyz | color | obs              PointId | InstanceId
```

## Lado geometría (dueño: thread de geometría)
| Tipo | Clase | Contiene |
|---|---|---|
| `Frame` | valor | `FrameId`, RGB, profundidad |
| `PinholeCamera` | valor | fx, fy, cx, cy, ancho, alto; `project` / `unproject` |
| `Pose` | valor | transformación c2w |
| `PointBlock` | valor inmutable | `PointId[]`, xyz[], color[], obs[] de los puntos que creó un keyframe |
| `SlamKeyframe` | entidad | `FrameId`, `Pose`, su `PointBlock` |
| `GeometricMap` | agregado | los `SlamKeyframe` vivos (= el pointcloud) |

## Lado semántica (dueño: thread de semántica)
| Tipo | Clase | Contiene |
|---|---|---|
| `Mask` | valor transitorio | máscara 2D de SAM; vive durante un frame |
| `Descriptor` | valor | tipo de encoder (CLIP, PE...) + vector |
| `SemanticKeyframe` | entidad | id propio, `FrameId`, qué máscara fue a qué instancia |
| `Instance` | entidad | `InstanceId`, keyframes donde se vio, top-k vistas, `map<Encoder, Descriptor>` |
| `PointLabels` | datos | `PointId -> InstanceId` |
| `SemanticSubmap` | agregado | instancias + etiquetas + co-ocurrencia + evidencia del contest |
| `SemanticMap` | agregado | `SubmapId -> SemanticSubmap` (al principio uno) |

## Eventos geometría -> semántica
| Evento | Efecto en semántica |
|---|---|
| `PointsAdded{ids}` | nada inmediato; el tracking semántico los etiquetará |
| `BlockTransformed{kf, ids}` | etiquetas siguen válidas (mismo id); quizá invalidar cachés geométricas |
| `PointsRemoved{ids}` | borrar etiquetas; instancias sin puntos mueren |
| `SubmapOpened` / `SubmapsMerged` | futuro (ver visión en `cpp/CLAUDE.md`) |

## Recorrido de ejemplo
1. Frame 30: geometría crea puntos 100..199 -> `PointsAdded`.
2. Tracking semántico en frame 30: **lee** geometría, proyecta; punto 150 cae en máscara 3;
   `labels[150] == 7` -> por voto mayoritario máscara 3 = instancia 7; escribe etiquetas en **su** tabla.
3. LC mueve KF 30: xyz cambian, ids no -> etiquetas intactas.
4. ORB borra KF 30 -> `PointsRemoved{100..199}` -> semántica borra esas etiquetas.
5. Fusión 7 absorbe 12: solo semántica (`12 -> 7` en su tabla); geometría no se entera.

## Abierto
- Pertenencia guardada una sola vez: `PointLabels` (punto->instancia) o `Instance::points` (instancia->puntos).
  Python guarda ambas (`points_ins_ids` y `Instance3D.points_ids`) y pueden desincronizarse.
- Cómo lee la semántica la geometría (snapshot inmutable del `GeometricMap`).
