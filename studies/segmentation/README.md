# studies/segmentation

Estudio de la **etapa de máscara** de OVO: cómo se generan y se podan las máscaras de
cada frame, y qué cuesta mejorarlas. La pregunta de fondo, de cara al TFM:

> La segmentación por frame es un **techo** para OVO: la asignación hereda una máscara y
> no la vuelve a cuestionar. ¿Qué produce esa etapa, por qué descarta lo que descarta, y
> qué cuesta sacar mejores máscaras?

Dos caras de la misma pregunta:
- **cualitativa**: qué máscaras salen y por qué se podan (baseline actual).
- **cuantitativa**: cuánto cuesta trocear para sacar más (multi-crop, tiempos).

## Qué hay ahora (Fase 1: baseline SAM2, cualitativo)

```
core/                 # PURO. Ni modelo, ni disco, ni pintado.
├── segmenters.py     # frontera con el modelo: frame -> máscaras CRUDAS de SAM2
├── nms_decision.py   # NMS externo de OVO como cálculo puro -> DecisionBreakdown
│                     #   (por máscara: kept/removed + qué regla la mató y quién)
└── tests/            # paridad exacta con ovo.utils.segment_utils.mask_nms
viz/                  # rendering puro; consume DecisionBreakdown
├── pipeline_steps.py # tríptico: crudo -> poda OVO -> segmap
├── masks_gallery.py  # una imagen por máscara individual, con su veredicto OVO
├── removed_masks.py  # por cada máscara muerta: quién la mató + solape
└── decision_trace.py # tabla de auditoría de la poda (markdown)
cli.py                # orquesta (frame->core->viz), calcula rutas y escribe meta.json
results/              # salida, gitignored (regenerable)
```

Arquitectura: **I/O (cli) -> puro (core) -> pintado (viz)**, dependencias hacia abajo.

### Correr
```bash
conda run -n ovo2 python -m studies.segmentation office0 70
```
Flags: `--variant`, `--lenses pipeline masks`, `--points-per-side`, `--crop-n-layers`,
`--iou-thr/--score-thr/--inner-thr`.

### Organización de resultados (by_frame)
```
results/{escena}/f{frame:04d}/{variante}/
├── meta.json     # config completa + umbrales + counts (autodescriptivo)
├── pipeline.png
└── masks/
```
- **variante** = la config entera que generó las máscaras (`sam2_baseline`, luego
  `sam2_multicrop`, `sam3`). Etiqueta corta; lo exhaustivo vive en `meta.json`.
- La ruta y el `meta.json` los **calcula el cli**, no se nombran a mano.

## Alcance actual (honesto)
- Es **un frame suelto** (baseline cualitativa), no evolución temporal entre frames.
- La evolución mostrada empieza en las máscaras **ya filtradas por el NMS interno de
  SAM2** (cajas). Ese primer paso (candidatas -> crudo) no se disecciona aquí.

## A dónde se pretende llegar (TFM)
1. **El problema** (hecho): un punto -> N máscaras ambiguas; poda por heurística. Baseline.
2. **Cómo funciona**: baseline SAM2 + NMS de OVO, de punta a punta.
3. **SAM3, frente A**: mismo punto, SAM2 vs SAM3 (comparación cualitativa).
4. **SAM3, frente B**: el AMG de SAM3 replicado -> qué máscaras saca vs SAM2 y cuánto le
   cuesta el encoder troceado.

### Fases
- **Fase 1** (hecha): baseline SAM2, forense de poda completo. Lentes `pipeline`,
  `masks`, `removed` (detalle de cada máscara muerta) y `trace` (tabla de auditoría).
- **Fase 2**: motor de tiempos (viene del worktree `sam2_amg_study`): Pipeline A vs B,
  coste multi-crop, agregación por escena.
- **Fase 3**: SAM3 (frentes A y B), reusando el motor de tiempos de la Fase 2.
- **Fase 4**: figuras finales para el TFM (se curan aparte, a `tfm/figures/`).

## Notas
- Trabajo original disperso, conservado hasta consolidar: `segmentation_exp/` (rama,
  forense) y `.claude/worktrees/sam2_amg_study/` (tiempos + AMG de SAM3 rescatado en la
  rama `feature/sam3_amg_recovered`).
