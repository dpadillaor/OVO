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
core/                 # frontera con el modelo + lógica pura.
├── segmenters.py     # frame -> máscaras CRUDAS. SamSegmenter (SAM2), Sam3Segmenter
│                     #   (AMG oficial de SAM2 + predictor SAM3), Sam3PointPredictor
├── nms_decision.py   # NMS externo de OVO como cálculo puro -> DecisionBreakdown
├── profiling.py      # coste por crop (encoder/decode/VRAM) instrumentando el AMG (I/O)
├── timing_stats.py   # PURO: List[FrameProfile] -> TimingStats (mean/std/min/max)
│                     #   (por máscara: kept/removed + qué regla la mató y quién)
└── tests/            # paridad exacta con ovo.utils.segment_utils.mask_nms
viz/                  # rendering puro; consume DecisionBreakdown
├── pipeline_steps.py # tríptico: crudo -> poda OVO -> segmap
├── masks_gallery.py  # una imagen por máscara individual, con su veredicto OVO
├── removed_masks.py  # por cada máscara muerta: quién la mató + solape
├── decision_trace.py # tabla de auditoría de la poda (markdown)
├── point_ambiguity.py# pinchar UN punto -> las 3 máscaras dispares (agnóstico al modelo)
└── segmenter_compare.py # segmap final de sam2 vs sam3, lado a lado
cli.py                # orquesta (frame->core->viz), calcula rutas y escribe meta.json
results/              # salida, gitignored (regenerable)
```

Arquitectura: **I/O (cli) -> puro (core) -> pintado (viz)**, dependencias hacia abajo.

### Correr
```bash
# lentes del AMG (pipeline/masks/removed/trace) — SAM2 o SAM3
conda run -n ovo2 python -m studies.segmentation frame office0 70 --model sam2
conda run -n ovo2 python -m studies.segmentation frame office0 70 --model sam3
# pinchar un punto -> 3 máscaras multimask (--model sam2|sam3|both)
conda run -n ovo2 python -m studies.segmentation point office0 70 --xy 600 560 --model both
# segmap FINAL (tras poda): sam2_baseline vs sam3, lado a lado
conda run -n ovo2 python -m studies.segmentation compare office0 70
# coste: total/encoder/decode/VRAM, sam2 vs sam3 (reps para estabilidad)
conda run -n ovo2 python -m studies.segmentation timing office0 70 --reps 5
```
Flags: `--variant`, `--lenses`, `--points-per-side`, `--crop-n-layers`,
`--iou-thr/--score-thr/--inner-thr`.

### Organización de resultados (by_frame)
```
results/{escena}/f{frame:04d}/
├── {variante}/            # lentes del AMG (automático)
│   ├── meta.json          # config completa + umbrales + counts
│   ├── pipeline.png
│   ├── masks/ · removed/ · trace.md
└── point/                 # prompts interactivos (no AMG, cross-modelo)
    └── pt_{X}_{Y}/
        ├── sam2.png
        └── sam3.png        # frente A (Fase 3)
```
- **variante** = la config AMG entera (`sam2_baseline`, luego `sam2_multicrop`, `sam3`).
  Etiqueta corta; lo exhaustivo vive en `meta.json`.
- **point/** vive a nivel de frame, no dentro de una variante: el prompt interactivo no
  usa el AMG ni el NMS de OVO, y compara modelos (SAM2 vs SAM3).
- La ruta la **calcula el cli**, no se nombra a mano.

## Comparabilidad SAM2 vs SAM3 (importante para la tesis)
El AMG es **el mismo tubo** para los dos: `generate`/`_process_crop`/`_process_batch` de
Meta (byte a byte), mismo grid (pps=16, crop_n_layers=0), mismos filtros (pred_iou=0.8,
stability=0.95, box_nms=0.7) y el mismo NMS de OVO después (0.8/0.7/0.5). **Lo único que
cambia es el predictor.** Comparación apples-to-apples: mismo pipeline, otro modelo.

Qué modelo usa SAM3: `build_sam3_image_model` (modelo de IMAGEN, no vídeo). Su encoder es
un **`SAM3VLBackbone` de 816M params** (visión-lenguaje, open-vocab); el decoder de máscaras
es minúsculo (~11M). Frente a SAM2 Hiera-L (~214M), el backbone de SAM3 solo ya es ~3.8x
todo SAM2. De ahí: pesos ~4x, encoder ~2x en tiempo, activaciones idénticas.

Cautelas (no son perfectamente comparables):
- Cada predictor redimensiona/normaliza a la resolución nativa de su modelo: mismos puntos
  del grid, pero cada modelo "ve" la imagen distinto.
- `pred_iou`/`stability` son la **autoestima de cada modelo**, sin calibrar entre ellos:
  filtrar a 0.8 no es igual de estricto para los dos, así que el "23 vs 30 máscaras" mezcla
  diferencia real de segmentación con diferencia de calibración de score.
- El AMG pincha solo **puntos**: pagas un encoder visión-lenguaje de 816M sin usar su parte
  de lenguaje.

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
- **Fase 2** (hecha): motor de tiempos re-derivado limpio del worktree `sam2_amg_study`.
  `core/profiling.py` instrumenta el AMG por crop (encoder/decode/VRAM, con sync+warmup);
  `core/timing_stats.py` agrega (puro); `viz/timing.py` pinta barras; `timing` en el cli.
  Falta (opcional): barrer multi-crop (`--crop-n-layers>0`) y agregar sobre la trayectoria.
- **Fase 3**: SAM3.
  - Frente A (mismo punto, SAM2 vs SAM3): **hecho**. `point --model both` carga SAM3
    (`Sam3PointPredictor`, checkpoint `facebook/sam3` vía HF cache) y saca `compare.png`.
    SAM3 usa la vía **oficial de imagen** del repo sam3 (`build_sam3_image_model` +
    `Sam3Processor` + `predict_inst`), la misma que `examples/sam3_for_sam1_task_example`.
    Puntos canónicos de office0/f70 (los que en el AMG dan las máscaras ID4 e ID22):
    ID4=(1012,404), ID22=(1162,21), recuperados de `record['point_coords']`.
  - Frente B (AMG de SAM3): **hecho** (máscaras + coste). `frame --model sam3` usa
    `Sam3Segmenter` = la maquinaria **oficial** del AMG de SAM2 (grid+filtros+NMS de Meta,
    intacta) con el predictor interactivo del **modelo de imagen** de SAM3 (misma puerta que
    el point predictor; backbone enchufado a mano). NO se reusa el AMG rescatado (sin verificar).
    Todas las lentes de frame funcionan con SAM3 (viz agnóstica). office0/f70: SAM2 23->19,
    SAM3 30->23. Coste (`timing`): SAM3 ~1.4x total, encoder ~2x. VRAM SAM2 2.9GB / SAM3 6.0GB:
    la diferencia son los PESOS (SAM3 ~4x más pesado, 4GB vs 1GB); las activaciones son
    idénticas (~2GB). El AMG de SAM3 NO carga ningún SAM2 (se construye sin __init__).
- **Fase 4**: figuras finales para el TFM (se curan aparte, a `tfm/figures/`).

## Notas
- Trabajo original disperso, conservado hasta consolidar: `segmentation_exp/` (rama,
  forense) y `.claude/worktrees/sam2_amg_study/` (tiempos + AMG de SAM3 rescatado en la
  rama `feature/sam3_amg_recovered`).
