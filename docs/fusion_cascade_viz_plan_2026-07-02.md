# Visualización del cascade de fusión + ramas del OR — plan (2026-07-02)

Conclusiones de la sesión sobre cómo visualizar los datos nuevos del estudio
`studies/fusion_metrics`: la vista **`by_criterion`** (cada criterio de fusión como
clasificador en cascada) y las **ramas del OR de aceptación** (`accept_modes`
A/B/AB). Documento vivo — decisiones de diseño, no implementación cerrada.

## 1. Qué datos tenemos

Por escena, en `fusion_eval_summary.json > verdicts`:

- **`totals`** — `{eval, accepted, rejected}`.
- **`counts`** — TP/FP/FN/TN globales (positivo = merge).
- **`by_criterion`** — lista ordenada por cadena. Cada gate como clasificador
  (positivo = "deja pasar"):
  `eval, pass, reject, TP, TN, FP, FN, precision, recall, f1`.
  El gate **terminal** lleva `accept_modes`: por rama del OR
  (`A` = geometría `p_dist>0.5`, `B` = semántica `cos_sim>0.9 & p_dist>0.2`,
  `AB` = ambas) → `{pass, TP, FP}`.
- **`by_epoch`** — lo mismo, partido por bucket de drift (pre/post jump).

Semántica clave:
- Un par **aceptado** solo puede ser TP/FP. Un par **rechazado** solo TN/FN.
  Por eso `accept_modes` no tiene FN/TN (solo desglosa el `pass`).
- La cadena se infiere de los `reason` observados en el CSV (orden canónico),
  robusto a configs sin `fusion_method`/`fusion_criteria`.

## 2. Qué queremos comunicar

1. **El embudo**: cómo se estrecha el stream gate a gate (eval→pass), y dónde
   mueren los TP (buenos) frente a los FP (malos).
2. **Rol de cada gate**: cuál filtra fino (sube precision) vs cuál destroza
   recall (mata merges buenos).
3. **Rol del OR semántico**: cuánto aporta la rama B (semántica) y a qué coste de
   precisión, frente a A (geometría).

## 3. Catálogo de visualizaciones

### Single-run (analizar un experimento/escena)
| Viz | Qué muestra | Estado |
|---|---|---|
| **Sankey embudo** | Flujo entrada→gates→accept A/B/AB, ramas de rechazo colgando, color same/diff. Replica el diagrama-boceto en una figura. | propuesta |
| Waterfall de recall | TP acumulado cayendo gate a gate (dónde se pierde recall). | propuesta |
| `cascade_bars` | TP/FP que deja pasar cada gate + P/R. | implementada |
| `gate_waterfall` | FN/TN por gate de rechazo. | implementada |
| `accept_mode_bar` | TP/FP por rama A/B/AB del terminal. | implementada |

### Multi-run (comparar N experimentos)
| Viz | Qué muestra | Nota |
|---|---|---|
| **Trayectorias precision-recall** | Cada run una polilínea en el plano P-R, un punto por gate en orden de cadena. Qué pipeline llega más arriba-derecha y *cómo*. | Escala ~5-8 runs; robusto aunque las cadenas difieran. **La joya comparativa.** |
| Grouped bars por gate | Mismo gate, barras por run, métrica seleccionable (F1/P/R). | Requiere gates con nombre común. |
| Heatmap runs×gates | Filas=runs, cols=gates, celda=F1. | Compacto; barridos/ablation grandes. |
| Comparar terminal (ramas) | Barras A/B/AB por run con su precision. | Rol del OR semántico entre configs. |

## 4. Recomendación por destino

El destino cambia los criterios de diseño:

### Exploración / dashboard interactivo (uso propio)
- Plotly, HTML, interactivo. Sankey embudo como pieza central single-run;
  trayectorias P-R para comparar. Aquí vale toda la riqueza.

### Defensa TFM (slides)
- El **Sankey** brilla: explica el mecanismo de un vistazo, animable.
- Trayectorias P-R para el "por qué una config gana".

### Documento TFM / paper (impreso, revisado)
Criterios: **estático, vectorial (PDF/SVG), colorblind-safe, legible en B/N,
compacto, autoexplicativo**. Los dashboards interactivos **no** van aquí.

Prioridad:
1. **Tabla LaTeX por criterio** (`by_criterion` → `tabular`): el dato duro que un
   revisor quiere. `eval/pass/reject/TP/FP/TN/FN/P/R/F1` por gate. Máximo valor,
   mínimo esfuerzo.
2. **Figura precision-recall multi-run** en **matplotlib** (control vectorial fino,
   no plotly). Cada config una polilínea, markers/linestyle distintos para B/N,
   paleta Okabe-Ito.
3. **Figura de método**: embudo estático (tipo el boceto, con conteos + color
   TP/FP) para la sección de mecanismo. Una sola, ilustrativa.
4. **Heatmap runs×gate** solo si hay ablation/barrido amplio.

Estilo paper: matplotlib + PDF, fuente ≥8pt, Okabe-Ito, markers+linestyle para
robustez en escala de grises.

## 5. Roadmap propuesto

1. **Exportador LaTeX** de `by_criterion` (rápido, alto valor para TFM).
2. **Sankey embudo** single-run (dashboard/defensa).
3. **Trayectorias P-R multi-run** — matplotlib (paper) + plotly (dashboard).
4. Figura de método (embudo estático).
5. Heatmap runs×gate (si se necesita ablation grande).

## 6. Decisiones abiertas

- Qué runs concretos entran en la comparación del TFM (define el eje de las
  figuras multi-run).
- Idioma del documento final (afecta labels de las figuras).
- Si el Sankey de método se genera estático (matplotlib/graphviz) o se exporta el
  plotly a SVG.

## Referencias

- Datos: `studies/fusion_metrics/core/agnostic_impact/cascade.py`,
  `merge_decision_eval.py` (`summarize` → `by_criterion`).
- Charts actuales: `studies/fusion_metrics/viz/charts.py`,
  `dashboards.py` (scene dashboard).
- Rama del OR en producción: `ovo/entities/fusion/criteria.py`
  (`PointOverlapCriterion`, `PointOverlapOldCriterion` → `accept_mode`).
