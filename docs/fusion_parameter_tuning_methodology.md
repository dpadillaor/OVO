# Metodología para el ajuste de parámetros de fusión de instancias

> Documento de método. Objetivo: definir cómo afrontar de forma rigurosa el ajuste
> de los parámetros que gobiernan la fusión/merge de instancias en OVO, evitando los
> errores metodológicos habituales (sobre todo asumir independencia entre parámetros
> que en realidad están acoplados).

---

## 1. El problema y por qué importa

La decisión de fusionar dos instancias en el mapa semántico la toma una **cadena de
criterios** (`ovo/entities/fusion/criteria.py`), ensamblada en
`FusionStrategy.same_instance` (`strategy.py`) y construida por
`create_fusion_strategy` (`factory.py`). Cada criterio puede:

- devolver `None` → pasa al siguiente criterio,
- devolver `False` → rechaza el merge (corta la cadena),
- devolver `True` → acepta el merge (corta la cadena).

El comportamiento global del sistema (cuántas instancias se fusionan, cuáles, y por
tanto la calidad del mapa semántico final) depende de un puñado de umbrales. Ajustarlos
mal produce dos patologías opuestas:

- **Over-merge** (umbrales demasiado permisivos): objetos distintos colapsan en una
  instancia → se pierde granularidad, baja la calidad de instancia.
- **Under-merge / fragmentación** (umbrales demasiado estrictos): un mismo objeto queda
  partido en varias instancias → ruido, duplicados, peor mIoU.

El ajuste de estos parámetros es, por tanto, una parte central del estudio y **no es un
detalle de implementación**: es un experimento científico con su propia metodología.

### 1.1. La intuición correcta: los parámetros NO son independientes

La tentación natural es el método **OFAT** (*One-Factor-At-A-Time*): fijar todos los
parámetros menos uno, barrer ~100 valores de ese, mirar la métrica, quedarse con el
mejor, y repetir para el siguiente. **Es un anti-patrón conocido en diseño de
experimentos (DOE).** Falla por dos razones:

1. **Asume aditividad / independencia.** OFAT solo encuentra el óptimo global si la
   superficie de respuesta es separable (el efecto de cada parámetro no depende del
   valor de los demás). En cuanto hay **interacciones**, OFAT se queda atrapado en
   óptimos locales que dependen del orden en que barriste y de los valores que dejaste
   fijos.
2. **Explora una fracción ridícula del espacio.** Barrer cada eje por separado recorre
   solo las "cruces" del hipercubo, nunca el interior donde suelen estar los óptimos con
   interacción.

En OVO la interacción **no es hipotética, es estructural**. Pruebas concretas:

- `PointOverlapCriterion.check` (líneas 114-119) decide con:

  ```python
  accepted = p_dist > 0.5 or (cos_sim is not None and cos_sim > 0.9 and p_dist > 0.2)
  ```

  El veredicto final depende **simultáneamente** de `cos_sim` (producido por
  `CosSimilarityCriterion` y guardado en `ctx`) y de `p_dist`. Cambiar `th_cossim`
  cambia *qué pares de instancias llegan vivos hasta `overlap`* y, además, la rama
  `cos_sim > 0.9` interactúa con el valor de overlap. Acoplamiento puro.

- El `ctx` compartido (`cos_sim`, `centroid_dist`, `shared_kfs`, `p_dist`) es
  literalmente un canal de dependencia entre criterios. Está diseñado para que unos
  criterios usen valores calculados por otros.

- El **orden de la cadena** cambia el resultado: como cualquier `False`/`True` corta la
  cadena (early-exit en `strategy.py:30-33`), un criterio que rechaza pronto impide que
  los siguientes lleguen a evaluarse. Mover `cos_sim` antes o después de `centroid`
  cambia qué pares se rechazan y por qué motivo.

Conclusión: hay que tratar el conjunto de parámetros como un **vector θ** y optimizar
sobre la superficie de respuesta conjunta, no eje por eje.

---

## 2. Inventario completo de parámetros (lo que realmente hay que tunear)

Antes de optimizar nada hay que saber qué variables existen. En OVO son de tres tipos:

### 2.1. Umbrales expuestos por config (`factory.py`)

| Parámetro | Default | Criterio | Significado |
|---|---|---|---|
| `cooccurrence_veto_threshold` | `5` | `CooccurrenceCriterion` | Si dos instancias comparten > N keyframes → se asume que son objetos distintos co-visibles → veto al merge. **Entero.** |
| `th_centroid` | `1.5` | `CentroidDistanceCriterion` | Distancia máxima entre centroides para considerar merge (m). |
| `th_aabb` | `0.3` | `AabbDistanceCriterion` | Distancia máxima entre bounding boxes (no está en la cadena por defecto, ver 2.3). |
| `th_cossim` | `0.81` | `CosSimilarityCriterion` | Similitud coseno mínima de features (CLIP/DINO/PE/SAM3) para no rechazar. |
| `th_points` | `0.1` | `PointOverlapCriterion` | Umbral de distancia (m) para contar un punto como "solapado" en `compute_pcd_overlap`. **OJO: no es el umbral de solape, es la tolerancia geométrica punto-a-punto.** |

### 2.2. Constantes hardcoded (parámetros ocultos — hay que exponerlos)

En `PointOverlapCriterion.check` (línea 114) hay **tres hiperparámetros reales sin
exponer**:

```python
accepted = p_dist > 0.5 or (cos_sim is not None and cos_sim > 0.9 and p_dist > 0.2)
```

- `0.5` → ratio de solape suficiente para aceptar por geometría sola.
- `0.9` → similitud semántica "alta" que relaja el requisito de solape.
- `0.2` → solape mínimo cuando la semántica es alta.

**Si el estudio tunea solo los `th_*` del `__init__` e ignora estos tres números, el
estudio está incompleto y es irreproducible** (alguien que cambie esa línea obtiene otro
sistema). Acción previa obligatoria: promoverlos a config (p.ej. `overlap_accept_hi`,
`cossim_relax`, `overlap_accept_lo`) con sus defaults actuales, para que entren en θ.

### 2.3. Parámetros estructurales (categóricos / de diseño)

- **Composición de la cadena** (`fusion_criteria`): qué criterios están activos. Por
  defecto `["cooccurrence", "centroid", "cos_sim", "overlap"]`; `aabb` existe pero no se
  usa. Activar/desactivar criterios es una variable de diseño.
- **Orden de la cadena**: por el early-exit, el orden afecta resultado y coste.
- **`fusion_method`** (`clip` / `dino` / `pe` / `sam3`): selecciona qué feature
  semántica alimenta `cos_sim`. Cambia el significado de `th_cossim` (cada espacio de
  embeddings tiene su propia escala de similitud), así que **`th_cossim` debe re-tunearse
  por cada `fusion_method`** — no es transferible.

> **Implicación clave:** `th_cossim` óptimo para CLIP no sirve para PE o SAM3. Cualquier
> comparación entre métodos semánticos debe tunear `th_cossim` por separado para cada
> uno, o se está comparando con una ventaja arbitraria.

---

## 3. Definir el objetivo antes de optimizar

No se puede optimizar lo que no se ha definido. Hay que fijar una **función objetivo
escalar** (o un Pareto explícito) *antes* de lanzar experimentos. Opciones:

1. **Mono-objetivo:** maximizar mIoU (métrica primaria del proyecto). Simple,
   directo, comparable con los baselines del paper.
2. **Multi-objetivo (Pareto):** maximizar mIoU **y** controlar el número de instancias
   (proxy de over/under-merge). El ajuste de fusión es intrínsecamente un *trade-off*
   entre fusionar de más y de menos; un solo número lo esconde. Un frente de Pareto
   (mIoU vs. nº instancias, o vs. una métrica de pureza/cobertura de instancia) muestra
   el compromiso de forma honesta.

Recomendación: **define mIoU como objetivo primario** para la fase de optimización y
**reporta el Pareto** (mIoU vs. recuento/pureza de instancias) para el análisis y el
paper. Evita los objetivos compuestos ad-hoc tipo `mIoU - λ·n_inst` salvo que puedas
justificar `λ`; suelen ser difíciles de defender ante un revisor.

---

## 4. Pipeline metodológico recomendado

La idea es **no gastar runs caros a lo bruto**. Cada experimento en OVO es una corrida
completa de SLAM + semántica → caro. La estrategia profesional separa **cribado**
(¿qué importa?) de **optimización** (¿qué valores?) y, opcionalmente, **análisis de
sensibilidad** (¿por qué?).

```
0. Exponer todos los parámetros (incl. los hardcoded)        → vector θ bien definido
1. Definir objetivo (mIoU) + métricas secundarias            → función a optimizar
2. Fijar el confounder (ruido SLAM) y el protocolo de escenas
3. Cribado (Morris / factorial fraccionado)                  → reduce dimensión, detecta acoplamiento
4. Optimización bayesiana (Optuna TPE / GP)                  → encuentra θ* eficiente en nº de runs
5. Análisis de sensibilidad global (Sobol, opcional)         → cuantifica interacciones (para el paper)
6. Validación de robustez (multi-escena, multi-seed)         → evita overfitting al benchmark
```

### 4.1. Fase 0-2: preparación (barata pero crítica)

- **Exponer los hardcoded** (sección 2.2).
- **Controlar el confounder del ruido de SLAM.** Tu `ovo.yaml` tiene
  `noise.jump_drift_enabled: true` con jumps fijos y `jump_seed: 42`. El ruido de
  trayectoria interactúa con los umbrales geométricos (`th_centroid`, `th_points`): un
  óptimo encontrado con un perfil de ruido no vale para otro. Decide explícitamente:
  - o **fijas** el ruido (mismo seed/perfil en todos los runs) y reportas "óptimo
    condicionado a este ruido",
  - o **estratificas** (tuneas y validas sobre varios perfiles de ruido) si quieres
    robustez frente a la deriva.
  Documentar esta elección es parte del rigor.
- **Protocolo de escenas:** separa escenas de *ajuste* y de *test* (ver 4.5). No tunees
  y reportes sobre la misma escena.

### 4.2. Fase 3: cribado / screening (¿qué parámetros importan?)

No empieces optimizando los 8 parámetros a la vez. Primero averigua **cuáles mueven la
métrica y cuáles interactúan**, con pocos runs:

- **Método de Morris (elementary effects).** Coste ~ `r·(k+1)` evaluaciones (r
  trayectorias, k parámetros) — del orden de decenas, no cientos. Devuelve para cada
  parámetro:
  - `μ*` → importancia global (cuánto mueve la salida en promedio),
  - `σ` → grado de **no-linealidad / interacción** (σ alto = el efecto del parámetro
    depende de los demás).

  Es exactamente la herramienta que responde a tu sospecha: *te dice qué parámetros
  están acoplados.*

- **Alternativa: diseño factorial fraccionado 2-niveles.** Pones cada parámetro en
  `{bajo, alto}` y, con una fracción bien elegida (resolución IV o V), estimas efectos
  principales **e interacciones de 2º orden** con muy pocos runs. Más clásico y fácil de
  explicar en un paper de ingeniería.

Salida de esta fase: **descartar los parámetros irrelevantes** (fijarlos a su default) y
**quedarte con el subconjunto que importa**, sabiendo además qué pares están acoplados.
Esto reduce la dimensión de θ y hace barata la fase siguiente.

### 4.3. Fase 4: optimización (¿qué valores?)

Sobre el θ reducido, optimiza con un método **eficiente en muestras** (porque cada
evaluación es cara). NO uses grid search ni random search puro: desperdician runs.

- **Optimización bayesiana (BO).** Construye un *surrogate* (Gaussian Process o, en
  Optuna, TPE) de la superficie objetivo y usa una función de adquisición para elegir el
  siguiente θ a probar donde más información/mejora se espera. Converge en **decenas** de
  evaluaciones donde grid necesitaría miles, y **modela las interacciones de forma
  implícita** (el surrogate aprende la superficie conjunta).
- **Herramienta:** [Optuna](https://optuna.org). Razones:
  - sampler TPE por defecto, robusto y sin dependencias pesadas;
  - **pruning** (`MedianPruner`, etc.) para abortar runs malos antes de terminar →
    ahorra cómputo en un dominio donde cada run es largo;
  - paralelización con storage (SQLite/Postgres) → varios runs a la vez;
  - **multi-objetivo nativo** (`NSGAIISampler`, MOTPE) si vas por la vía Pareto;
  - persistencia del *study* → reproducible y reanudable.
- **Alternativa con GP explícito:** Ax/BoTorch si quieres GP + incertidumbre calibrada y
  análisis posterior más formal.

Define rangos/prior sensatos por parámetro (informados por el cribado y por el
significado físico: p.ej. `cos_sim ∈ [0,1]`, `th_centroid` en metros razonables para la
escena). Rangos absurdos malgastan el presupuesto de evaluaciones.

### 4.4. Fase 5: análisis de sensibilidad global (¿por qué? — para entender y publicar)

Si además de *optimizar* quieres *entender y defender* el sistema (lo cual da mucho peso
a un paper), cuantifica las interacciones con **índices de Sobol** (descomposición de la
varianza):

- `S1_i` (first-order): fracción de la varianza de la salida explicada por el parámetro
  *i* actuando solo.
- `ST_i` (total-order): incluye **todas** las interacciones en las que participa *i*.
- La diferencia **`ST_i − S1_i > 0` cuantifica exactamente cuánta de la influencia de
  ese parámetro pasa por interacción con otros.** Es la confirmación numérica y
  publicable de tu intuición de "esto es una cadena, no parámetros independientes".

- **Herramienta:** [SALib](https://salib.readthedocs.io) (Sobol, Morris, FAST). Coste de
  Sobol más alto (escala con el nº de parámetros), por eso se hace **después** del
  cribado, sobre el subconjunto relevante.

### 4.5. Fase 6: validación de robustez (no te engañes a ti mismo)

- **Multi-escena con split.** Un óptimo sobre una sola escena de Replica es casi seguro
  *overfitting al benchmark*. Tunea sobre un conjunto de escenas de *ajuste* y **reporta
  sobre escenas de test no vistas durante el tuning**. Si tienes pocas escenas, usa
  **validación cruzada** (leave-one-scene-out).
- **Variabilidad / seeds.** Si hay cualquier estocasticidad (ruido de SLAM, sampling),
  repite cada configuración con varios seeds y reporta **media ± desviación**. Un pico en
  una sola corrida puede ser azar.
- **Significancia estadística.** Al comparar dos configuraciones (o θ* frente al default),
  usa un test pareado por escena (p.ej. Wilcoxon signed-rank) en vez de comparar dos
  números sueltos. Evita "subió 0.3 mIoU" sin contexto de varianza.
- **Criterio de selección final:** el θ que **generaliza** (buena media en test, baja
  varianza), no el que tiene el pico más alto en una escena de ajuste.

---

## 5. Ideas adicionales específicas de OVO

Más allá del marco genérico, OVO ofrece palancas concretas:

### 5.1. Atribución de rechazos (explota tu logging ya existente)

`FusionStrategy` ya acumula decisiones (`pop_decisions`) con el motivo de cada
rechazo/aceptación (`reason`: `cooccurrence`/`centroid`/`cos_sim`/`overlap`) y los
valores (`cos_sim`, `centroid_dist`, `p_dist`, `shared_kfs`). También registra tiempos
por criterio (`pop_timings`). Antes de optimizar a ciegas:

- **Histograma de motivos de rechazo.** Si el 95% de los rechazos son por `centroid`,
  los demás umbrales apenas importan en ese régimen → te dice por dónde empezar y qué
  rangos explorar.
- **Distribuciones de los valores en `ctx`** (p.ej. el histograma de `cos_sim` de los
  pares que deberían fusionarse vs. los que no, usando el ground-truth de Replica). Eso
  te da **rangos informados** para los umbrales antes de gastar un solo run de
  optimización, y a veces revela directamente el umbral separador.

Esto convierte el ajuste de "fuerza bruta" en "ajuste guiado por datos" y reduce
muchísimo el presupuesto de runs.

### 5.2. Reducir el coste por evaluación

- **Pruning** (Optuna): aborta configuraciones claramente malas a mitad de ejecución.
- **Cribado en subconjunto barato** (menos keyframes / 1-2 escenas) y optimización fina
  en el conjunto completo: *multi-fidelity*. Algoritmos como Hyperband/BOHB explotan
  esto si el coste lo justifica.
- **Cache de etapas deterministas:** si los features semánticos por keyframe no dependen
  de los umbrales de fusión, precómputalos una vez y reúsalos en todas las
  configuraciones (la fusión es solo la etapa final). Esto puede reducir drásticamente el
  coste por evaluación. *Verificar en el código antes de asumirlo.*

### 5.3. El orden y la composición de la cadena como variable

Trata `fusion_criteria` (qué criterios y en qué orden) como un **hiperparámetro
categórico/estructural**, no como algo fijo. Como mínimo:
- prueba mover `cos_sim` antes/después de `centroid`,
- evalúa si añadir `aabb` (hoy inactivo) aporta,
- mide el impacto en coste (early-exit) además de en métrica.

### 5.4. Reparametrización para reducir acoplamiento

Cuando dos umbrales están muy correlacionados, a veces conviene **reparametrizar** a
variables más interpretables/menos acopladas. Ejemplo en `overlap`: en vez de tres
constantes sueltas (`0.5`, `0.9`, `0.2`), modelar la regla como "umbral de solape base +
relajación proporcional a la semántica" puede dar parámetros más ortogonales y más
fáciles de tunear y explicar.

### 5.5. Reproducibilidad

- **Registra θ completo junto a cada experimento** (en el folder de salida
  `{DATE}_{SLAM_CONFIG}_{FUSION}_{LABEL}`), incluido el commit. Sin esto, los resultados
  de tuning no son reproducibles.
- **Versiona el *study* de Optuna** (storage en SQLite dentro de `data/output/` o
  similar) para poder reanudar, auditar y graficar la historia de optimización.

---

## 6. Resumen ejecutivo

1. **OFAT está mal** aquí: los criterios comparten `ctx` y se cortan entre sí
   (early-exit) → hay interacción estructural, no parámetros independientes.
2. **Trata θ como un vector** y optimiza la superficie conjunta.
3. **Expón primero los 3 hiperparámetros hardcoded** del `overlap` o el estudio queda
   incompleto.
4. **Define el objetivo** (mIoU primario; Pareto mIoU vs. nº instancias para análisis).
5. **Controla el confounder del ruido SLAM** y **separa escenas de ajuste/test**.
6. **Pipeline:** cribado (Morris/factorial) → optimización bayesiana (Optuna) →
   sensibilidad global (Sobol, opcional) → validación multi-escena/seed.
7. **Explota el logging de decisiones** ya existente para fijar rangos informados y
   reducir el nº de runs caros.
8. **`th_cossim` no es transferible entre `fusion_method`**: re-tunea por cada espacio de
   features.

---

## 7. Herramientas y referencias

- **Optuna** — optimización bayesiana / TPE, pruning, multi-objetivo: <https://optuna.org>
- **SALib** — Morris, Sobol, FAST (análisis de sensibilidad): <https://salib.readthedocs.io>
- **Ax / BoTorch** — BO con GP explícito y multi-objetivo: <https://ax.dev>
- **scikit-optimize** — BO ligero (alternativa simple): <https://scikit-optimize.github.io>
- Saltelli et al., *Global Sensitivity Analysis: The Primer* (Wiley) — referencia de
  Morris/Sobol.
- Montgomery, *Design and Analysis of Experiments* — DOE clásico (factorial,
  fraccionado, response surface).
- Shahriari et al. (2016), *Taking the Human Out of the Loop: A Review of Bayesian
  Optimization*, Proc. IEEE.

---

## 8. Archivos relevantes en el repo

- `ovo/entities/fusion/criteria.py` — definición de los criterios y la regla de `overlap`
  (incl. constantes hardcoded en línea 114).
- `ovo/entities/fusion/factory.py` — mapeo config → criterios y **defaults** de los
  umbrales.
- `ovo/entities/fusion/strategy.py` — ejecución de la cadena, early-exit, logging de
  decisiones (`pop_decisions`) y tiempos (`pop_timings`).
- `ovo/utils/instance_utils.py` — `compute_pcd_overlap` (semántica real de `th_points`).
- `data/working/configs/ovo*.yaml` — configs de experimento (`fusion_method`, ruido SLAM).
