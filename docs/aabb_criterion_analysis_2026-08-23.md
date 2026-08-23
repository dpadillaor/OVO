# Criterio AABB en la fusión clásica: hallazgo y propuesta (2026-08-23)

**Rama:** `study/contest-realtime-online`.
**Estado:** análisis cerrado, **código NO tocado**. Documento de continuidad.
**Origen:** sesión de inspección de pares no fusionados en `db1` (ScanNet scene0231_00, 2x2 BAL x contest).

Lectura relacionada: `docs/instance_fusion_strategy.md`, `docs/optimizations.md`.

---

## 0. Resumen

Tres hallazgos, en orden de importancia: una **geometría caduca dentro de cada pasada de fusión**
que afecta al centroide hoy (§5ter), un **precómputo que gasta el 94% del tiempo en la máscara
booleana** y no en las reducciones (§4), y un **criterio `aabb` implementado y desconectado** (§2).

El criterio `aabb` está implementado, tiene umbral configurado (`th_aabb: 0.3`) y **no se usa**:
no aparece en ninguna cadena por defecto de `_DEFAULT_CHAINS`. En su lugar la cadena de `clip`
filtra por distancia entre centroides, que es justamente el criterio que el docstring del AABB
señala como menos robusto para objetos alargados.

Al medir el coste para justificar si compensa activarlo apareció un segundo hallazgo, mayor: el
precómputo de la fusión gasta el **94% del tiempo en extraer las nubes por instancia**, no en
calcular centroides ni cajas. Reorganizándolo con reducciones dispersas, añadir el AABB sale a
**coste neto negativo**.

El AABB, además, es un estadístico de orden extremo y no es robusto a outliers como sí lo es el
centroide, así que la comparación justa es contra una caja robusta (§5bis).

El traspaso contest -> fusión de LC está verificado y es **correcto** (§5quater), pero al mirarlo
apareció que la fusión clásica **deshace en torno al 10% de los splits del contest**.

**Ninguno de los hallazgos está validado sobre métricas de mapa.** Ver §6.

---

## 1. Qué hace el criterio AABB

`compute_aabb_distance` (`ovo/utils/instance_utils.py`) devuelve la **distancia euclídea mínima
entre las dos cajas alineadas con los ejes**, tratadas como sólidos:

```python
min1, max1 = points1.min(0), points1.max(0)
min2, max2 = points2.min(0), points2.max(0)
gap = clamp(maximum(min1 - max2, min2 - max1), min=0)
return gap.norm()
```

Eje por eje calcula el hueco entre las cajas; el `clamp` anula los ejes en que se solapan. La norma
del vector de huecos es la distancia real, no una aproximación: como las cajas están alineadas con
los ejes, los tres huecos son ortogonales e independientes.

**Se satura en 0** cuando las cajas se cruzan. Eso es correcto (la distancia mínima entre dos
sólidos que se intersecan es cero) y es gratis para su uso actual, porque el criterio solo compara
contra un umbral. Pero implica que **el AABB nunca puede ser más que una puerta**: no distingue
"se rozan" de "una contiene a la otra".

En `criteria.py` es un **criterio de veto puro**: solo puede rechazar (`dist > th` → `False`), y si
pasa devuelve `None`, que `FusionStrategy` interpreta como "no me pronuncio" y sigue la cadena.
Nunca acepta una fusión por sí mismo.

## 2. El hallazgo: está desconectado

```python
# ovo/entities/fusion/factory.py
_DEFAULT_CHAINS = {
    "clip":  ["centroid", "cos_sim", "overlap_old"],
    "dino":  ["centroid", "cos_sim", "overlap_old"],
    "pe":    ["centroid", "cos_sim", "overlap_old"],
    "sam3":  ["centroid", "cos_sim", "overlap_old"],
}
```

`aabb` no aparece. Solo entra si se sobreescribe `fusion_criteria` en el config, y en las 40
tiradas del 2x2 de scene0231_00 ese campo va a `null`.

### Los dos filtros no están anidados: seleccionan poblaciones distintas

Medido sobre el mapa final de `db1` (389 instancias, 75 466 pares):

| | pares que pasan |
|---|--:|
| centroide ≤ 1,5 (puerta actual) | 7 756 (10,3%) |
| aabb ≤ 0,3 (puerta AABB) | 5 383 (7,1%) |
| pasan centroide pero **no** aabb | 4 053 |
| pasan aabb pero **no** centroide | **1 680** |

Esos 1 680 pares son nubes que se tocan con centroides lejanos: objetos grandes y alargados. La
cadena actual **no los examina nunca**.

### Caso testigo: par 66-117 en `db1`

Dos paredes, ambas de 2,34 m de alto, de suelo a techo.

```
frame 1601  REJECTED  centroid  dist=2.9021
frame 2041  REJECTED  centroid  dist=2.3065
frame 4371  REJECTED  centroid  dist=2.3331
frame 4406  REJECTED  centroid  dist=2.3503
```

Rechazadas ocho veces por el primer criterio, con las columnas `cos_sim` y `p_dist` vacías porque
la cadena corta ahí. Pero:

- **distancia mínima entre las nubes: 1 mm.** Se tocan. Su `aabb_dist` sería 0,0.
- `cos_sim` = **0,8955**, muy por encima del umbral de 0,81. Habría pasado el segundo criterio.
- `overlap_old` rama B pide `cos_sim > 0.9` y `p_dist > 0.2`: el solape es 0,264 (cumple) y el
  coseno **falla por 0,0045**.

O sea que ese par habría muerto igual, pero por cuatro milésimas en el último criterio y no por dos
metros de centroide en el primero. Con la cadena actual eso no se puede ni saber: la puerta se
cierra antes y las otras dos señales, una a favor y otra casi, no se consultan.

### Distribución de `aabb_dist`

De los 5 383 pares que pasarían la puerta, **2 769 están empatados a 0,0 exacto** (cajas cruzadas).
Más de la mitad. Confirma la sección 1: como puerta funciona, como señal ordenable no sirve.

## 3. Coste: lo que parecía

| | por par | 75 466 pares |
|---|--:|--:|
| centroide | 3,70 µs | 279 ms |
| aabb, tal como está | 34,31 µs | **2 590 ms** |
| aabb, con cajas precomputadas | 5,95 µs | 449 ms |

### Las dos escalas del precómputo (no confundirlas)

```
PASADA DE FUSIÓN (8 veces por escena)
│
├─ 1. contest: mueve puntos en points_ins_ids (in situ)
├─ 2. bucle de precómputo        <- aquí nacen los centroides, 389 iteraciones
└─ 3. doble bucle de comparación <- 75 466 iteraciones, reutilizan lo del paso 2
```

- **Dentro de una pasada**: el centroide se calcula una vez y se reutiliza 75 466 veces. Por eso
  comparar cuesta 3,70 µs: solo una resta de dos vectores de 3 números. El AABB no está en el paso
  2, así que cada comparación recalcula las cajas recorriendo las nubes: 34,31 µs.
- **Entre pasadas**: `obj_pcds` es local, nace al entrar en `_fuse_overlapping_instances` y muere al
  salir. No persiste. `Instance3D` **no guarda centroide ni caja**, así que no hay nada que pueda
  quedarse obsoleto entre pasadas.

Las dos afirmaciones ("llega precomputado" y "siempre está fresco") no se contradicen: es barato
dentro de la pasada precisamente porque se calcula una sola vez al principio de ella, y ese
principio cae después del contest.

El 9x aparente es artificial. El centroide se precomputa una vez por instancia en el bucle de
`_fuse_overlapping_instances` (`ovo/entities/ovo.py:825`):

```python
for instance in objects_list:
    obj_pcd = points_3d[points_ins_ids == instance.id]
    obj_pcds[instance.id] = [obj_pcd, obj_pcd.mean(axis=0)]
```

Las cajas **no**, así que `compute_aabb_distance` las recalcula en cada par. La instancia 73
(108 342 puntos) aparece en 388 pares: su caja se recalcula 388 veces dando siempre lo mismo.

Precomputadas, la diferencia real es 1,6x, con la misma complejidad O(1) por par: el AABB hace tres
operaciones más sobre vectores de 3 elementos.

## 4. Coste: lo que realmente pasa

Desglose del precómputo de las 389 instancias:

```
extraer las nubes (máscara booleana)      495 ms   (94%)
mean + min + max sobre las nubes           29 ms   ( 6%)
```

`points_ins_ids == instance.id` recorre la nube **entera** (6,2 M de puntos) una vez por instancia.
389 veces. Son ~2 400 millones de comparaciones para repartir 3,2 M de puntos asignados.

La discusión de si el AABB cuesta 4 ms más que el centroide es ruido al lado de eso.

### Reducciones dispersas: una pasada para todo

```python
k   = remap[obj_ids]                                        # id de instancia -> fila
cnt = zeros(K).index_add_(0, k, ones)
s   = zeros(K, 3).index_add_(0, k, xyz)
mn  = full((K,3), +inf).scatter_reduce_(0, k, xyz, reduce='amin')
mx  = full((K,3), -inf).scatter_reduce_(0, k, xyz, reduce='amax')
centroides = s / cnt[:, None]
```

```
hoy (máscara por instancia)     486 ms
scatter                          38 ms      13x más rápido
```

Verificado: `mn` y `mx` salen **idénticos** (desviación 0,00e+00). Los centroides difieren en
4,8e-04, ruido de coma flotante por el orden de las sumas, irrelevante frente a un umbral de 1,5 m.

**Añadir el AABB sale a coste negativo**: pagas ~450 ms menos por pasada de fusión y encima tienes
la caja.

Nota: `torch.aminmax`, que debería sacar min y max en una pasada, mide **peor** (12,8 ms contra
8,7 ms) en CPU con nubes de este tamaño. No usarlo.

---

## 5. Propuesta

Tres cambios independientes, en orden de riesgo creciente. Los dos primeros no alteran ninguna
decisión de fusión; el tercero sí.

### 5.1. Precómputo por reducción dispersa (sin cambio de comportamiento)

Sustituir el bucle de `_fuse_overlapping_instances` por una función dedicada. **No dejar el scatter
escrito a pelo dentro de `ovo.py`**: es geometría de nubes, y su sitio es `ovo/utils/instance_utils.py`,
junto a `compute_centroid_distance` y `compute_aabb_distance`.

```python
# ovo/utils/instance_utils.py
def instance_geometry(points_3d, points_ins_ids, instance_ids):
    """Centroide y AABB de cada instancia, en pasadas sobre la nube completa."""
    ...
    return {iid: InstanceGeometry(points, centroid, aabb_min, aabb_max) for ...}
```

`InstanceGeometry` como `NamedTuple`, siguiendo el patrón de `_KFEntry` en `ovo/slam/orbslam2.py`.
Sustituye a la lista `[obj_pcd, obj_pcd.mean(axis=0)]`, que hoy es una lista posicional sin nombres.

Riesgo: bajo. Los mínimos y máximos son exactos; los centroides cambian en 4,8e-04. Ese cambio es
suficiente para alterar alguna decisión limítrofe en un umbral de 1,5 m, así que **no es un
refactor bit a bit** y hay que verificarlo sobre una tirada completa antes de darlo por bueno.

### 5.2. `AabbDistanceCriterion` consume la caja precomputada

Hoy `check` recibe `(i1, i2, p1, c1, p2, c2, ctx)`: nubes y centroides. Con 5.1, la firma pasa a
recibir el `InstanceGeometry` completo y `compute_aabb_distance` toma cajas en vez de nubes:

```python
def compute_aabb_distance(min1, max1, min2, max2) -> float:
    gap = torch.clamp(torch.maximum(min1 - max2, min2 - max1), min=0)
    return gap.norm().item()
```

Es un cambio de firma que toca los siete criterios. Alternativa menos invasiva: dejar la firma y
que el criterio lea la caja de `ctx`, que ya se usa para pasar `cos_sim` y `centroid_dist` entre
criterios. Menos limpio pero cero fricción.

Riesgo: nulo. Mismo resultado, sin recálculo.

### 5.3. Meter `aabb` en la cadena (cambia decisiones)

```python
"clip": ["centroid", "aabb", "cos_sim", "overlap_old"],
```

**No** sustituir el centroide por el AABB: son puertas distintas y el centroide sigue descartando
4 053 pares que el AABB dejaría pasar. Ponerlos en serie los hace más restrictivos, no menos, así
que esta variante **no recupera los 1 680 pares** del caso 66-117.

Para recuperarlos hace falta que sean alternativos, no acumulativos:

```python
"clip": ["proximity", "cos_sim", "overlap_old"]   # proximity = centroide <= 1.5 OR aabb <= 0.3
```

Eso amplía la puerta de 7 756 a 9 436 pares (+22% de trabajo para el coseno y el overlap, que son
los criterios caros). Es la variante que hay que medir.

Riesgo: **alto**. Cambia qué pares se examinan y por tanto el mapa. Requiere las 10 tiradas de la
celda y comparar contra la baseline, no una inspección visual.

---

## 5bis. Robustez frente a outliers (anotado, sin medir)

El centroide es una **media**: un punto espurio se diluye entre N y desplaza el resultado en 1/N.
La caja son **mínimo y máximo**, estadísticos de orden extremo: un solo punto la mueve tanto como
esté de lejos, sin diluirse. Con la instancia 73 (108 342 puntos), un punto suelto a 10 m mueve el
centroide 0,09 mm y estira la caja **10 metros**.

O sea que la comparación honesta no es "centroide contra AABB", sino **"centroide contra caja
robusta"**: enfrentar un estadístico robusto contra uno que no lo es le da al AABB una desventaja
que no le corresponde, y le da una ventaja engañosa en el coste (min/max son más baratos que
cualquier alternativa robusta).

Opciones, de mejor a peor encaje con el scatter de §4:

1. **Recorte por sigma en dos pasadas de scatter** (recomendada). La pasada 1 saca media y varianza
   por instancia (la media ya la necesitas para el centroide); se construye una máscara de puntos
   dentro de `K_SIGMA` desviaciones de su propia instancia; la pasada 2 saca `amin`/`amax` solo
   sobre los supervivientes. Todo vectorizado, sigue siendo O(N), conserva la ganancia de §4.
   Pega: la desviación típica también la inflan los outliers, así que con un outlier muy extremo el
   recorte se afloja. Se mitiga iterando dos veces, o con mediana y MAD a costa de perder el scatter.
2. **Caja por percentiles** (p1/p99 por eje). Robusta por construcción, pero `scatter_reduce_` no
   soporta cuantiles: vuelves a operar instancia por instancia y pierdes el 13x. Además `torch.quantile`
   ordena, O(N log N).
3. **K-ésimo extremo** (`torch.kthvalue`, el 5º menor y el 5º mayor). Exacto, O(N), inmune a 4
   outliers por lado, conceptualmente idéntico al min/max. Pero es por instancia: adiós al scatter.
4. **Limpiar la nube, no la caja** (`remove_statistical_outlier`). El más interesante a medio plazo
   porque ataca el origen: esos puntos ya están inflando el `p_dist` del overlap y metiendo ruido en
   la evidencia del contest. La caja solo lo hace visible por ser un estadístico extremo. Coste alto
   y cambia el mapa, así que es una decisión mucho mayor que tocar un criterio de fusión.

**No medido:** cuántas instancias tienen outliers ni de qué magnitud. No se sabe si el problema es
teórico o real en estos mapas. Barato de comprobar (comparar la caja de min/max contra la de p1/p99
por instancia).

---

## 5ter. Geometría caduca dentro de una pasada de fusión (fallo vivo, independiente del AABB)

### Entre pasadas: cubierto

`obj_pcds` es local a `_fuse_overlapping_instances` y se reconstruye entera en cada llamada. Lo que
hagan el contest (merges, splits) o el refresh de BAL entre pasadas se recoge solo: la pasada
siguiente recalcula desde los `points_ins_ids` actualizados. El AABB heredaría esta propiedad gratis.

### Dentro de una pasada: NO se actualiza

```python
elif self.fusion_strategy.same_instance(
        instance1, instance2, obj_pcds[instance1.id], obj_pcds[instance2.id]):
    instance1, points_ins_ids = instance_utils.fuse_instances(instance1, instance2, map_data)
```

`fuse_instances` reasigna `points_ins_ids` (la superviviente ya tiene más puntos) pero **no toca
`obj_pcds[instance1.id]`**. El bucle interior sigue comparando `instance1` contra el resto con su
nube y su centroide de antes de absorber nada.

Medido en `db1`:

```
frame 1601   19 131 comparaciones    4 747 con nube caducada  (24,8%)
frame 4371  128 987 comparaciones   21 332 con nube caducada  (16,5%)
...
TOTAL: 34 032 de 431 180 comparaciones (7,9%)
```

49 de los 289 merges aceptados son encadenados: la superviviente ya había absorbido algo cuando se
decidió.

**Afecta al centroide hoy**, no es deuda que traiga el AABB. Pero el AABB lo agravaría: absorber una
instancia solo puede agrandar la caja, nunca encogerla, así que el error de una caja caduca es
siempre por defecto y siempre en la dirección de rechazar fusiones que deberían pasar. El centroide
se mueve en la dirección que toque y solo un poco, proporcional a la fracción de puntos nuevos.

### Arreglo

Con el `InstanceGeometry` de §5.1, es recomputar la geometría de la superviviente tras
`fuse_instances` y volver a meterla en el diccionario. Sobre la nube ya fusionada: un `mean`/`min`/
`max` de una sola instancia, no un scatter global. 289 recómputos por escena, ~25 µs cada uno.
Despreciable.

**Cambia el mapa.** Esas 34 032 comparaciones darán otro resultado, casi siempre más permisivo. No es
un refactor neutro: medir con las 10 tiradas de la celda.


---

## 5quater. Traspaso contest -> fusión de LC (verificado, correcto)

Pregunta: en un loop closure el contest corre **antes** que la fusión clásica. ¿Llega la fusión con
la geometría actualizada, o compara sobre el estado previo al contest?

**Llega actualizada.** Dos razones independientes:

1. `Instance3D` no almacena centroide ni caja. No existe ningún valor cacheado que pueda desfasarse.
2. El contest escribe **in situ** sobre `map_data[2]`, el mismo tensor que la fusión lee después:

```python
points_3d, points_ids_all, points_ins_ids = map_data          # points_ins_ids ES map_data[2]
points_ins_ids, contest_fused, _ = self._apply_contest_merges(...)   # muta in situ
...
self._fuse_overlapping_instances(objects_list, points_3d, map_data) # lee map_data[2], ya mutado
```

Merges (`points_ins_ids[points_ins_ids == instance2.id] = instance1.id`, dentro de `fuse_instances`)
y splits (`points_ins_ids[torch.isin(...)] = v.challenger`) son asignaciones enmascaradas in situ,
no rebindings. Y `objects_list` se rehace desde `self.objects` excluyendo `contest_fused`.

### Verificación empírica (`dcb3`)

```
defenders absorbidos por el contest: 34
  de esos, con puntos en el mapa final: 0
ids con puntos pero sin objeto (huérfanos): 0
objetos sin ningún punto: 0
```

Ni zombis ni puntos huérfanos. El traspaso es consistente.

### Hallazgo colateral: la fusión clásica deshace ~10% de los splits del contest

| tirada | pares separados por el contest | vueltos a unir por la fusión clásica |
|---|--:|--:|
| dcb3 | 163 | 18 (11,0%) |
| dcb1 | 151 | 14 (9,3%) |
| dc1  | 160 | 19 (11,9%) |

Los merges del contest **nunca** coinciden con los de la clásica (0 en las tres tiradas): la
colisión es exclusiva de los splits.

Mecanismo probable: el split hace *más* fusionable el par que acaba de separar. Al pasarle un trozo
de A a B, los dos comparten geometría y el descriptor de B se desplaza hacia A, así que el par pasa
los criterios que antes no pasaba. Cuanto mejor trabaja el contest, más fácil se lo pone a la
clásica para deshacerlo.

Esto **no es un problema de frescura**: la geometría está fresca, es un choque de criterios. Es
candidato a explicar parte del solapamiento BAL x contest observado en el 2x2 de scene0231_00
(cada mecanismo sube el eje aware solo cuando el otro no está).

**Cautelas:** la coincidencia de pares no prueba orden temporal, solo que el mismo par aparece en
los dos sitios; para afirmar "deshecho" hay que confirmar que la fusión es posterior al split en la
misma pasada, cruzando el número de report con el frame. Son 3 tiradas de 20.

---

## 6. Lo que NO está demostrado

- **Que activar el AABB mejore las métricas.** Todo lo anterior demuestra que hay 1 680 pares que
  la cadena no mira y que al menos uno de ellos (66-117) es geométricamente el mismo objeto. No
  demuestra que fusionarlos suba mIoU ni AP. El GT de scene0231_00 tiene 86 instancias y `db1`
  predice 341: hay sobre-segmentación de 4x, así que fusionar de más es un riesgo real.
- **Que el caso 66-117 sea representativo.** Es un par mirado a mano entre seis. No hay un cruce
  contra el GT que diga cuántos de los 1 680 son fusiones correctas y cuántas serían errores. Ese
  cruce es el siguiente paso lógico y está a medio montar (las máscaras predichas son RLE sobre los
  419 537 vértices del mesh GT, así que el cruce puede ser exacto).
- **Que el ahorro de 450 ms se mantenga durante la secuencia.** Está medido sobre el mapa final,
  con 389 instancias y 6,2 M de puntos. Las pasadas de fusión tempranas trabajan con nubes mucho
  más pequeñas: la forma del problema no cambia, el factor sí.
- **Que el coste importe.** La fusión corre 8 veces en toda la escena. Incluso sin arreglar nada,
  el AABB añadiría ~21 s sobre una secuencia de 7 minutos. Esto es una mejora de higiene, no de
  rendimiento.

## 7. Reproducir las medidas

Scripts de la sesión (fuera del repo, en el scratchpad):

- `bench.py` coste por par: centroide, aabb actual, aabb precomputado
- `prep.py` coste del precómputo: `mean` contra `min`+`max`
- `onepass.py` máscara por instancia contra scatter, con verificación de equivalencia
- `aabb.py` distribución de `aabb_dist` y solape de puertas contra el centroide

Todos leen `data/output/ScanNet/20260822_orbslam2_CLIP_scannet0231-base_db1/scene0231_00/ovo_map.ckpt`,
que **ya no existe si esa tirada se ha relanzado**. Cualquier ckpt de la celda `db` sirve; los
números cambiarán en el detalle, no en el orden de magnitud.
