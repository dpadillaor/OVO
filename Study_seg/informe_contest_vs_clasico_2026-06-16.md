# Informe — Contest-only vs fusión clásica, y por qué combinarlos hace daño

> Fecha: 2026-06-16. Rama: `study/point-instance-assignment`.
> Sesión de evaluación: mecanismo de contest aislado vs fusión clásica, efecto de
> combinarlos (`contest_fusion: both`), impacto del guard de color, y propuesta de
> métrica de separación de superficie para over-merge de misma clase.
> Lee primero `HANDOFF_contest.md` (mecanismo) y `flujo_decision_contest.puml`.

---

## 0. Resumen ejecutivo

1. **El contest gana con drift suave, pero pierde con drift fuerte (crossover).**
   - Light (J2-**T0.03**): contest-only AP_agnostic **0.206 vs 0.154** = **+34%**.
   - Aggressive (J2-**T0.3**, forward-axis): contest-only **0.074 vs 0.095** = **−22%**.
   - mIoU/mAcc planos en ambos. El cruce sugiere que el contest depende de geometría
     intacta; con saltos de 30 cm la contención/persistencia se corrompen y su lógica
     instance-aware se vuelve ruido, mientras la clásica (cos_sim, geometry-light)
     degrada más suave.
2. **Combinar mecanismo + clásica (`both`) hace daño.** En office3 limpio, `both` con
   `[aabb, cos_sim, overlap]` cae a 0.184 vs 0.222 de contest-only (−0.038). La pasada
   clásica re-fusiona adyacencias que el contest mantuvo separadas + crea hubs.
3. **El guard de color no mueve AP.** Bloquea 23–27 splits (verdicts cambian) pero
   AP_agnostic idéntico (0.192). Los splits que mata son AP-neutrales.
4. **aabb funciona pero es ciego a la adyacencia.** Filtra el 94% de pares (lejanos),
   pero da 0 en objetos pegados/contenidos → ahí solo deciden cos+overlap → over-merge.
5. **Métrica propuesta para "misma clase, cerca pero sin tocar": separación de
   superficie** (cross-NN punto-a-punto a umbral apretado), no bbox. Coste ~0 (ya se
   calcula dentro de `compute_pcd_overlap`).

---

## 1. Contest-only vs clásico bajo jump-drift LIGHT (T0.03, 8 escenas)

Config idéntica salvo fusión. Jump-drift `J2-T0p03-R0p0` (2 saltos, kf 50 y 100, 3 cm).
- contest: `contest_fusion: only`, `contest_split_mode: all`, `reeval_frontier: true`.
- clásico: `fusion_criteria: [centroid, cos_sim, overlap]`.

Runs: `20260616_GTJump-J2-T0p03-R0p0_CLIP_contest-only-light-pia_6aed7` vs
`20260613_GTJump-J2-T0p03-R0p0_CLIP_light-baseline_82bfd` (a este se le corrió
`eval_instances` a posteriori reusando sus máscaras).

### Instance AP_agnostic (agregado)

| métrica | contest | clásico | Δ |
|---|---|---|---|
| AP      | 0.206 | 0.154 | **+0.052 (+34%)** |
| AP_50   | 0.369 | 0.307 | +0.062 |
| AP_25   | 0.572 | 0.512 | +0.060 |

### AP_agnostic por escena

| escena | contest | clásico | Δ |
|---|---|---|---|
| office0 | 0.127 | 0.128 | −0.001 |
| office1 | 0.182 | 0.173 | +0.009 |
| office2 | 0.196 | 0.091 | **+0.105 (+115%)** |
| office3 | 0.209 | 0.170 | +0.039 |
| office4 | 0.282 | 0.160 | **+0.122 (+76%)** |
| room0   | 0.259 | 0.183 | +0.076 |
| room1   | 0.301 | 0.227 | +0.074 |
| room2   | 0.119 | 0.127 | −0.008 |

6/8 escenas suben, 2 planas. Bestiales office2 y office4.

### Semántica mIoU / mAcc (media por escena, ignorando clases ausentes)

| escena | cont mIoU | cls mIoU | ΔIoU | cont mAcc | cls mAcc | ΔAcc |
|---|---|---|---|---|---|---|
| office0 | 0.264 | 0.290 | −0.026 | 0.351 | 0.389 | −0.038 |
| office1 | 0.172 | 0.187 | −0.016 | 0.328 | 0.351 | −0.023 |
| office2 | 0.261 | 0.256 | +0.005 | 0.323 | 0.319 | +0.004 |
| office3 | 0.262 | 0.256 | +0.006 | 0.325 | 0.322 | +0.003 |
| office4 | 0.462 | 0.459 | +0.004 | 0.594 | 0.590 | +0.004 |
| room0   | 0.340 | 0.319 | +0.021 | 0.405 | 0.392 | +0.013 |
| room1   | 0.370 | 0.407 | −0.037 | 0.511 | 0.555 | −0.044 |
| room2   | 0.237 | 0.275 | −0.039 | 0.322 | 0.371 | −0.049 |
| **media** | **0.296** | **0.306** | **−0.010** | **0.395** | **0.411** | **−0.016** |

(Agregado 51-clases con 0 para ausentes: mIoU 0.272 vs 0.266 +0.006; mAcc 0.401 vs
0.400 plano. El signo difiere según convención de media — por-escena vs todas-clases.)

**Lectura:** el contest cambia **fronteras de instancia** (mueve AP), no la etiqueta de
clase por-punto (mIoU ~plano). Bajo drift la clásica empeora porque overlap/centroid
fusionan instancias desplazadas por el salto; el contest (contención + persistencia +
disputa por punto) resiste el desalineo. Consistente con hallazgos previos
(mecanismo/cooc sube AP_agnostic, mIoU plano).

---

## 1b. Contest-only vs clásico bajo jump-drift AGGRESSIVE (T0.3, 8 escenas)

Misma comparación, drift 10× más fuerte: `J2-T0p3-R0p0`, saltos de 30 cm sobre el eje
forward (`on_forward_axis: true`). Runs:
`20260616_GTJump-J2-T0p3-R0p0_CLIP_contest-only-aggressive-pia_aecde` vs
`20260613_GTJump-J2-T0p3-R0p0_CLIP_agressive-baseline_5f2f1` (eval_instances a posteriori).

### Instance AP_agnostic (agregado)

| métrica | contest | clásico | Δ |
|---|---|---|---|
| AP      | 0.074 | 0.095 | **−0.021 (−22%)** |
| AP_50   | 0.172 | 0.213 | −0.041 |
| AP_25   | 0.409 | 0.435 | −0.026 |

### AP_agnostic por escena

| escena | contest | clásico | Δ |
|---|---|---|---|
| office0 | 0.019 | 0.021 | −0.002 |
| office1 | 0.049 | 0.081 | −0.032 |
| office2 | 0.118 | 0.084 | +0.034 |
| office3 | 0.087 | 0.124 | −0.037 |
| office4 | 0.122 | 0.163 | −0.041 |
| room0   | 0.074 | 0.101 | −0.027 |
| room1   | 0.079 | 0.113 | −0.034 |
| room2   | 0.037 | 0.056 | −0.019 |

Contest peor en 6/8 escenas (solo office2 sube, office0 plano).

### Semántica mIoU / mAcc (media por escena)

| escena | cont mIoU | cls mIoU | ΔIoU | cont mAcc | cls mAcc | ΔAcc |
|---|---|---|---|---|---|---|
| office0 | 0.176 | 0.195 | −0.019 | 0.254 | 0.284 | −0.030 |
| office1 | 0.181 | 0.182 | −0.001 | 0.315 | 0.334 | −0.019 |
| office2 | 0.193 | 0.260 | −0.067 | 0.260 | 0.325 | −0.065 |
| office3 | 0.247 | 0.239 | +0.007 | 0.316 | 0.311 | +0.006 |
| office4 | 0.430 | 0.420 | +0.010 | 0.551 | 0.557 | −0.005 |
| room0   | 0.302 | 0.230 | +0.072 | 0.372 | 0.293 | +0.079 |
| room1   | 0.341 | 0.313 | +0.028 | 0.467 | 0.449 | +0.017 |
| room2   | 0.233 | 0.270 | −0.036 | 0.312 | 0.335 | −0.023 |
| **media** | **0.263** | **0.264** | **−0.001** | **0.356** | **0.361** | **−0.005** |

### Crossover light↔aggressive

| drift | contest AP_ag | clásico AP_ag | Δ |
|---|---|---|---|
| Light (T0.03)      | 0.206 | 0.154 | **+34%** |
| Aggressive (T0.3)  | 0.074 | 0.095 | **−22%** |

**Lectura:** el contest se apoya en geometría por-punto (contención, persistencia,
disputa). Con drift suave la geometría sigue mayormente intacta → su lógica
instance-aware separa/fusiona bien y gana. Con drift fuerte (30 cm) los puntos caen
lejos de su posición real → contención/persistencia se vuelven ruido → el contest
decide mal. La clásica, más apoyada en el descriptor (cos_sim, poco dependiente de
geometría fina), degrada más suave y acaba por encima. El AP absoluto cae para ambos
(0.206→0.074 contest, 0.154→0.095 clásico): el drift fuerte rompe el mapa en general.

---

## 2. `contest_fusion: both` hace daño (office3 limpio)

Run `20260616_GT_CLIP_contest-both-aabb-cos-overlap-pia_fe2a8`:
`contest_fusion: both` (contest mergea primero → clásica `[aabb, cos_sim, overlap]` sobre
el resto), `contest_split_mode: all`, `reeval_frontier: true`.

| run (office3) | AP_ag | AP_50 | AP_25 |
|---|---|---|---|
| **both [aabb,cos,overlap]** | **0.184** | 0.351 | 0.611 |
| contest-only (mecanismo) `8bedf` | 0.222 | 0.382 | 0.612 |
| classic-nocooc [centroid,cos,overlap] `2628c` | 0.182 | 0.357 | 0.616 |

`both` cae al nivel de la clásica sola. La pasada clásica acepta **51 merges extra** sobre
el mapa ya resuelto por el contest:
- Re-fusiona 3 de las 4 adyacencias que el contest había separado (36→38, 173→175,
  159↔168), todas cos≈0.9: CLIP no distingue instancias de la misma clase.
- Crea hubs: la **mesa 103** se traga **6 objetos** que tiene encima (109, 131, 132, 134,
  145, 200), todos `p_dist=1.0` (objeto entero dentro del footprint de la mesa) con cos
  bajo ~0.82. La regla de overlap (acepta si overlap>0.5) los traga.

**Conclusión:** el orden contest→clásica no respeta los veredictos del contest. Si se
quisiera combinar, habría que excluir del overlap los pares ya juzgados NO_ACTION/SPLIT,
o invertir el orden (clásica→contest). Tal cual, **contest-only es estrictamente mejor**.

---

## 3. Guard de color: sin impacto en AP

Runs `_00d1e` (color) vs `_9b9ef` (sin color), 8 escenas limpias. AP_agnostic agregado
**idéntico** (0.192 / 0.349 / 0.553), per-escena igual salvo office0 (−0.001).

Pero los verdicts cambian: SPLIT 55→32 (color bloquea 23 splits vía ΔE_W>ΔE_L). Los 27
NO_ACTION por color son **AP-neutrales**: no tocan el matching con GT.

- Casos con señal fuerte (guard acierta): `room1 31→4` (ΔE_W=12.0 vs ΔE_L=0.6) — el chunk
  disputado tiene el color del perdedor 31, no del ganador 4 → no fusionar. Confirma
  adyacencia (cont=0.125, pers=0.142): dos objetos misma clase que se rozan.
- Casos borde (guard dispara con poca señal): ΔE_W≈ΔE_L (hasta empate 0.8/0.8, 2.9/2.9).
  Ruido — sim alto, ΔE indistinguible.

El guard reduce over-split sin perder AP, pero no lo mejora.

---

## 4. aabb funciona, pero es ciego a la adyacencia

En el run `both`, `aabb` es el filtro dominante:

| reason | REJECTED |
|---|---|
| aabb | 6875 |
| cos_sim | 428 |
| overlap | 321 |

`aabb_dist` poblado y correcto (rechaza pares con bbox separada > th=0.3 m). Pero da
**0** cuando las bboxes se solapan, aunque las nubes no se toquen. Ej. par `127↔130`
(misma clase): bboxes solapan en los 3 ejes → `aabb_dist=0.000` → pasa trivialmente.
La decisión recae en cos (0.94) + overlap → over-merge.

---

## 5. Métrica propuesta: separación de superficie (misma clase, cerca sin tocar)

Caso `127↔130` (misma clase). Comprobación cross-NN punto-a-punto:

```
resolución (spacing intra-nube): 0.4 cm
min cross dist:                  0.3 cm   (1 punto suelto, ruido/borde)
pct1 cross:                      2.1 cm
pct5 cross:                      4.4 cm
pct50 cross:                    22.4 cm
frac de puntos a <1 cm:          0.3 %
frac de puntos a <3 cm:          2.1 %
```

= **separados por aire**, rozándose en un punto. Contacto real daría una banda de puntos
a ~resolución (0.4 cm) a lo largo de la costura; aquí 0.2%.

### Propuesta

Reemplazar/endurecer la señal de overlap por una **contact fraction con umbral atado a la
resolución** (no a bbox ni centroide):

```
contact = frac(puntos de la nube pequeña a < τ de la grande),  τ ≈ 2–3× spacing (~1–1.5 cm)
```

- contact ≈ 0 → separados → NO merge (este caso: 0.3%).
- contact alto, banda fina → fragmento del mismo objeto → merge.
- contact medio en la costura → objetos pegados → lo decide el seam-normal.

Alternativa robusta de un solo número: **pct1 de la cross-NN dist** (el min es sensible a
ruido). Regla: `pct1_cross > k·spacing (k~4, ~1.5 cm)` → instancias distintas.

**Por qué el overlap actual falla:** `th_points = 0.1 m` (10 cm) es enorme; a 10 cm dos
objetos a 6 cm cuentan como solapados, y con cos>0.9 la regla los acepta.

### Coste: ~0

`compute_pcd_overlap` (`ovo/utils/instance_utils.py`) ya calcula el array de distancias:

```python
dists = small.compute_point_cloud_distance(big)   # cross-NN, ya se hace
return (dists < th_points).astype(float).mean()
```

La parte cara (kd-tree query) ya ocurre y solo sobre pares que pasan aabb (~370, no 7600).
La contact-fraction-apretada y el pct1 son **reducciones distintas del mismo array** →
coste despreciable. Opciones de implementación:
1. **Tunear**: `th_points` 0.1 → ~0.015. 1 línea de config, cero código.
2. **Criterio `surface_gap`**: reusa `dists`, rechaza si `pct1 > k·spacing`. ~10 líneas,
   más robusto.

---

## 6. Próximos pasos

- Implementar la separación de superficie (opción 1 rápida para validar señal, opción 2
  como criterio definitivo).
- Validar en Rerun los over-merge de la mesa 103 (objetos sobre superficie) y el par
  127↔130.
- Repetir la comparación contest-only vs clásico bajo más niveles de drift para confirmar
  la robustez del +34%.
