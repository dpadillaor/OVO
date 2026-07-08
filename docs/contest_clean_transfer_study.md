# Contest — estudio de fallos y gate de transferencia limpia

**Fecha:** 2026-07-06 · **Escena:** Replica/office4 (+ office3) · **Rama:** `study/point-instance-assignment`

Estudio del mecanismo de contest sobre casos reales, análisis por instancias de dónde falla la
heurística, y tres cambios que arreglan los patrones encontrados: `min_count` efectivo, señal de
**exclusividad**, y una última rama de la cascada — **transferencia limpia** — que refina máscaras
mal segmentadas.

---

## 1. Objetivo

Los números del análisis de timing (`docs/contest_timing_analysis.html`) venían de una época en la
que `persistence` estaba mal calculada. Tras arreglarla (denominador = `claims`, acotada en `[0,1]`),
había que **reverificar con datos reales** cómo se comporta la heurística y buscar fallos concretos,
instancia a instancia.

## 2. Herramienta — `studies/contest_metrics/probe/`

Para interrogar la heurística sin re-correr, se construyó un módulo que **replica el `classify`
exacto** (mismo aggregator + discriminator + callbacks geométricos) sobre un `ovo_map.ckpt` guardado:

```bash
python -m studies.contest_metrics.probe --exp <ID> --scene office4 <A> <B>   # interroga el par A/B
python -m studies.contest_metrics.probe --exp <ID> --scene office4 --list 20 # pares más disputados
python -m studies.contest_metrics.probe ... 82 17 --set min_count=10          # what-if de umbrales
```

Estructura (I/O en los bordes, núcleo puro):

| fichero | responsabilidad |
|---|---|
| `mapstate.py` | `MapState` (datos del mapa: xyz, ins, normals, colors, clip) |
| `loader.py` | lee ckpt + store + config (I/O) |
| `callbacks.py` | reconstruye sim/seam/color (espejo de `ovo.update_map`) |
| `probe.py` | `ContestProbe` — agrega una vez, clasifica bajo demanda |
| `report.py` | formateo de la traza |
| `cli.py` | argumentos + `--set` de overrides |

Salida por par: features en ambos sentidos, veredicto dirigido de cada lado, veredicto resuelto, y un
aviso `⚠` cuando el challenger que preguntas no es el que dirige la decisión.

## 3. Diagnóstico por casos — tres patrones de fallo

Todos parten de la misma raíz: **`firm_points` cuenta como "firme" cualquier punto robado, aunque sea
1 KF** (porque `min_count=1`). Eso contamina `containment` y `focus` con parpadeos.

### Patrón 1 — falsa simetría (82 ↔ 17)

Dos superficies grandes que se disputan un seam. Ambos sentidos dan `SPLIT`, el resolve lo declara
`frontera` → `NO_ACTION`. Pero la asimetría es real y grande:

| señal | 82←17 | 17←82 |
|---|---|---|
| persistence | 0.629 | **0.104** |
| c (KFs robado), mediana | **27** | 3 |
| % puntos con c=1 | 2% | **27%** |

Con `min_count=10`, el sentido flojo (17←82) se desploma (containment 0.27 → 0.04), la simetría se
rompe, y el resolve aplica el `SPLIT` direccional bueno: **el trozo de 82 va a 17**.

### Patrón 2 — falso segundo dueño (85 → 75)

85 está contenido al 99% en 75 (persist 0.79) → debería fusionarse. Pero el veredicto es
`NO_ACTION "frontera real (≥2 raíces)"`, porque 85 tiene **dos** dueños fuertes:

| challenger | containment | persist |
|---|---|---|
| 85←75 | 0.993 | **0.786** (real) |
| 85←82 | 0.966 | **0.127** (parpadeo) |

El segundo dueño (82) es falso — containment inflado por robos de refilón. Con `min_count` alto, 82
cae, queda una raíz (75), y **85 se fusiona en 75**.

### Patrón 3 — fragmento real ahogado por normalización (82 → 75)

82 tiene un trozo de 378 puntos que **pertenece claramente a 75**:

- 75 los reclama **86%** del tiempo; 82 (dueño oficial) solo **10%**.
- **79%** de esos puntos no los toca nadie más que 75 (exclusivos).

Pero el mecanismo **nunca lo transfiere**, porque `focus` lo mata:

```
focus(82←75) = firm(82,75) / disputados_totales(82) = 378 / ~46.600 = 0.008
```

`focus` normaliza por **toda** la disputa de 82 (dominada por el seam con 17, ~46.000 puntos
distintos). El trozo de 75, aunque impecable, es una fracción minúscula → no entra en ninguna banda.

**`focus` responde "¿este challenger domina TODA la disputa del defender?" cuando la pregunta útil es
"¿este trozo es de un solo dueño?".** Se rompe en cuanto un objeto mal segmentado arrastra varias
disputas a la vez. La señal correcta (persistencia 0.83 + exclusividad 0.79) existe, pero no participa
en decidir la banda.

## 4. Cambios de código

Todo detrás de flags; con la config por defecto el comportamiento no cambia salvo el nuevo `min_count`.

### 4.1 · `min_count` efectivo (default 5)

`firm_points` ahora exige que un punto se robe en **≥ `min_count` KFs** para contar. Default subido de
1 a **5**.

- **Por qué no 1:** con 1, "firm" = "robado alguna vez" → no filtra nada, y `persist`=1.0 sale de una
  sola muestra (1/1). Alimenta los patrones 1 y 2.
- **Por qué no 10:** demasiado alto para esta escena. Fragmentos reales pero cortos (67, 78, 62 → 30)
  topan en c=9 KFs — la trayectoria solo pasó ~9 veces por esas zonas — y con umbral 10 desaparecen.
- **5** mata el parpadeo (c 1-2) sin perder los fragmentos de c 6-9.

> **Limitación conocida:** un absoluto no transfiere entre escenas (depende de cuántos KFs se ve cada
> zona). El flag relativo `firm_tau` (firme si `grabs/claims ≥ τ`) ya está implementado y sería
> escena-independiente; queda como siguiente paso.

### 4.2 · Señal de exclusividad (`PairFeatures.exclusivity`)

Nueva feature en el aggregator: fracción de los `firm_points` de un par que **solo** ese challenger
disputa (ningún otro no-dueño los toca). Mide "¿este trozo es limpio de un solo dueño?" a nivel de
trozo, sin normalizar por el defender.

### 4.3 · Gate de transferencia limpia (última rama de la cascada)

`ovo/entities/contest/clean_transfer.py` + pasada en `manager.report()` tras resolve.

Es la **última rama de la cascada, por par**: solo se aplica a pares que no entraron en strong/partial
(`containment < low`) ni en dominancia (`focus < 0.7`) — la zona que hoy caía a `NO_ACTION borde`. Ahí,
antes de rendirse, comprueba si el trozo es claramente de otro dueño:

```
containment < low  Y  focus < 0.7
        ↓
persist ≥ clean_persist (0.5)  Y  exclusividad ≥ clean_excl (0.7)
   sí → SPLIT · transferencia limpia (trozo → challenger)
   no → NO_ACTION borde
```

Resuelve el patrón 3: 82 puede soltar su seam a 17 (disputa, vía focus) **y** su trocito a 75
(refinamiento, vía esta gate) a la vez — el árbol da un veredicto por defender, esta pasada añade los
trozos limpios que el winner-take-all dejó pasar.

Config (`semantic.contest`): `clean_transfer` (bool, default `false`), `clean_persist` (0.5),
`clean_excl` (0.7), `min_count` (5), `firm_tau` (0.0).

### 4.4 · Ficheros tocados

| fichero | cambio |
|---|---|
| `contest/types.py` | `PairFeatures.exclusivity` |
| `contest/aggregator.py` | cálculo de exclusividad; `min_count` default 5; flag `firm_tau` |
| `contest/clean_transfer.py` | **nuevo** — gate por par con guarda de ruteo |
| `contest/manager.py` | pasada `_add_clean_transfers` en `report()`; config |
| `studies/contest_metrics/probe/` | **nuevo** — módulo de interrogación |
| `diagrams/contest_discriminator.drawio` | árbol actualizado con la gate |

## 5. Resultados

Runs sobre office4 (`contest_fusion=only`, `contest_split_mode=all`, GT + jump). Comparación de mapas
finales:

| config | instancias vivas | nota |
|---|---|---|
| `min_count=1`, sin gate (`c6b48`) | 82 | over-merge de parpadeo (83 se traga 10 instancias) |
| `min_count=10` + gate (`7a69b`) | 114 | pierde fragmentos cortos (67/78/62 no fusionan) |
| **`min_count=5` + gate (`1f112`)** | **105** | punto medio sano |

Verificación en `1f112` (config final):

- **67, 78, 62 → 30** — fragmentos cortos (c≤9) fusionados correctamente. ✓
- **82 descompuesto** (133.9k → 89.7k): seam → 17 (`+42.5k`), trozo → 75 (`+263`, transferencia limpia). ✓
- **85 → 75** (fusión desbloqueada al caer el falso segundo dueño). ✓
- **42 transferencias limpias** aplicadas; veredictos `{NO_ACTION: 76, MERGE: 26, SPLIT: 43}`.

Validación visual pendiente:
```
rerun data/output/Replica/20260706_GT_CLIP_contest-clean-transfer_1f112/office4/rerun.rrd
```

## 6. Conclusiones

1. **El bug de raíz era `min_count=1`:** "firm" sin umbral cuenta parpadeos, y de ahí salen falsas
   simetrías, falsos dueños y `persist`=1.0 de una muestra. Un umbral (5 aquí) lo corta.
2. **`focus` está mal escalado:** normaliza por el defender entero, así que un trozo limpio dentro de
   un objeto mal segmentado con otra disputa mayor es invisible. No es que `focus` mienta — es que
   responde a otra pregunta.
3. **La transferencia limpia lo arregla como refinamiento:** por par, sobre la zona borde, con
   persistencia + exclusividad (señales locales al trozo, no normalizadas por el defender).
4. **Pendiente:** `firm_tau` relativo para no calibrar `min_count` por escena; suelo de masa opcional
   para la cola de trozos diminutos; y decidir si `82→17` (42.5k, reasignación masiva) es siempre
   correcto o necesita una guarda.
