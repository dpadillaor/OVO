# Señal temporal del store (idea aparcada — solo para versión LIVE)

**Fecha:** 2026-07-09
**Estado:** IDEA sin madurar del todo. **NO implementar aún.** Tiene sentido cuando el
algoritmo sea *live* (online, KFs en orden, un solo paso). NO encaja con el diseño
actual de **hot path / cold path** (el cold path reprocesa → "ahora"/racha se
enturbian, y meter bookkeeping temporal en el hot path acopla mal). Ver §5.

---

## 1. Problema que ataca

El store hoy guarda **cantidad**, no **temporalidad**:
- `_grabs[p][g]` = nº de KFs que el grabber `g` robó el punto `p`
- `_claims[p]`   = nº de KFs que `p` cayó bajo alguna máscara (leal o robo)
- `_sightings[p]`= nº de KFs que `p` reproyectó bien (visible, con o sin máscara)

Con conteos se han hecho maravillas, pero pierden el **intercalado**: no es lo mismo
que los votos leales del dueño salgan solo al principio (punto ya asentado) que
intercalados con los robos (disputa viva). Dos historias opuestas dan el mismo
contador. Queremos "última vez / cómo de reciente vota cada instancia", no solo
cuántas veces.

## 2. El nudo del reloj (por qué es sutil)

Recency = "hace cuánto" necesita un **reloj**, y elegir el reloj mal miente. Tres
opciones, cada una con su fallo:

1. **KF global** (`ahora − last_KF`). Penaliza a la vez la invisibilidad del punto
   Y la ausencia de la instancia. El peor. `98−42=56` no dice si el challenger dejó
   de robar o si la cámara miró para otro lado.

2. **Reloj = sightings del punto** (`_sightings[p]` en el momento del robo; recency =
   sightings_ahora − snapshot). **Ventaja: merge-safe** (los puntos no se fusionan,
   sus ids son permanentes → inmune a merges de instancias). **Fallo:** confunde
   *instancia ausente* con *instancia que abstiene*. La cámara puede seguir viendo el
   punto (lo tiene el dueño) mientras el challenger se fue de escena; esos sightings
   cuentan como "abstenciones" falsas.

3. **Reloj = presencia de la instancia** (`_seen[ins]`, contador +1 por KF que la
   instancia tiene máscara; snapshot `_last_grab[p][g] = _seen[g]`; recency =
   `_seen[g] − snapshot` = "veces que g estuvo presente y no reclamó p"). **Separa
   "se fue" de "volvió y abandonó"** — congela durante la ausencia, avanza solo al
   volver. Es el reloj honesto. **Deudas:**
   - **Merge-fragile:** el snapshot está en el reloj de la instancia fusionada; al
     moverlo a la superviviente cambia de escala (no hay offset único). Mitigación:
     `max` aceptando error menor + **auto-cura** (el siguiente robo reescribe el
     snapshot con el reloj nuevo; solo una disputa que se calla justo tras el merge
     queda mal, y esa ya está casi muerta).
   - **"Presente" ≠ "p en su vista":** `_seen` cuenta que la instancia tenía máscara
     *en algún sitio*, no que `p` cayera en su campo. Segundo orden (p vive en la
     frontera disputada de la instancia → presente ⇒ zona de p casi siempre visible).

**Tensión de fondo:** reloj anclado al punto (opción 2) = merge-safe pero no separa
ausencia/abstención. Reloj anclado a la instancia (opción 3) = separa pero merge-frágil.

## 3. Cómo guardarlo (memoria — NO es locura)

Miedo a evitar: guardar la **lista de todos los KFs** por punto → crece sin fin →
locura. **No es eso.**

Escalones, de más barato a más rico:

- **1 número por celda (last-tick).** `_last_grab[p][g]` = un int, se **sobrescribe**
  en cada robo (se tira el resto de la historia). Coste: la celda pasa de 1 int a 2
  ints, mismo nº de celdas (solo disputados = minoría). Doblar un rincón pequeño
  (~orden de decenas de KB). Da **recency**.
- **2 números (first + last).** Añade **span** (ráfaga vs sostenido).
- **Ventana deslizante como MÁSCARA DE BITS (la buena):** por celda (punto, challenger)
  UN `int32` cuyos bits son las últimas 32 veces que se vio el punto: `bit=1` lo robó
  el challenger, `bit=0` lo tuvo el dueño (leal). Cada vez que el punto se ve: shift a
  la izquierda (entra 0), set bit0 si robó; las veces viejas se caen solas. **Un
  entero por celda**, igual de barato que el contador, pero encierra el intercalado
  reciente. Nota: un `0` codifica también la lealtad del dueño → una sola máscara
  encierra challenger-vs-dueño.

### Stats que salen de la máscara (bit tricks)
- **% reciente** = popcount(mask) / X  → presión reciente del challenger
- **racha actual** = nº de 1s (o 0s) al final (trailing run) → ¿está caliente AHORA?
- **racha máxima** = run más largo de 1s → ¿llegó a dominar?
- **recency** = posición del 1 más reciente

Ejemplo: mask con 69% de 1s y racha máxima 5 pero **racha actual 0** = "dominó
histórico pero se enfrió ahora mismo". Eso es la temporalidad que el conteo tira.

### Por qué la ventana-máscara esquiva el nudo del reloj
La ventana está **anclada al punto** (avanza por sighting del punto) → **merge-safe**
como la opción 2. Y es **corto plazo auto-olvidante** (solo últimas X) → el problema
ausencia/abstención queda **acotado a la ventana y suave**: para "¿quién está
caliente ahora?" da igual si el challenger se fue o abstuvo — en ambos casos no está
reclamando ahora. La pregunta cambia de "¿abandonó?" (necesita reloj de oportunidad,
duro) a "¿quién reclama ahora?" (mirar los slots recientes, fácil). Ese cambio de
pregunta es la clave.

## 4. Dominio: A vs B

- **A — solo robos.** `_last_grab`/máscara espeja `_grabs`. Barato, cero estructura
  nueva. Recency del robo en el reloj del propio challenger.
- **B — toda instancia (incl. dueño leal).** Tabla nueva `punto → instancia → tick`
  más ancha (el dueño leal nunca se graba como grabber hoy). Permite comparar
  "cuándo votó el dueño" vs "cuándo el challenger" = **quién manda AHORA**. Más
  memoria + otro call site en el hot path. El reloj de presencia (opción 3) es
  auto-contenido por instancia → comparar dos relojes de presencia distintos en B es
  peras con manzanas → el reloj de presencia empuja hacia A. La ventana-máscara (0=leal)
  ya mete al dueño gratis sin tabla nueva.

## 5. Por qué se APARCA (condición del usuario)

Todo esto tiene sentido con el algoritmo **live** (online, un paso, KFs en orden,
"ahora" bien definido, shift por sighting natural en el hot path). El diseño **actual
es hot path / cold path**: el cold path reprocesa/replay → "ahora" y las rachas se
enturbian, y el bookkeeping temporal (shift por sighting) acopla mal con el hot path.
→ **No implementar hasta la versión live.** Retomar entonces desde este doc.

## 6. Señales adyacentes ya en el store pero SIN explotar
- **fiabilidad** = `claims / sightings` (el punto se ve mucho pero rara vez bajo máscara)
- **P4 huérfano-asignado** = `sightings − claims` (visto sin máscara, asignado por herencia)

Nombradas en comentarios de `store.py` y `aggregator.py`, nunca llegan a `PairFeatures`.

## 7. Próximo paso cuando se retome
1. Confirmar plumbing: `_track_objects` (ovo.py) puede marcar presencia por instancia
   y pasar snapshot a `record_grab`.
2. Elegir escalón (last-tick vs ventana-máscara) y dominio (A vs B).
3. Decidir uso en el discriminador: gate activo (nueva señal `liveness`/racha) vs
   decay pasivo (pesar `firm_points` por recency). Gate activo se mide aislado mejor.
