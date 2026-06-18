# Las señales del contest — qué miramos para decidir unir/separar

> Divulgativo. Cada señal: de dónde sale, cómo se calcula, qué significa.
> Detalle de decisiones en `flujo_decision_contest.puml`; código en
> `ovo/entities/contest/aggregator.py` (features) y `ovo/entities/ovo.py` (guardas).

## La idea en una frase

Cuando dos detecciones (las llamamos **A** = perdedor y **W** = ganador) se pelean por
los mismos puntos 3D, calculamos unas cuantas **señales** sobre esa disputa y, con ellas,
decidimos: **unir**, **separar** o **dejar igual**.

---

## 1. Datos crudos (de dónde sale todo)

Antes de calcular señales, el sistema apunta cosas básicas mientras construye el mapa:

| Dato crudo | De dónde viene | Qué es (en cristiano) |
|---|---|---|
| **Reclamaciones por punto** | Se anota cada vez que la máscara de una instancia "pisa" un punto que es de otra, en cada fotograma (keyframe). | Un marcador: "el punto P fue reclamado por la instancia W en N fotogramas". Es la materia prima de la disputa. |
| **Observaciones por punto** | Cuántos fotogramas vieron ese punto en total. | El "tiempo en pantalla" del punto. Sirve para saber si una reclamación es mucha o poca. |
| **Normales de superficie** | Se estiman de la profundidad (depth) en cada punto. | La dirección a la que "mira" la superficie en ese punto. Detecta esquinas/escalones. |
| **Color** | El RGB de cada punto. | El color real de la superficie. |
| **Descriptor (CLIP)** | Un vector que resume el aspecto de cada objeto (modelo de visión). | Una "huella de aspecto": objetos parecidos tienen huellas parecidas. |
| **Tamaño de instancia** | Nº de puntos de cada objeto. | Cómo de grande es A y W (denominador de varias señales). |

---

## 2. Señales derivadas (las que de verdad deciden)

Todas se calculan por cada par A→W. Las 6 primeras salen del aggregator; las 3 últimas
son "guardas" que solo se consultan en casos dudosos.

| Señal | Cómo se calcula | Rango | Qué significa (intuición) |
|---|---|---|---|
| **Containment** (contención) | puntos de A reclamados por W ÷ tamaño total de A | 0–1 | "¿Qué fracción de A está realmente dentro de W?" Alto = A es casi todo un trozo de W → candidato a UNIR. |
| **Reverse containment** (contención inversa) | lo mismo al revés: puntos de W dentro de A ÷ tamaño de W | 0–1 | Si **ambos** sentidos son altos, se contienen mutuamente = frontera ambigua → no cortar. |
| **Strong points** (puntos fuertes) | nº de puntos de A que vieron a W en suficientes fotogramas | conteo | Cuántos puntos sostienen la disputa de verdad (no un roce de un fotograma). |
| **Mass** (masa) | suma de todas las reclamaciones (fotogramas) sobre esos puntos | conteo | Peso total de la evidencia. Masa baja = ruido → ni se considera. |
| **Persistence** (persistencia) | media, por punto, de (fotogramas con W ÷ fotogramas totales del punto) | 0–1 | "¿La disputa es constante o un parpadeo?" Baja = W solo roza esos puntos de pasada → no fiable. |
| **Focus** (foco) | strong points ÷ total de puntos en disputa de A | 0–1 | "¿La pelea de A es contra UN solo rival o contra muchos?" Alto = un ganador domina → disputa clara. |

### Guardas (desempate en zona dudosa)

| Guarda | Cómo se calcula | Qué significa (intuición) |
|---|---|---|
| **Descriptor sim** (parecido de aspecto) | similitud coseno entre los descriptores CLIP de A y W | Alto = se ven igual → probablemente el MISMO objeto partido → UNIR. CLIP **no** distingue dos objetos iguales pegados (2 sillas iguales), por eso es solo un desempate, no la regla principal. |
| **Seam-normal angle** (giro de superficie en la costura) | ángulo entre las normales del trozo disputado y las de W en la frontera | Casi 0° = superficie continua (mismo objeto, ej. trozo de mesa) → cortar y devolver. Grande (>~20°) = hay un escalón/esquina = dos objetos pegados (cojín sobre sofá) → no cortar. |
| **Color ΔE** (dos sentidos) | distancia de color (Lab) del trozo frente a W, y frente al resto de A | Si el trozo se parece en color más a W → es de W → transferir. Si se parece más a A → es de A → no tocar. Caza objetos coplanares (misma superficie) pero de **distinto color**, que la normal no separa. |

---

## 3. Cómo encajan (resumen)

1. **¿Hay masa suficiente?** Si no → ruido, dejar igual.
2. **¿Containment alto?** → es un trozo de W → UNIR (salvo que sea mutuo = frontera).
3. **Zona media:** desempata el **descriptor** (¿se parecen? → unir).
4. **Disputa concentrada (focus alto) pero poca contención:** podría ser un fragmento
   robado. Antes de cortar, tres guardas:
   - **persistence** baja → parpadeo, no fiable → dejar igual.
   - **seam-normal** con quiebre → objetos pegados → dejar igual.
   - **color** distinto de W → no es de W → dejar igual.
   - Si pasa las tres → **SEPARAR** (devolver el trozo a W).

La filosofía: la **geometría** (containment, normales) manda; el **descriptor** y el
**color** solo desempatan donde la geometría no llega; la **persistencia/masa** filtran
el ruido temporal.
