# Casos de contención — taxonomía y límites

Aclaración de qué casos distingue (y cuáles NO) el mecanismo contest.

## Lo fundamental: containment es DIRECCIONAL y NORMALIZADO

```
containment(A→W) = | puntos de A vistos bajo W | / | A |
```

Dos consecuencias que confunden:

1. **No es simétrico.** `cont(A→W) ≠ cont(W→A)`. El mismo par físico (A,B) produce
   **2 filas** en el CSV: `A→B` y `B→A`, cada una con su número.

2. **Está dividido por el tamaño de A.** Un objeto grande sale siempre con
   containment pequeño aunque el solape absoluto sea enorme. → penaliza a los
   objetos grandes.

## Las geometrías posibles entre A y B

Pensar en las nubes de puntos como áreas:

| Geometría | cont(A→B) | cont(B→A) | Veredicto código | ¿Correcto? |
|---|---|---|---|---|
| **A dentro de B** (anidado) | alta | baja | MERGE | ✓ A es fragmento |
| **Casi idénticos** (solapan todo) | alta | alta | DEFER_TO_FUSION (simétrico) | ✓ deja a fusión |
| **Parcial** (medio metido) | 0.5–0.8 | baja | SPLIT parcial | ✓ trozo / ✗ casi-merge |
| **Adyacentes, MISMO objeto** | baja | baja | SPLIT×2 ✗ | ✗ debería MERGE |
| **Adyacentes, objetos DISTINTOS** | baja | baja | SPLIT×2 / NO_ACTION | ✗ debería NO_ACTION |
| **Sin relación** | ~0 | ~0 | NO_ACTION borde | ✓ |

### "SPLIT simétrico" = las dos filas del mismo par dicen SPLIT

En las geometrías adyacentes, el mecanismo mira cada dirección por separado:
- fila `A→B`: cont baja + focus alto → SPLIT ("dale a B tus puntos de roce")
- fila `B→A`: cont baja + focus alto → SPLIT ("dale a A tus puntos de roce")

→ **ambas filas SPLIT sobre el mismo par.** Se pasan los puntos en círculo.
Síntoma de que NO es un split: es frontera (distintos) o mismo objeto.

## El problema clave: las dos filas "Adyacentes" son INDISTINGUIBLES por geometría

```
  MISMO objeto (silla partida)        DISTINTOS (silla pegada a mesa)
  ┌──A──┐┌──B──┐                       ┌──A──┐┌──B──┐
  │     ││     │                       │     ││     │
  └─────┘└─────┘                       └─────┘└─────┘
       ↑ roce                                ↑ roce
  cont baja / baja                      cont baja / baja
  QUIERE merge                          QUIERE que la dejes
```

Misma señal geométrica (cont baja/baja), realidad opuesta. **Containment no puede
separarlas.** La única señal fiable es **apariencia/semántica → descriptor
(CLIP/DINO)**: mismo objeto = descriptores parecidos; objetos distintos =
descriptores distintos. Por eso el plan es meter el descriptor en el discriminador.

## Matiz importante: 63/151 NO es tan indistinguible

Tamaños despejados (`size = strong_points / containment`):

```
151 ≈  57.755 puntos   (trozo de pared)
63  ≈ 252.893 puntos   (pared grande)
solape absoluto ≈ 33–45k puntos   <- ENORME
```

`cont(151→63) = 0.794` → el trozo pequeño está **79% dentro** de la pared grande.
Eso casi es MERGE (high=0.8). La asimetría (0.79 vs 0.13) no es ruido: es **exactamente
porque 63 es ~4× más grande**. El solape grande, dividido por los 253k de 63, sale 13%.

**Conclusión:** la dirección pequeña→grande YA lleva la señal de merge. No hace falta
descriptor para 63/151 — basta **mirar el máximo de las dos direcciones** y bajar
`high` un pelín (0.794 < 0.8 por los pelos). Entonces:
- `151→63` pasa a MERGE
- `_reconcile` mata el `63→151` SPLIT inverso

Esto separa dos sub-casos de "adyacentes mismo objeto":
- **trozo grande + trozo pequeño** (pared): el pequeño tiene cont alta → MERGE por
  dirección máxima. **Resoluble sin descriptor.**
- **dos trozos de tamaño similar** (silla respaldo+asiento, 148/194: cont 0.32/0.06):
  ninguna dirección llega → **necesita descriptor**.

## Resumen de qué arregla qué

| Sub-caso | Ejemplo | Señal que lo resuelve |
|---|---|---|
| Fragmento anidado | MERGE actuales | containment (ya funciona) |
| Adyacente, tamaños dispares | 63/151 | **máx de las 2 direcciones** + high≈0.75 |
| Adyacente, tamaños similares, mismo obj | 148/194 | **descriptor** (containment ciego) |
| Adyacente, objetos distintos | 31/22, 18/26 | **descriptor** + subir min_split_cont |
| Blob upstream mal segmentado | 91, 89/103 | split-primero en el Actuator |
