# Concepto: punto en disputa, store, mass y strong_points

Explicación lenta, desde cero, de qué es la **masa**.

## Paso 1: qué es un punto en disputa

Un punto 3D del mapa pertenece a una instancia (su **dueño**).

> Ejemplo: el punto **P** pertenece a la instancia **A**.

En un keyframe (frame de cámara) miras dónde cae P en la imagen. Si cae **dentro
de la máscara de OTRA instancia B**, hay conflicto:

> P es de A, pero se ve bajo B → **P es un punto en disputa**.

## Paso 2: qué guarda el store

Cada vez que pasa eso, se suma 1. El store guarda, por cada punto, cuántos
keyframes lo vieron bajo cada instancia:

```
punto P → { B: 3 }
```

Lectura: **"el punto P se vio bajo B en 3 keyframes distintos."**

El número (3) es el contador. Cuantos más frames ven ese punto cayendo en B, más
sube.

## Paso 3: ejemplo con varios puntos

La instancia A tiene 3 puntos en disputa, todos vistos bajo B:

```
punto P1 → { B: 3 }     (P1 se vio bajo B en 3 KFs)
punto P2 → { B: 5 }     (P2 se vio bajo B en 5 KFs)
punto P3 → { B: 2 }     (P3 se vio bajo B en 2 KFs)
```

## Paso 4: las dos cuentas

| Métrica | Qué cuenta | En el ejemplo |
|---|---|---|
| **strong_points** | cuántos PUNTOS distintos hay | P1, P2, P3 → **3** |
| **mass** | SUMA de todos los contadores | 3 + 5 + 2 → **10** |

- `strong_points` = número de filas.
- `mass` = suma de los números de la derecha.

Por eso siempre `mass ≥ strong_points`. Si un punto se vio muchos KFs, aporta
mucho a la masa pero sigue siendo 1 punto.

## Para qué sirve la masa

Solo como **filtro de ruido**. Antes de decidir nada:

```python
pairs = [p for p in pairs if p.mass >= min_mass]   # min_mass = 50
```

Par con masa baja = pocas observaciones = coincidencia fugaz → se descarta.
Es una medida de **confianza/peso**, NO de geometría.

## Aviso

`mass` es **cruda, sin normalizar**. Un objeto grande visto en muchos KFs acumula
masa enorme (la pared del experimento: 1.4M) sin que eso diga nada de si debe
fusionarse. Por eso la masa solo vale para el corte mínimo, nunca para decidir
merge/split. Para decidir se usa `containment` (que sí está normalizado por
tamaño).
