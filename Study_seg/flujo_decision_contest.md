# Flujo de decisión — Contest (merge/split de instancias)

Diagrama del camino real que recorre cada **perdedor** en
`ContestDiscriminator.classify()` + reconciliación en `ContestManager._reconcile()`.

Umbrales por defecto: `high=0.8`, `low=0.5`, `min_mass=50`,
`min_split_cont=0.05`, `min_count=1`.

```mermaid
flowchart TD
    START([Perdedor A + sus pares A→W]) --> FILT{¿Algún par<br/>mass ≥ min_mass=50?}
    FILT -->|no| NA_RUIDO[NO_ACTION<br/>'ruido masa baja']

    FILT -->|sí| CLAS[Clasificar pares por containment]
    CLAS --> STRONG{¿Hay pares<br/>cont ≥ high=0.8?<br/>strong}

    %% ---------- rama STRONG ----------
    STRONG -->|sí| SYM{len strong == 1<br/>Y rev_cont ≥ high?}
    SYM -->|sí| DEFER[DEFER_TO_FUSION<br/>'simétrico → geometría/descriptor']
    SYM -->|no| ROOTS{¿roots de winners<br/>fuertes == 1?<br/>union-find fusión}
    ROOTS -->|sí, 1 raíz| MERGE[MERGE_CONTAINMENT<br/>'contención alta, 1 raíz']
    ROOTS -->|no, ≥2 raíces| NA_FRONT[NO_ACTION<br/>'frontera real ≥2 raíces']

    %% ---------- rama PARTIAL ----------
    STRONG -->|no| PARTIAL{¿Hay pares<br/>low=0.5 ≤ cont < 0.8?}
    PARTIAL -->|sí| SPLIT_P[SPLIT<br/>'parcial' → candidato]

    %% ---------- rama DOMINANCIA ----------
    PARTIAL -->|no| FOCUS{¿Hay par con<br/>focus ≥ 0.7<br/>Y cont ≥ min_split_cont=0.05<br/>Y mass ≥ 50?}
    FOCUS -->|sí| SPLIT_D[SPLIT<br/>'dominancia' → candidato]
    FOCUS -->|no| NA_BORDE[NO_ACTION<br/>'borde, contención baja']

    %% ---------- reconciliación ----------
    SPLIT_P --> REC{¿Existe MERGE inverso<br/>A→B para este SPLIT B→A?}
    SPLIT_D --> REC
    REC -->|sí| NA_SUP[NO_ACTION<br/>'split suprimido, MERGE inverso gana']
    REC -->|no| KEEP[SPLIT se mantiene]

    classDef merge fill:#2e7d32,color:#fff,stroke:#1b5e20;
    classDef split fill:#1565c0,color:#fff,stroke:#0d47a1;
    classDef noact fill:#9e9e9e,color:#fff,stroke:#616161;
    classDef defer fill:#6a1b9a,color:#fff,stroke:#4a148c;
    class MERGE merge;
    class SPLIT_P,SPLIT_D,KEEP split;
    class NA_RUIDO,NA_FRONT,NA_BORDE,NA_SUP noact;
    class DEFER defer;
```

## Dónde cae cada caso real (office3)

| Caso | Realidad | Veredicto actual | Rama | ¿Correcto? |
|------|----------|------------------|------|------------|
| 63/151 | misma pared, 2 trozos adyacentes | SPLIT ambos sentidos | parcial + dominancia | ✗ debería MERGE |
| 148/194 | misma silla (respaldo+asiento) | SPLIT | parcial/dominancia | ✗ debería MERGE |
| 95 | trozo que debería fusionar | NO_ACTION | frontera real ≥2 raíces | ✗ blob vecino bloquea |
| 91 | trozo de mesa → 103, hay blob mesa+silla | NO_ACTION | frontera real ≥2 raíces | ~ se protege, no arregla |
| 89/103 | mesa mal segmentada unida a silla | SPLIT | dominancia | ~ split real pero veneno upstream |
| 31/22 | objetos distintos | SPLIT | dominancia | ✗ falso positivo |
| 18/26 | objetos distintos | SPLIT | dominancia | ✗ falso positivo |
| 188 | suelo oversegmentado | NO_ACTION | borde | ~ ruido |

## Limitaciones que revela el diagrama

1. **Containment solo ve anidamiento, no adyacencia.** Trozos del mismo objeto
   pegados lado a lado (silla, pared) dan cont baja en ambos sentidos → caen en
   SPLIT en vez de MERGE. La rama STRONG nunca se activa. Falta una señal de
   **descriptor/cooccurrence** paralela a containment.

2. **Blob upstream mal segmentado envenena `roots`.** Un vecino-basura aparece
   como 2ª raíz fuerte → `frontera real` bloquea merges legítimos (95, 91). El
   orden debería ser: **ejecutar SPLIT del blob → recomputar → MERGE.**

3. **Rama dominancia demasiado agresiva.** cont~0.1 + focus alto marca SPLIT
   entre objetos vecinos distintos (31/22, 18/26). Subir `min_split_cont`
   0.05 → 0.15 las elimina sin tocar las legítimas.
