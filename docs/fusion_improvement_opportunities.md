# Oportunidades de mejora en la fusión de instancias semánticas

---

## Reparto de responsabilidades

Antes de entrar en mejoras concretas conviene delimitar qué es problema de quién, porque es fácil confundir síntomas de módulos distintos.

**SLAM — geometría correcta.** El SLAM es responsable de que las posiciones 3D estén bien. Si una instancia aparece desalineada respecto a donde debería estar, eso no es un fallo de la fusión semántica — es un fallo de odometría o de la corrección del loop closure. El módulo de fusión opera sobre los puntos tal como los entrega el SLAM; no puede compensar una geometría fundamentalmente incorrecta.

**Segmentación 2D — reconocer bien y no fragmentar.** La calidad del segmentador (SAM2) determina con qué instancias se trabaja. Si SAM2 parte un objeto en dos máscaras, en 3D aparecerán como dos instancias distintas que habrá que fusionar. Si no detecta un objeto, no existe en el mapa. Hay margen de mejora aquí — filtrado de máscaras, NMS 2D antes de levantar a 3D — pero es un problema de la capa de percepción, no de fusión.

**Fusión de instancias 3D — nuestro problema.** Dado un conjunto de instancias 3D construidas a partir de la segmentación 2D y las posiciones del SLAM, decidir cuáles representan el mismo objeto y deben unificarse. Esto ocurre principalmente tras optimizaciones globales del mapa (loop closures, bundle adjustment). Es el foco de este documento.

---

## Contexto

La fusión se ejecuta en `ovo.py::update_map()`, que el orquestador llama cuando el SLAM detecta un loop closure o realiza una actualización global:

```
complete_semantic_info()          # vacía la cola de descriptores pendientes
_remove_missing_instances()       # elimina instancias que han quedado sin puntos
_fuse_overlapping_instances()     # ← aquí ocurre la fusión
_update_descriptors_after_fusion()
update_objects_clip() / fusion_encoder.update_objects()
```

La decisión de fusión la toma `SemanticGeometricFusion.same_instance()` con tres criterios en cascada: distancia de centroides, similitud coseno del descriptor semántico, y solapamiento de la nube de puntos.

---

## Casos que el sistema debe discriminar

El sistema se enfrenta a situaciones muy distintas y la dificultad está en que algunas comparten la misma firma aparente (similitud alta, centroides próximos) pero requieren decisiones opuestas.

**Loop closure.** El robot recorre una zona dos veces. La misma silla genera dos instancias 3D en posiciones ligeramente distintas (antes y después de la corrección global). Deben fusionarse. Señal característica: nunca co-ocurrieron en el mismo keyframe (son de recorridos distintos), pero después de la corrección su posición 3D coincide.

**Bundle adjustment / más observaciones del mismo objeto.** No es un loop closure, simplemente el objeto se ha visto desde más ángulos a lo largo del recorrido y hay instancias parcialmente solapadas del mismo objeto. Deben fusionarse. La señal es solapamiento geométrico alto, aunque sea asimétrico si una observación es más completa que la otra.

**Over-segmentation.** SAM2 parte un objeto en dos o más máscaras (el respaldo y el asiento de una silla, o una planta con varias ramas). En 3D aparecen como instancias adyacentes con similitud semántica muy alta. Deben fusionarse, aunque la co-ocurrencia pueda ser alta porque SAM2 los veía juntos y los separaba. Es el caso más difícil porque comparte muchas señales con el caso siguiente.

**Observación parcial que evoluciona a completa.** Al principio el objeto solo se ve de un lado y tiene pocos puntos. Más adelante, con más observaciones, la instancia es más completa. Las dos versiones deben fusionarse. La señal geométrica es asimétrica: la instancia pequeña está contenida en la grande pero no al revés.

**Mismo tipo, instancias distintas.** Dos papeleras idénticas juntas, dos sillas iguales en una sala de reuniones. Su similitud CLIP es altísima, sus centroides están próximos, y pueden tener solapamiento bajo pero no nulo si están muy juntas. No deben fusionarse. La señal discriminativa es que SAM2 las segmentó como objetos distintos en los frames donde aparecían juntas — co-ocurrencia alta con identidades separadas.

**Objeto pequeño dentro o encima de uno grande.** Un cojín sobre un sofá, un libro sobre una mesa. El solapamiento es asimétrico: la mayoría de los puntos del objeto pequeño están cerca del grande, pero no al revés. No deben fusionarse. La señal es el ratio de tamaño muy asimétrico y que semánticamente son distintos.

**Objetos en contacto.** Una televisión sobre su mueble, una lámpara sobre una mesa. Comparten región espacial pero son semánticamente distintos. En general no es un caso problemático porque la similitud semántica es baja.

El par más difícil es **loop closure vs mismo tipo instancias distintas**: misma firma semántica, centroides próximos, bajo solapamiento. La diferencia clave es la co-ocurrencia: en el loop closure las dos instancias nunca estuvieron en el mismo keyframe; en el caso de las dos papeleras, sí.

---

## Problemas en la implementación actual

### El solapamiento se mide solo en una dirección

`compute_pcd_overlap` calcula qué fracción de los puntos de `pcd1` están cerca de `pcd2`, pero no al revés. Para el caso de objeto pequeño dentro de uno grande: el 90% de los puntos del cojín estarán cerca del sofá, pero solo el 10% de los del sofá estarán cerca del cojín. Dependiendo del orden en que lleguen los argumentos al bucle, el mismo par puede dar resultados opuestos.

La solución es calcular ambas direcciones y exponer tres métricas:

```python
sym_overlap  = min(overlap_12, overlap_21)   # los dos se solapan mutuamente
max_overlap  = max(overlap_12, overlap_21)   # al menos uno contiene al otro
containment  = max_overlap - sym_overlap     # grado de asimetría
```

Cada caso de uso tiene una métrica natural: para un loop closure lo relevante es `sym_overlap` (las dos observaciones del mismo objeto deberían solaparse bien); para una observación parcial que evoluciona a completa, lo relevante es `containment`.

### 2. El bucle greedy produce resultados dependientes del orden

El bucle actual fusiona instancias de forma incremental y mutable:

```python
for i, inst1 in enumerate(objects_list):
    for inst2 in objects_list[i+1:]:
        if same_instance(inst1, inst2):
            fuse(inst1, inst2)   # inst1 muta: ahora es A+B
```

Una vez que A y B se fusionan, `inst1` ya no representa a A sino a la instancia combinada A+B. Si en la siguiente iteración A+B parece compatible con C, los tres se fusionan — aunque A y C no sean compatibles por sí solos. Es el problema clásico de la transitividad no controlada.

La solución estándar en la literatura (OVIR-3D, ICCV 2023) es separar el paso de evaluación del paso de fusión:

1. Calcular todas las compatibilidades sin mutar nada → grafo de compatibilidad
2. Encontrar componentes conexas con union-find → O(N + E), prácticamente gratuito
3. Fusionar cada componente

Esto garantiza que el resultado es independiente del orden de iteración y que no hay fusiones que ocurran "por accidente" por transitividad.

### 3. No se aprovecha la señal de co-ocurrencia

Cada `Instance3D` almacena `kfs_ids`: los keyframes en los que fue observada. Esta información no se usa en la decisión de fusión, y es probablemente la señal más discriminativa que tenemos disponible sin coste adicional.

El caso más problemático en la práctica es este: dos sillas idénticas juntas. Su similitud semántica (CLIP) puede ser cercana a 1.0, sus centroides están a poca distancia, y el solapamiento puede ser bajo pero no nulo. Con los criterios actuales, la decisión es ambigua. Pero si las dos sillas aparecen en los mismos keyframes, SAM2 las segmentó explícitamente como objetos distintos en esos frames — evidencia directa de que son instancias separadas.

La señal contraria también es informativa: si dos instancias nunca co-ocurrieron en ningún keyframe pero tienen alta similitud semántica y posiciones similares después de la corrección global, es un candidato natural a loop closure.

---

## El espacio de casos que hay que discriminar

Para diseñar mejoras concretas ayuda tener claro qué situaciones debe distinguir el sistema:

| Caso | Descripción | Señal clave |
|------|-------------|-------------|
| **A** | Loop closure — mismo objeto, dos recorridos | No co-ocurrencia + similitud alta |
| **B** | Observación parcial → completa | `containment` alto, `sym_overlap` bajo |
| **C** | Over-segmentation (silla en partes) | Similitud extrema + AABBs adyacentes |
| **D** | Mismo tipo, instancias distintas (dos papeleras) | Co-ocurrencia alta → no fusionar |
| **E** | Objeto pequeño dentro de grande (cojín en sofá) | Ratio de tamaño muy asimétrico |
| **F** | Objetos en contacto (TV sobre mueble) | Semánticamente distintos → fácil |
| **G** | Mismo objeto, más observaciones, mismo recorrido | `sym_overlap` alto |

El par más difícil es **A vs D**: misma firma semántica, centroides próximos, bajo solapamiento en ambos. La única diferencia observable es la co-ocurrencia en keyframes.

---

## Mejoras propuestas, por prioridad

### Alta prioridad — correcciones con impacto inmediato

**Overlap simétrico** (bug). Calcular ambas direcciones en `compute_pcd_overlap` y actualizar la lógica de decisión para usar `sym_overlap`, `max_overlap` y `containment` según el caso. Bajo coste de implementación, impacto directo en los casos B y E.

**Co-ocurrencia como veto**. Si dos instancias comparten más de N keyframes, vetarlas independientemente de su similitud semántica. El umbral N es un hiperparámetro (inicialmente 3). Esto resuelve el caso D sin afectar A.

**Componentes conexas en lugar de greedy**. El cambio es localizado en `_fuse_overlapping_instances` y no modifica ninguna interfaz externa. El coste adicional es O(N + E) sobre el O(N²) ya existente — despreciable.

**Hard veto por ratio de tamaño**. Si una instancia tiene más de 5 veces los puntos de la otra, es muy improbable que sean el mismo objeto. Es un check O(1) sin dependencias externas:

```python
if max(n_pts1, n_pts2) / min(n_pts1, n_pts2) > 5.0:
    return False
```

### Media prioridad — mejoras de lógica con más coste


### IoU 3D del bounding box como pre-filtro espacial

El criterio espacial actual es la distancia entre centroides, que tiene un problema conocido: para objetos grandes o elongados, dos objetos distintos pueden tener centroides muy próximos (una estantería y un libro sobre ella), mientras que el mismo objeto visto desde dos ángulos muy distintos puede tener centroides alejados antes de la corrección.

El IoU 3D del axis-aligned bounding box es una alternativa más robusta y barata de calcular:

```
IoU3D = vol(intersección de AABBs) / vol(unión de AABBs)
```

Como criterio de decisión final tiene limitaciones (los AABBs son sensibles a objetos no convexos, y la literatura prefiere el overlap de nube de puntos para la decisión final). Pero como **pre-filtro** es muy práctico: si `IoU3D = 0` y la distancia de centroides supera un umbral amplio, se puede descartar el par sin calcular nada más. Reemplaza o complementa al filtro de centroide actual con información más completa de la geometría del objeto.

La implementación es directa: Open3D expone `get_axis_aligned_bounding_box()` sobre nubes de puntos, y el IoU 3D de dos AABBs es aritmética pura sobre sus esquinas.

### Mejoras de lógica con más coste

**NMS 2D antes de levantar a 3D.** Si dos máscaras SAM2 en el mismo frame tienen IoU 2D > 0.6, suprimir la más pequeña antes de crear la Instance3D. Reduce la población de instancias redundantes en origen, que es donde más barato es resolver el over-segmentation.

**Threshold adaptativo post loop-closure**. La optimización global mejora la alineación pero no la elimina. Relajar temporalmente `th_centroid` durante el pase de fusión post-corrección global (por ejemplo ×2) aumenta el recall de loop closures sin afectar la fusión online normal.

**CLIP + DINO como señales combinadas**. Actualmente los modos son excluyentes. CLIP captura bien la categoría semántica global; DINO es más robusto a variaciones de viewpoint y captura mejor la apariencia local. Para el over-segmentation (partes de un mismo objeto), combinarlos con una suma ponderada da más información que cualquiera solo.

### Baja prioridad — mejoras más costosas o con impacto incierto

**Threshold adaptativo por confianza del descriptor**. Una instancia observada en 3 keyframes tiene un descriptor mucho más ruidoso que una observada en 50. Se puede ajustar el umbral de similitud inversamente a la confianza mínima de los dos descriptores comparados.


---

## El grafo de covisibilidad del SLAM

ORB-SLAM3 mantiene internamente un grafo de covisibilidad entre keyframes: dos keyframes están conectados si observan un mínimo de puntos del mapa en común. Esta información es más rica que la co-ocurrencia de instancias que ya tenemos:

- La co-ocurrencia nos dice: "estas dos instancias aparecieron en el mismo frame"
- La covisibilidad nos diría: "estos dos keyframes observaban la misma región, aunque las instancias no coincidieran exactamente"

Hay dos usos concretos que serían interesantes:

*Pre-filtrado O(N²) → O(N·K)*: solo comparar instancias cuyos keyframes tienen covisibilidad en el grafo. En escenas grandes esto reduciría drásticamente el número de comparaciones.

*Señal de loop closure a nivel de instancia*: si dos instancias tienen covisibilidad cero entre sus keyframes (recorridos completamente separados) pero sus posiciones 3D coinciden tras la corrección — eso es exactamente la firma de un loop closure. Es la señal contraria a la co-ocurrencia: la ausencia de covisibilidad entre dos instancias similares es evidencia positiva de que son el mismo objeto visto en momentos distintos.

El coste de implementación es moderado: requiere exponer el grafo desde el wrapper de ORB-SLAM3 en `ovo/slam/` y degradar graciosamente para backends que no lo tengan (GT, etc.).

---

## Referencia de literatura

Los sistemas más relevantes para comparar:

**ConceptGraphs** (ICRA 2024) — el más similar a OVO en arquitectura. Usa CLIP+DINO con criterio AND (overlap Y similitud semántica). Incluye NMS 2D antes de levantar a 3D para controlar over-segmentation. Su punto débil es el mismo que el nuestro: loop closure sin mecanismo explícito.

**OVIR-3D** (ICCV 2023) — referencia directa para el problema de transitividad. Propone explícitamente componentes conexas como reemplazo del greedy, con ratio de volumen como gate previo. Directamente aplicable.

**HOV-SG** (RSS 2024) — grafo de escena jerárquico. Documenta bien el uso del ratio de volumen del AABB como gate duro. Tres gates en secuencia: overlap 3D → similitud semántica → consistencia de escala.

**Gaussian Grouping** (ECCV 2024) — resuelve el over-segmentation en 2D antes de levantar a 3D usando tracking DINO entre frames. La lección es que resolver la co-identidad en 2D es más limpio que fusionar en 3D a posteriori.

Lo que destaca de OVO respecto a estos sistemas es el tratamiento del loop closure (la corrección global del mapa ya es un trigger explícito para la fusión), pero la lógica de decisión dentro de ese trigger es más simple que en los sistemas comparables.
