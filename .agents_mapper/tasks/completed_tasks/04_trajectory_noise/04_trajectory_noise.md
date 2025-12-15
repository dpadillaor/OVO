# Feature Log: 04 - Trajectory Noise Simulation

**Objetivo:** Añadir una funcionalidad a `GroundTruthSLAM` para introducir ruido configurable en las poses de la trayectoria de ground truth, simulando la deriva acumulativa de un sistema SLAM real.

**Motivación:** La implementación actual de `GroundTruthSLAM` utiliza una trayectoria perfecta, lo cual es ideal para probar la lógica semántica de forma aislada. Sin embargo, para simular un entorno más realista y probar la robustez del sistema de fusión de instancias ante errores de SLAM, es necesario simular la deriva (drift) en la trayectoria. Esto también permitirá forzar y probar la lógica de cierre de bucles de una manera más controlada y agresiva.

## Diseño y Arquitectura

El objetivo es simular una trayectoria que se desvía de la verdad fundamental con el tiempo, imitando el "drift" de un sistema SLAM real. Para ello, corromperemos el *movimiento relativo* entre frames, no las poses absolutas.

### Lógica de Implementación Paso a Paso (para cada frame `k` > 0):

1.  **Obtenemos las Poses de Ground Truth (GT):**
    *   `Pose_GT_k`: La pose perfecta para el frame actual `k`.
    *   `Pose_GT_k-1`: La pose perfecta para el frame anterior `k-1`.

2.  **Calculamos el Movimiento Relativo Real:**
    *   Esta es la transformación que mueve la cámara de su posición en `k-1` a su posición en `k`.
    *   **Cálculo:** `Movimiento_Real = Inversa(Pose_GT_k-1) @ Pose_GT_k`
        *   `@` representa la multiplicación de matrices.
        *   `Inversa()` representa la matriz inversa de la transformación.

3.  **Creamos una "Perturbación" Aleatoria (Ruido):**
    *   Generamos un pequeño vector de traslación aleatorio `[dx, dy, dz]` a partir de una distribución Gaussiana con media 0 y desviación estándar `translation_noise_std`.
    *   Generamos una pequeña rotación aleatoria (ej., ángulos de Euler `[d_roll, d_pitch, d_yaw]`) a partir de una distribución Gaussiana con media 0 y desviación estándar `rotation_noise_std`.
    *   Convertimos estos valores aleatorios de traslación y rotación en una matriz de transformación 4x4, `Perturbacion`.

4.  **Calculamos el Movimiento Relativo "Ruidoso":**
    *   Aplicamos la perturbación al movimiento relativo real: `Movimiento_Ruidoso = Movimiento_Real @ Perturbacion`.

5.  **Calculamos la Nueva Pose "Ruidosa" (Acumulativa):**
    *   Obtenemos la *última pose ruidosa calculada*, `Pose_Ruidosa_k-1`.
        *   Para el primer frame (`k=0`), `Pose_Ruidosa_0` será igual a `Pose_GT_0` (sin ruido inicial).
    *   Aplicamos el movimiento relativo ruidoso a la pose ruidosa anterior: `Pose_Ruidosa_k = Pose_Ruidosa_k-1 @ Movimiento_Ruidoso`.
    *   Esta `Pose_Ruidosa_k` es la que se pasará al resto del sistema OVO y se guardará como `Pose_Ruidosa_anterior` para el siguiente frame.

### Configuración y Reproducibilidad

*   **Parámetros de Configuración:**
    *   `noise_enabled`: Booleano para activar/desactivar la funcionalidad de ruido.
    *   `translation_noise_std`: Desviación estándar para el ruido de traslación.
    *   `rotation_noise_std`: Desviación estándar para el ruido de rotación.
    *   `noise_seed`: Semilla entera para el generador de números aleatorios, asegurando la reproducibilidad del ruido.

*   **Generación Reproducible:** Se inicializará un generador de números aleatorios (`torch.Generator`) con `noise_seed` en `GroundTruthSLAM.__init__`. Todas las perturbaciones aleatorias se generarán utilizando este generador.

## Próximos Pasos
1.  Añadir los nuevos parámetros de configuración de ruido a `data/working/configs/slam/groundtruth/replica.yaml`.
2.  Modificar `GroundTruthSLAM.__init__` para leer los parámetros de ruido, inicializar el generador aleatorio y `self.last_noisy_c2w`.
3.  Modificar `GroundTruthSLAM.track_camera` para aplicar el ruido acumulativo a la pose de la cámara de forma reproducible.

## Refinamiento y Diagnóstico (2025-11-20)

### Problema Observado
Tras la implementación inicial, se observó que la trayectoria con ruido no mostraba una deriva (drift) suave y acumulativa, sino una vibración caótica de alta frecuencia (jitter) alrededor de la posición inicial. Esto indicaba que, aunque la funcionalidad estaba "completa", no producía el resultado físicamente realista esperado.

### Análisis y Evolución de la Hipótesis
1.  **Hipótesis Inicial:** Se sospechó que la lógica de acumulación en `_noisy_tracking` era incorrecta, y que el ruido se estaba aplicando a la pose absoluta del ground truth en cada frame en lugar de al movimiento relativo.

2.  **Revisión del Código:** Un análisis del código reveló que la hipótesis inicial era incorrecta. La lógica `self.c2w = self.last_noisy_c2w @ noisy_relative_motion` sí implementa correctamente la acumulación de error.

3.  **Nueva Hipótesis (La Relación Señal-Ruido):** La nueva conclusión fue que el problema no radicaba en la lógica, sino en los parámetros. Específicamente, en la relación entre la magnitud del movimiento real inter-frame (la "señal") y la magnitud del ruido inyectado (el "ruido"). En datasets de alta frecuencia de muestreo como Replica, el movimiento real entre fotogramas consecutivos es muy pequeño. Si la desviación estándar del ruido configurado es comparable o mayor que este movimiento, la trayectoria resultante es dominada por el paseo aleatorio del ruido, no por la señal del movimiento real. Se planteó la distinción entre **escala espacial** (conocida por el ground truth) y **escala dinámica** (el movimiento por frame), siendo esta última la relevante para el problema.

### Acción Tomada
Para verificar la nueva hipótesis, se decidió reducir drásticamente los parámetros de ruido en el fichero de configuración (`data/working/configs/slam/groundtruth/replica.yaml`) por un factor de 10x, cambiando:
*   `translation_noise_std`: de `0.001` a `0.0001`
*   `rotation_noise_std`: de `0.003` a `0.0003`

La expectativa es que, con esta reducción, la "señal" del movimiento real domine sobre el "ruido", y la trayectoria resultante muestre una deriva suave y progresiva, fiel a la forma de la trayectoria del ground truth. El siguiente paso es ejecutar la evaluación con estos nuevos parámetros para confirmar el resultado.

### Corrección del Diagnóstico (2025-11-20)

**Observación Clave:** Se ha identificado una inconsistencia crítica gracias a la observación de que el sistema no procesa todos los frames. La configuración del tracker (`track_every: 5`) indica que solo se procesa un fotograma de cada cinco del dataset de ground truth.

**Diagnóstico Final:** El análisis anterior, aunque correcto en sus principios sobre la relación señal/ruido, partía de una premisa falsa. El verdadero bug no reside en la magnitud del ruido ni en el modelo de rotación (ángulos de Euler), sino en una simple pero crucial **inconsistencia en los índices de los fotogramas**.

La función `_noisy_tracking` calculaba el movimiento relativo (`real_relative_motion`) entre el frame `k` y el `k-1`:
`prev_gt_c2w = self.trajectory[frame_id - 1].to(self.device)`

Sin embargo, debido al submuestreo, la pose ruidosa anterior (`self.last_noisy_c2w`) no correspondía al frame `k-1`, sino al último frame procesado, es decir, `k-5`. Al componer la pose de `k-5` con el movimiento de `k-1` a `k`, se estaba descartando el movimiento ocurrido en los 4 frames intermedios.

Este descarte sistemático del 80% del movimiento real es la causa directa del "encogimiento" de la trayectoria.

**Solución Propuesta:**
La solución es modificar `GroundTruthSLAM` para que guarde el índice del último frame procesado (`last_processed_frame_id`) y lo use para calcular el `real_relative_motion`, asegurando que el movimiento relativo abarque todo el intervalo desde el último procesamiento.

## Resolución (2025-11-20)

La solución propuesta se implementó con éxito, abordando dos puntos clave:
1.  **Cálculo Correcto del Movimiento Relativo:** Se modificó `GroundTruthSLAM` para que recuerde el `last_processed_frame_id`, asegurando que el `real_relative_motion` se calcula sobre el intervalo correcto, respetando la configuración de submuestreo (`track_every`).
2.  **Modelo de Ruido Robusto:** Simultáneamente, se reemplazó la generación de ruido de rotación basada en ángulos de Euler por un modelo basado en la representación de eje-ángulo (vector de rotación), implementando la fórmula de Rodrigues en `geometry_utils.py` para evitar artefactos como el "encogimiento".

Tras aplicar estos cambios, se ha verificado que la simulación de drift funciona como se esperaba. La trayectoria con ruido ahora sigue la forma de la trayectoria del ground truth mientras se desvía de forma suave y acumulativa. La magnitud del drift es controlable a través de los parámetros de configuración, y el comportamiento es físicamente realista.

**Esta funcionalidad se considera completa y exitosa.**