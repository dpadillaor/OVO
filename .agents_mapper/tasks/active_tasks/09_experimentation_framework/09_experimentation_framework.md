# 09 - Experimentation Framework

## 1. Objetivo

Definir y establecer un Framework de Experimentación OVO robusto y escalable para gestionar múltiples experimentos, organizando configuraciones, ejecuciones, resultados y visualizaciones de manera sistemática.

## 2. Motivación

La necesidad de evaluar el rendimiento del sistema OVO bajo diversas condiciones (diferentes niveles de drift, distintas estrategias de fusión, múltiples datasets y escenas) requiere un enfoque estructurado. Sin un framework de experimentación, la gestión de la configuración, la ejecución repetitiva de experimentos, la recopilación y el análisis de resultados se volverían inmanejables, dificultando la comparación rigurosa y la extracción de conclusiones científicas válidas.

## 3. Estado Actual

IN_PROGRESS (Implementación del Orquestador y Manifiesto completada, pendiente ejecución real).

## 4. Conocimiento Adquirido: Estructura de Archivos y Flujo OVO (Sesión 05/12/2025)

Durante la sesión de análisis se ha desglosado el funcionamiento interno de `run_eval.py` y la estructura de sus outputs. Este conocimiento es la base para diseñar el framework.

### 4.1. Concepto de "Experimento"
Un experimento (`experiment_name`) en OVO es un conjunto de ejecuciones sobre N escenas bajo **una única configuración** (mismo SLAM, mismos parámetros de fusión, mismo modelo semántico).
*   Ejemplo: `experiment_name="FUSION_CLIP_DRIFT_LOW"` aplicado a `scenes=["office0", "office1"]`.

### 4.2. Flujo de Ejecución Estándar
El flujo completo se compone de tres pasos, que pueden ejecutarse juntos o separados:
1.  **Run (`--run`):** Ejecuta SLAM + OVO. Genera el mapa 3D y la trayectoria.
2.  **Segment (`--segment`):** Carga el mapa generado, clasifica las instancias finales y las proyecta sobre el Ground Truth (GT) para generar máscaras de evaluación.
3.  **Eval (`--eval`):** Compara las máscaras proyectadas con el GT oficial y calcula métricas (mIoU, Acc).

### 4.3. Mapa de Archivos de Salida
Ruta base: `data/output/{Dataset}/{Experiment_Name}/`

#### A. Resultados por Escena (Generados por `--run`)
Ubicación: `.../{Experiment_Name}/{Scene_Name}/` (Ej: `.../test1/office0/`)
*   **`config.yaml`**: Configuración exacta usada para esta escena.
*   **`estimated_c2w.npy`**: Archivo binario numpy con la trayectoria estimada de la cámara (poses 4x4). **"Por dónde pasó la cámara"**.
*   **`ovo_map.ckpt`**: Checkpoint maestro. Contiene:
    *   `map_params`: Nube de puntos 3D reconstruida (Geometría).
    *   `ovo_map_params`: Instancias 3D, sus descriptores (CLIP/DINO) y sus asociaciones (Semántica).
    *   **"Qué vio y construyó el sistema"**.
*   **`logger/`**: Logs detallados de rendimiento (`t_sam.log`, `vram.log`, etc.).

#### B. Resultados de Segmentación (Generados por `--segment`)
Ubicación: `.../{Experiment_Name}/instance_pred/`
*   **`{Scene_Name}.txt`**: Índice resumen de instancias detectadas. Cada línea indica: `Ruta_JSON Clase_ID Confianza`.
*   **`predicted_masks/*.json`**:
    *   **Definición Clave:** Cada JSON es un **mapeo binario** (RLE) de una instancia OVO sobre los puntos del **Ground Truth**.
    *   No es una imagen. Es una lista de selección que dice: "Los puntos X, Y, Z del archivo GT original pertenecen a esta instancia".
    *   Son necesarios para evaluar geométricamente la calidad de la segmentación sobre la "verdad".

Ubicación: `.../{Experiment_Name}/{dataset_eval}/` (Ej: `.../test1/replica/`)
*   **`{Scene_Name}.txt`**: Archivo crudo de etiquetas predichas para cada punto del GT. Input directo para `--eval`.
    *   **Proceso de Generación:** Para cada vértice de la malla de Ground Truth, se buscan los 5 puntos más cercanos en la nube de puntos 3D predicha por OVO. La etiqueta de clase más frecuente entre estos 5 puntos se asigna al vértice del GT (votación k-NN). Esto "rellena" la malla de Ground Truth con las predicciones semánticas de OVO.

#### C. Resultados de Evaluación (Generados por `--eval`)
Ubicación: `.../{Experiment_Name}/{dataset_eval}/` (Ej: `.../test1/replica/`)
*   **`statistics.txt`**: Reporte numérico final (IoU por clase, mIoU global, Accuracy).
*   **`confmat.png`**: Imagen de la matriz de confusión.
*   **`plot_iou_acc.png`**: Gráfico de barras de métricas por clase.

## 5. Convención de Nombres para Experimentos (argumento `--experiment_name`)

Para garantizar la organización, trazabilidad y facilidad de análisis automático, se ha definido una convención estricta para el argumento `--experiment_name` que se pasa a `run_eval.py`. Este nombre se convertirá en el nombre de la carpeta dentro de `data/output/{Dataset}/`.

**Formato General:**

```
[FECHA]_[SCENES_ID]_[SLAM_CONFIG]_[FUSION_CONFIG]_[ETIQUETA]
```

**Descripción de los Tokens:**

*   **`[FECHA]`**: `YYYYMMDD` (Año, Mes, Día). Ej: `20251210`.
*   **`[SCENES_ID]`**: Identificador de las escenas usadas. Si es una lista, el nombre del archivo sin extensión; si es una escena única, su nombre. **Sin guiones bajos (`_`)**.
*   **`[SLAM_CONFIG]`**:
    *   `GT`: Ground Truth puro.
    *   `ORBSLAM3`: Módulo ORB-SLAM3.
    *   `GTNoise-T<val>p<val>-R<val>p<val>`: GT con ruido. (Ej: `GTNoise-T0p1-R0p5`).
*   **`[FUSION_CONFIG]`**: `CLIP`, `DINO`, `Hybrid`.
*   **`[ETIQUETA]`**: Texto libre descriptivo **sin guiones bajos (`_`)**. (Ej: `Baseline`, `DriftTest`).

## 6. Implementación del Orquestador (`scripts/run_experiments_batch.py`)

Se ha implementado un script robusto basado en clases para automatizar la ejecución de experimentos.

### 6.1. Arquitectura
*   **`ExperimentRunner`**: Clase principal que gestiona el ciclo de vida de un experimento (`setup` -> `run` -> `cleanup`).
*   **`Dataclasses`**: Se utilizan `Manifest`, `Experiment`, `OVOConfigOverride` y `SLAMConfigOverride` para estructurar y tipar fuertemente la configuración leída del YAML.
*   **Manejo de Errores**: Uso de bloques `try...finally` para asegurar que los archivos de configuración originales se restauran siempre, incluso si el experimento falla o es interrumpido por el usuario (`Ctrl+C`).

### 6.2. Estrategia de Modificación de Configuración (Overrides)
El Orquestador modifica los archivos de configuración "in-place" usando una estrategia de backup y restauración:
1.  **Backup:** Crea copias `.bak` de `ovo.yaml` y del archivo de configuración del SLAM activo (ej: `replica.yaml`).
2.  **Inyección:**
    *   Lee los archivos YAML originales.
    *   Aplica los cambios definidos en el manifiesto.
    *   **Manejo de Anidamiento:** Para el ruido en `replica.yaml`, envuelve los parámetros en un diccionario `{"noise": ...}` para asegurar que `_update_recursive` actualice correctamente los valores anidados y no cree claves duplicadas en la raíz.
    *   Sobrescribe los archivos originales con la nueva configuración.
3.  **Ejecución:** Lanza `run_eval.py` como un subproceso.
4.  **Restauración:** Al finalizar, restaura los archivos originales desde los backups `.bak`.

### 6.3. Interfaz de Usuario (`scripts/experiments_manifest.yaml`)
Los experimentos se definen en un archivo YAML sencillo.
*   **`ovo_config`**: Permite cambiar el módulo SLAM (`slam: {slam_module: ...}`) y el método de fusión (`fusion_method: ...`).
*   **`slam_config`**: Permite inyectar ruido (`noise: {translation_noise_std: ...}`).

Este diseño desacopla la definición del experimento de la lógica de ejecución, permitiendo una experimentación flexible y reproducible.