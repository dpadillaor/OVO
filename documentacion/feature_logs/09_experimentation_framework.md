# 09 - Experimentation Framework

## 1. Objetivo

Definir y establecer un Framework de Experimentación OVO robusto y escalable para gestionar múltiples experimentos, organizando configuraciones, ejecuciones, resultados y visualizaciones de manera sistemática.

## 2. Motivación

La necesidad de evaluar el rendimiento del sistema OVO bajo diversas condiciones (diferentes niveles de drift, distintas estrategias de fusión, múltiples datasets y escenas) requiere un enfoque estructurado. Sin un framework de experimentación, la gestión de la configuración, la ejecución repetitiva de experimentos, la recopilación y el análisis de resultados se volverían inmanejables, dificultando la comparación rigurosa y la extracción de conclusiones científicas válidas.

## 3. Estado Actual

PENDING (En fase de diseño y discusión).

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

#### C. Resultados de Evaluación (Generados por `--eval`)
Ubicación: `.../{Experiment_Name}/{dataset_eval}/` (Ej: `.../test1/replica/`)
*   **`statistics.txt`**: Reporte numérico final (IoU por clase, mIoU global, Accuracy).
*   **`confmat.png`**: Imagen de la matriz de confusión.
*   **`plot_iou_acc.png`**: Gráfico de barras de métricas por clase.
