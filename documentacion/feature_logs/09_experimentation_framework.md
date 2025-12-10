# 09 - Experimentation Framework

## 1. Objetivo

Definir y establecer un Framework de Experimentación OVO robusto y escalable para gestionar múltiples experimentos, organizando configuraciones, ejecuciones, resultados y visualizaciones de manera sistemática.

## 2. Motivación

La necesidad de evaluar el rendimiento del sistema OVO bajo diversas condiciones (diferentes niveles de drift, distintas estrategias de fusión, múltiples datasets y escenas) requiere un enfoque estructurado. Sin un framework de experimentación, la gestión de la configuración, la ejecución repetitiva de experimentos, la recopilación y el análisis de resultados se volverían inmanejables, dificultando la comparación rigurosa y la extracción de conclusiones científicas válidas.

## 3. Estado Actual

IN_PROGRESS (Convención de nombres definida, framework de ejecución en diseño).

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

*   **`[FECHA]`**:
    *   Formato: `YYYYMMDD` (Año, Mes, Día).
    *   Ejemplo: `20251210`
    *   Propósito: Permite ordenar cronológicamente los experimentos y saber cuándo se ejecutó.

*   **`[SCENES_ID]`**:
    *   Identifica las escenas sobre las que se ejecutó el experimento.
    *   Si es una **sola escena**: Se usa directamente el nombre de la escena (ej: `office0`).
    *   Si es un **conjunto de escenas** (pasadas por `--scenes_list <archivo.txt>`): Se usa el nombre base del archivo `.txt` (sin extensión). (ej: si el archivo es `replica_offices.txt`, el ID es `replica_offices`).
    *   **Restricción:** El nombre de la escena o del archivo `.txt` (y, por tanto, el `[SCENES_ID]`) **NO debe contener guiones bajos (`_`)**. Puede usar guiones (`-`) o CamelCase.
    *   Ejemplos: `office0`, `replica-offices`

*   **`[SLAM_CONFIG]`**:
    *   Describe la configuración del módulo SLAM y el nivel de ruido/drift aplicado.
    *   **`GT`**: Ground Truth puro (sin ruido, trayectoria perfecta).
    *   **`ORBSLAM3`**: Usando el backend de ORB-SLAM3 real.
    *   **`GTNoise-T<val>p<val>-R<val>p<val>`**: Ground Truth con ruido sintético inyectado.
        *   `T<val>p<val>`: Desviación estándar de la traslación (en metros), con `p` sustituyendo al punto decimal.
        *   `R<val>p<val>`: Desviación estándar de la rotación (en grados), con `p` sustituyendo al punto decimal.
        *   Ejemplos: `GTNoise-T0p1-R0p5` (0.1m de translación, 0.5 grados de rotación), `GTNoise-T0p05-R0p2`.

*   **`[FUSION_CONFIG]`**:
    *   Indica la estrategia utilizada para la fusión de instancias.
    *   **`CLIP`**: Utiliza solo descriptores CLIP para la comparación de instancias.
    *   **`DINO`**: Utiliza solo descriptores DINO para la comparación de instancias (requiere implementación futura).
    *   **`Hybrid`**: Utiliza una combinación de descriptores CLIP y DINO (requiere implementación futura).

*   **`[ETIQUETA]`**:
    *   Texto descriptivo libre y corto para identificar la variante específica o el propósito del experimento.
    *   **Restricción:** No debe contener el carácter de guion bajo (`_`) para evitar problemas con el parsing del nombre. Puede usar guiones (`-`) o CamelCase.
    *   Ejemplos: `Baseline`, `LowDrift`, `FirstRun`, `TestUmbral`.

**Ejemplos de nombres de experimentos completos:**

1.  `20251210_office0_GT_CLIP_Baseline`
2.  `20251210_replica_offices_GTNoise-T0p1-R0p5_CLIP_DriftStudy`
3.  `20251211_scannet_val_ORBSLAM3_Hybrid_RealDataTest`
4.  `20251211_room0_GTNoise-T0p05-R0p2_DINO_FirstPass`

## 6. El Orquestador y la Gestión de Configuración

Para automatizar la ejecución de múltiples experimentos de manera organizada, se diseñará un script "Orquestador" (ej: `scripts/run_experiments_batch.py`). Este script tendrá las siguientes responsabilidades y seguirá la siguiente estrategia para gestionar la configuración de OVO:

### 6.1. Propósito del Orquestador
El Orquestador actuará como un "robot asistente" que automatiza los pasos manuales de:
1.  **Definición:** Leerá una lista de experimentos definidos en un archivo de manifiesto (ej: `experiments_manifest.yaml`). Este archivo servirá como una "lista de tareas" para el Orquestador, donde cada entrada describe un experimento con sus parámetros.
2.  **Preparación:** Para cada experimento, el Orquestador construirá el nombre de experimento (`--experiment_name`) siguiendo la convención definida en la Sección 5.
3.  **Ejecución:** Lanzará el script `run_eval.py` con los argumentos y configuraciones adecuadas.
4.  **Consolidación (futuro):** Una vez que todos los experimentos hayan sido ejecutados, el Orquestador podrá (en futuras versiones) recolectar y resumir los resultados para facilitar el análisis.

### 6.2. Estrategia de Modificación de Configuración
Para que `run_eval.py` ejecute cada experimento con su configuración específica sin modificar permanentemente los archivos de configuración originales del proyecto, el Orquestador empleará la estrategia de "Backup y Restauración" junto con la modificación programática de YAMLs.

1.  **Identificación de Archivos a Modificar:**
    *   `data/working/configs/ovo.yaml`: Controla la selección del módulo SLAM (`slam_module`) y la configuración general de OVO.
    *   `data/working/configs/slam/groundtruth/replica.yaml` (o el YAML del SLAM activo): Contiene parámetros específicos del backend SLAM, como `translation_noise_std` y `rotation_noise_std` para `GroundTruthSLAM`.

2.  **Procedimiento por Experimento:**
    Para cada experimento en la lista del manifiesto:
    *   **Backup:** El Orquestador creará copias de seguridad temporales de los archivos YAML originales (`ovo.yaml.bak`, `replica.yaml.bak`).
    *   **Modificación Programática:** Utilizando la librería `PyYAML` de Python, el Orquestador:
        *   Cargará el contenido de los archivos YAML originales en estructuras de datos de Python (diccionarios).
        *   Aplicará las modificaciones específicas de ese experimento (ej: cambiar `slam_module` a `groundtruth`, ajustar los valores de `translation_noise_std` y `rotation_noise_std`).
        *   Guardará las configuraciones modificadas de vuelta a los archivos originales (`ovo.yaml`, `slam/groundtruth/replica.yaml`). Se acepta que esta operación pueda eliminar comentarios existentes en los archivos, ya que los originales serán restaurados.
    *   **Lanzamiento de `run_eval.py`:** El Orquestador ejecutará `python run_eval.py` con el `--experiment_name` generado y los argumentos `--scenes` o `--scenes_list` correspondientes.
    *   **Restauración:** Una vez finalizada la ejecución de `run_eval.py` (o si se produce un fallo), el Orquestador restaurará inmediatamente los archivos YAML originales desde sus copias de seguridad, asegurando que el entorno quede limpio para el siguiente experimento o para el uso manual.

### 6.3. Propósito del `experiments_manifest.yaml`
Este archivo YAML será el "cerebro" donde se definirá cada experimento. Actuará como una interfaz declarativa para el usuario, permitiendo especificar:
*   El `dataset` sobre el que correr (ej: `Replica`).
*   Las `scenes` o `scenes_list` a usar.
*   Los overrides específicos para cada archivo de configuración (`ovo.yaml`, `slam/groundtruth/replica.yaml`).
*   La `etiqueta` descriptiva del experimento.

De esta manera, el usuario podrá definir, ejecutar y gestionar sus experimentos sin necesidad de modificar código Python directamente.
