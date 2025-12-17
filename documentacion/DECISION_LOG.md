# Registro de Decisiones del Proyecto - Fusión de Instancias con DINO

Este documento sirve como un registro de alto nivel y un índice para las decisiones y funcionalidades del proyecto.

## 1. Objetivo Principal

Mejorar el algoritmo de fusión de instancias 3D durante los cierres de bucle, utilizando descriptores DINO.

## 2. Metodología de Desarrollo

Se ha decidido adoptar un **Enfoque Híbrido de "Replay"**:
1.  **Grabar:** Ejecutar el pipeline una vez para grabar un log completo y realista de la salida del SLAM.
2.  **Reproducir:** Usar un simulador que reproduzca el log de forma determinista.
3.  **Desarrollar:** Implementar y probar nuevas funcionalidades en este entorno controlado y realista.

La documentación detallada de cada funcionalidad se registrará en ficheros separados, enlazados desde este documento.

## 3. Registro de Funcionalidades

### 01 - Implementación de Replay Logging
*   **Objetivo:** Implementar la fase de "Grabar" de nuestra metodología.
*   **Estado:** CANCELADO. (El sistema SLAM no está disponible, se adopta nueva estrategia).
*   **Log Detallado:** [./feature_logs/01_replay_logging.md](./feature_logs/01_replay_logging.md)

### 02 - Ground Truth Replay
*   **Objetivo:** Implementar un SLAM simulado (`GroundTruthSLAM`) que usa datos de ground truth para permitir el desarrollo sin un SLAM funcional.
*   **Estado:** COMPLETED.
*   **Log Detallado:** [./feature_logs/02_ground_truth_replay.md](./feature_logs/02_ground_truth_replay.md)

### 03 - DINO Integration
*   **Objetivo:** Preparar un módulo `DINOGenerator` capaz de cargar un modelo DINO y extraer descriptores de características para máscaras de segmentación dadas.
*   **Estado:** PENDING.
*   **Log Detallado:** [./feature_logs/03_dino_integration.md](./feature_logs/03_dino_integration.md)

### 04 - Trajectory Noise Simulation
*   **Objective:** Add functionality to `GroundTruthSLAM` to introduce configurable noise into the ground truth trajectory poses.
*   **Status:** COMPLETED. (Successfully resolved after fixing the relative motion calculation bug).
*   **Detailed Log:** [./feature_logs/04_trajectory_noise.md](./feature_logs/04_trajectory_noise.md)

### 05 - Trajectory Visualization
*   **Objetivo:** Crear un script para visualizar y comparar la trayectoria estimada con la de ground truth.
*   **Estado:** COMPLETED.
*   **Log Detallado:** [./feature_logs/05_trajectory_visualization.md](./feature_logs/05_trajectory_visualization.md)

### 06 - Point Cloud Visualization
*   **Objetivo:** Extender `visualize_trajectory.py` para mostrar simultáneamente la nube de puntos de ground truth y la generada en un experimento, permitiendo una comparación geométrica directa.
*   **Estado:** COMPLETED.
*   **Log Detallado:** [./feature_logs/06_point_cloud_visualization.md](./feature_logs/06_point_cloud_visualization.md)

### 07 - Loop Closure Geometric Correction
*   **Objetivo:** Implementar el "Método de Cálculo Directo" para simular una corrección geométrica realista durante los cierres de bucle, permitiendo que la fusión de instancias se pruebe en condiciones de deriva significativa.
*   **Estado:** CANCELLED. (Reemplazado por corrección global al final de la secuencia por feedback del tutor).
*   **Log Detallado:** [./feature_logs/07_loop_closure_geometric_correction.md](./feature_logs/07_loop_closure_geometric_correction.md)

### 08 - Global Geometric Correction
*   **Objetivo:** Implementar una corrección geométrica global al final de la secuencia, reemplazando todas las poses estimadas por GT y alineando la nube de puntos, para forzar una fusión masiva de instancias.
*   **Estado:** COMPLETED.
*   **Log Detallado:** [./feature_logs/08_global_geometric_correction.md](./feature_logs/08_global_geometric_correction.md)

### 09 - Experimentation Framework
*   **Objetivo:** Definir y establecer un Framework de Experimentación OVO robusto y escalable para gestionar múltiples experimentos, organizando configuraciones, ejecuciones, resultados y visualizaciones de manera sistemática.
*   **Estado:** IN_PROGRESS.
*   **Log Detallado:** [./feature_logs/09_experimentation_framework.md](./feature_logs/09_experimentation_framework.md)

### 10 - PE Integration
*   **Objetivo:** Integrar Perception Encoder (PE) como generador de embeddings para instancias 3D.
*   **Estado:** IN_PROGRESS.
*   **Log Detallado:** [./feature_logs/10_pe_integration.md](./feature_logs/10_pe_integration.md)


## 4. Próximo Paso

*   Continuar con la definición de la funcionalidad **09 - Experimentation Framework**.