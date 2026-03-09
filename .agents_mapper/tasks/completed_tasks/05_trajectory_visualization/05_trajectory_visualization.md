# Feature Log: 05 - Trajectory Visualization

**Objetivo:** Crear un script (`visualize_trajectory.py`) para cargar, comparar y visualizar una trayectoria de ground truth (GT) contra una trayectoria estimada de una ejecución del experimento.

## Diseño y Arquitectura

1.  **Ubicación del Script:** El nuevo script se llamará `visualize_trajectory.py` y estará ubicado en el directorio raíz del proyecto.
2.  **Entradas (Argumentos):**
    *   El script aceptará como argumento principal la ruta a la carpeta de una ejecución (`run_path`), por ejemplo: `data/output/noisy_trajectory_test/Replica/office0`.
    *   A partir de esta ruta, el script derivará la ubicación de la trayectoria estimada y la correspondiente trayectoria GT.
3.  **Librerías:** Se utilizará `matplotlib` para generar una gráfica 2D (vista cenital) que muestre ambas trayectorias. Se usará `numpy` y `torch` para el manejo de los datos.
4.  **Lógica de Carga:**
    *   **Función `load_gt_trajectory`:** Se encargará de leer el archivo `traj.txt` y convertirlo en una lista de poses.
    *   **Función `load_estimated_trajectory`:** Se encargará de leer el archivo `estimated_c2w.npy` usando `torch.load()` y lo convertirá a un formato compatible.
5.  **Lógica de Visualización:**
    *   Una función principal extraerá las coordenadas `(x, z)` o `(x, y)` de cada pose en ambas trayectorias.
    *   Se generará una gráfica con `matplotlib` mostrando la trayectoria GT en un color (ej. verde) y la estimada in otro (ej. rojo).
    *   Se añadirá una leyenda y títulos para claridad.

## Registro de Progreso

*   **2025-11-20:** Se ha definido la necesidad de esta funcionalidad y se ha creado este documento de diseño.

## Próximos Pasos

1.  Crear el archivo `visualize_trajectory.py` con la estructura básica y el parseo de argumentos.
2.  Implementar la función `load_gt_trajectory`.
3.  Implementar la función `load_estimated_trajectory`.
4.  Implementar la lógica principal de ploteo con `matplotlib`.

---

## Debugging: No Visible Drift (2025-11-20)

**Problem:** The visualization script is running, but it shows no visible drift between the ground truth and the noisy trajectory.

**Hypothesis:** The noisy trajectory data is identical to the ground truth data.

**Plan:**
1.  **Examine Noise Application:** Investigate `ovo/slam/groundtruth_slam.py` to confirm noise is being correctly applied to poses.
2.  **Add Console Logs:** Add `print` statements to `groundtruth_slam.py` to log the original and noisy poses for comparison.
3.  **Verify Visualization Script:** Check `visualize_trajectory.py` to ensure it loads two distinct trajectory files.
