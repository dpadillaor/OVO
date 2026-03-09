# Feature Log: 02 - Ground Truth Replay

**Objetivo:** Implementar un sistema de SLAM simulado (`GroundTruthSLAM`) que utiliza datos de "ground truth" de los datasets (poses de cámara, imágenes de profundidad) para proporcionar un entorno de desarrollo determinista y controlado para OVO.

**Motivación:** El sistema SLAM principal (ORB-SLAM) no está disponible actualmente. Esta estrategia nos permite desatascar el desarrollo y centrarnos en la lógica de fusión semántica (el objetivo principal) sin depender de un SLAM funcional. Esta decisión sustituye a la estrategia original de "Replay Logging" (ver `01_replay_logging.md`).

## Diseño y Arquitectura

La implementación seguirá un enfoque coherente con la estructura del proyecto existente:

1.  **Clase Base:** La nueva clase `GroundTruthSLAM` heredará de `ovo.slam.vanilla_mapper.VanillaMapper`. Esto nos permite reutilizar toda la lógica existente para la creación y gestión de nubes de puntos a partir de imágenes de profundidad.

2.  **Estructura Lógica:** La clase imitará el esqueleto lógico de `ovo.slam.orbslam2.WrapperORBSLAM2`, actuando como un orquestador que decide *cuándo* mapear y *cuándo* señalar un cierre de bucle.

3.  **Simulación de KeyFrames:** No se procesarán todos los frames. `GroundTruthSLAM` designará un frame como "KeyFrame" solo si la pose de la cámara ha superado un umbral de distancia o rotación desde el último KeyFrame. Cuando esto ocurra, se llamará al método `super().map()` (heredado de `VanillaMapper`) para generar los puntos 3D.

4.  **Simulación de Cierre de Bucle:** Se implementará una función que compruebe la proximidad de la pose del KeyFrame actual con las poses de todos los KeyFrames anteriores. Si la distancia es menor a un umbral, se activará la señal `last_big_change_id`, que el sistema OVO utiliza para disparar la lógica de fusión de instancias. No se realizará ninguna optimización de mapa (fusión geométrica), ya que las poses de ground truth son perfectas. Esto crea un escenario de prueba ideal y aislado para la fusión semántica.

## Fichero de Implementación
`ovo/slam/groundtruth_slam.py`

## Registro de Progreso

### 2025-11-18 - Inicio de Implementación
*   Se ha iniciado la implementación de la clase `GroundTruthSLAM` en `ovo/slam/groundtruth_slam.py`.
*   El primer paso será cargar los datos de trayectoria del ground truth en el método `__init__`.

### 2025-11-18 - Implementación Completada
*   Se ha completado la implementación inicial de la clase `GroundTruthSLAM`.
*   **`__init__`**: Carga la trayectoria de ground truth (`traj.txt`) y los umbrales desde la configuración.
*   **`track_camera`**: Recupera la pose de la cámara para cada frame.
*   **`_is_new_keyframe`**: Implementada la lógica para seleccionar KeyFrames basada en umbrales de distancia y rotación.
*   **`_check_for_loop_closure`**: Implementada la lógica para simular cierres de bucle basados en la proximidad con KeyFrames anteriores.
### 2025-11-18 - Detección de Cierre de Bucle Deshabilitada Temporalmente
*   Por instrucción del usuario, la detección de cierres de bucle en `GroundTruthSLAM` ha sido deshabilitada temporalmente.
*   La llamada a `_check_for_loop_closure` en el método `map` ha sido comentada y se ha añadido un `TODO` para su futura habilitación.
*   Esto asegura que la lógica de fusión semántica basada en cierres de bucle no se active por ahora, permitiendo probar el resto del sistema.

### 2025-11-18 - Resolución de Errores y Verificación Exitosa
*   **Error de Sintaxis YAML:** Se corrigió un error de sintaxis en `data/working/configs/ovo.yaml` introducido durante la configuración inicial.
*   **Fichero de Configuración SLAM Faltante:** Se creó el fichero `data/working/configs/slam/groundtruth/replica.yaml` para proporcionar la configuración específica del módulo `GroundTruthSLAM` para el dataset `Replica`.
*   **`AttributeError` (`_config` a `config`):** Se corrigió el acceso a la configuración en `ovo/slam/groundtruth_slam.py`, cambiando `self._config` por `self.config`.
*   **`KeyError` (`scene_name`):** Se ajustó la forma en que `scene_name` se accede en `ovo/slam/groundtruth_slam.py`, de `self.config["scene_name"]` a `self.config["data"]["scene_name"]`, para reflejar la estructura de configuración.
*   **`FileNotFoundError` (`traj.txt` - Sensibilidad a Mayúsculas/Minúsculas):** Se identificó que el `dataset_name` en la configuración (`replica.yaml`) estaba en minúsculas, mientras que la carpeta del dataset (`Replica`) usaba mayúsculas. Se discutió una solución programática (`.capitalize()`) para manejar esto de forma robusta, y el usuario confirmó que el problema se resolvió, permitiendo que el fichero `traj.txt` fuera encontrado.
*   **Verificación Exitosa:** Se ejecutó el script `run_eval.py` con la configuración de `GroundTruthSLAM` para la escena `office2` del dataset `Replica`, y la ejecución finalizó con éxito, produciendo métricas de evaluación. Esto confirma que el entorno de `GroundTruthSLAM` está configurado y funcionando correctamente.