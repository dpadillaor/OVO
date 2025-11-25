# Feature Log: 03 - DINO Integration

**Objetivo:** Preparar un módulo `DINOGenerator` capaz de cargar un modelo DINO (específicamente DINOv2) y extraer descriptores de características para máscaras de segmentación dadas. El objetivo inmediato no es la integración completa en la lógica de fusión, sino tener el generador listo y funcional.

## Diseño y Arquitectura

1.  **Clase `DINOGenerator`:** Se creará una nueva clase en `ovo/entities/dino_generator.py`, siguiendo la estructura de `CLIPGenerator`.
2.  **Carga del Modelo:** El modelo DINOv2 se cargará desde `torch.hub`.
3.  **Extracción de Características:** Para una máscara dada, la estrategia será:
    *   Recortar la imagen original a la caja delimitadora (bounding box) de la máscara.
    *   Redimensionar la imagen recortada al tamaño esperado por el modelo DINO.
    *   Pasar la imagen procesada a través del modelo para obtener el descriptor.
4.  **Integración en `OVO`:** `DINOGenerator` se instanciará en la clase `OVO`. Se creará un método `_extract_dino` para orquestar la extracción y se llamará desde `_compute_semantic_info`.

## Registro de Progreso

### 2025-11-18 - Inicio de la Preparación del Módulo
*   **Análisis:** Se ha analizado la estructura de `CLIPGenerator`, `OVO` y `Instance3D` para definir un plan de integración.
*   **Decisión:** Por instrucción del usuario, se ha decidido no modificar `Instance3D` por ahora. El foco está en crear un `DINOGenerator` funcional.
*   **Creación del Esqueleto:** Se ha creado el fichero `ovo/entities/dino_generator.py` con una clase `DINOGenerator` que contiene la estructura básica y placeholders para la carga del modelo y la extracción de características.

## Próximos Pasos
1.  Implementar la carga real del modelo DINOv2 en `DINOGenerator.__init__` usando `torch.hub`.
2.  Implementar la lógica de extracción de características en `DINOGenerator.extract_dino`, incluyendo el preprocesamiento de la imagen (recorte y redimensión).
3.  Integrar la instanciación de `DINOGenerator` en `OVO.__init__`.
4.  Añadir los métodos `_extract_dino` y `_update_matched_objects_dino` en la clase `OVO` para gestionar la extracción y el almacenamiento temporal de los descriptores DINO.

---

### ¿Cómo funciona DINO?

**DINO (Self-DIstillation with NO labels)** es un método de **aprendizaje auto-supervisado** para entrenar *Vision Transformers (ViT)*. La idea principal es enseñar a un modelo a aprender características visuales potentes de imágenes sin necesidad de etiquetas humanas (como "gato", "perro", "coche").

Funciona con una arquitectura **estudiante-profesor**:

1.  **Dos Modelos Idénticos:** Se usan dos redes neuronales (ViTs) con la misma arquitectura: un "estudiante" y un "profesor".
2.  **Augmentación de Vistas:** Se toma una imagen y se crean múltiples versiones de ella con diferentes "aumentos" (recortes, cambios de color, etc.). Unas vistas son "locales" (pequeños recortes) y otras "globales" (recortes grandes).
3.  **Objetivo del Estudiante:** Todas las vistas se pasan al **estudiante**. Las vistas globales también se pasan al **profesor**. El objetivo del estudiante es que su salida (su "entendimiento" de las vistas) sea lo más parecida posible a la salida del profesor.
4.  **El Profesor se Actualiza Lentamente:** La clave es que el profesor no se entrena directamente. Sus pesos se actualizan como una media móvil exponencial (EMA) de los pesos del estudiante. Esto hace que el profesor sea una versión más estable y promediada del estudiante a lo largo del tiempo.

Este proceso de "auto-destilación" fuerza al estudiante a aprender características visuales muy robustas y generales. Una de las propiedades más interesantes que emerge es que los mapas de atención del modelo aprendido con DINO son capaces de **segmentar objetos en una imagen de forma muy precisa**, sin haber sido entrenado explícitamente para ello.

### Versiones Existentes

1.  **DINO (o DINOv1):** Es la versión original. Demostró la viabilidad del método y produjo características de alta calidad que funcionaban bien para tareas como clasificación y búsqueda de imágenes similares.

2.  **DINOv2:** Es la segunda versión, mucho más potente. Los autores lo entrenaron con un conjunto de datos masivo y curado de 142 millones de imágenes. Las características que aprende DINOv2 son tan buenas que se consideran de "propósito general" y pueden usarse directamente para muchas tareas (como la nuestra) sin necesidad de re-entrenamiento (fine-tuning). **Esta es la versión que deberíamos usar.**

### ¿Cómo se descarga e instala en este proyecto?

La mejor parte es que **no requiere una instalación compleja**. DINOv2 se puede cargar directamente desde **PyTorch Hub**.

-   **Dependencias:** Solo necesitas tener `torch` y `torchvision` instalados, lo cual ya está en nuestro entorno de Conda (`ovo`).
-   **Descarga y Carga:** Se hace con una simple llamada en Python. Por ejemplo, para cargar el modelo "ViT-Giant":

    ```python
    import torch
    dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
    ```

    Existen varios tamaños de modelo (pequeño, base, grande, gigante), lo que nos da flexibilidad para elegir entre velocidad y rendimiento.

### Papers Relevantes

Para que puedas profundizar, aquí tienes los enlaces a los papers oficiales en ArXiv:

1.  **DINOv1:** "[Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)"
2.  **DINOv2:** "[DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)"