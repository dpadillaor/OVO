# ADR-0003: `Image<T>` propio y mínimo en el dominio; OpenCV solo en adaptadores

- **Estado:** aceptado
- **Fecha:** 2026-10-01

## Contexto
El `PointPixelMatcher` necesita leer la profundidad medida en un píxel (`depth[v, u]`), y la semántica
necesitará mapas de segmentos y máscaras. C++ no tiene un tipo común de "array con forma y tipo"
como numpy en Python, así que hay que elegir uno.

Qué hace el Python:
- Las imágenes son arrays de numpy. OpenCV solo aparece al leer del disco (`datasets.py`: `imread`,
  `resize`, BGR→RGB, `undistort`); la profundidad se convierte a metros `float32` nada más leerla.
- Las imágenes viven un frame (más una cola corta de keyframes para extraer descriptores); el mapa
  guarda puntos y descriptores, nunca imágenes.
- La lógica semántica central solo indexa: leer `seg_map[v, u]` en puntos proyectados, áreas de
  máscara, OR de máscaras, pintar mapas de instancias. Sin morfología, contornos ni filtros.
- Las operaciones complejas (caja de la máscara, recorte, relleno, redimensionado, `/255`,
  normalización, HWC→CHW) están pegadas a cada modelo (CLIP, PE, SAM) o al dataset.

## Opciones
1. **`cv::Mat`** en el dominio.
2. **`Eigen::Array<T, Dynamic, Dynamic, RowMajor>`** como imagen (Eigen ya es dependencia).
3. **`Image<T>` propio y mínimo.**
4. **`std::mdspan`** (C++23): descartado por ahora, g++ 13 no lo trae (llega en g++ 14).

## Decisión
Opción 3. `Image<T>` en `core/common`, header-only (template):
- Dueño de sus datos (`std::vector<T>`), con `width()`, `height()`, `at(u, v)` y `data()`. Nada más.
- Memoria como numpy: contigua, fila a fila, canales intercalados (HWC). Así `cv::Mat`,
  `torch::from_blob`, ONNX o `Eigen::Map` pueden envolver el bloque sin copiarlo.
- Tipos previstos: `Image<float>` (profundidad en metros), `Image<std::int32_t>` (mapa de segmentos,
  -1 = sin máscara), `Image<std::uint8_t>` (máscaras, no `bool`), `Image<Rgb>` (color).
- OpenCV (o `stb_image`) solo en el adaptador del dataset y en los adaptadores de los modelos.
  Esa elección se decide al escribir el adaptador.

## Por qué
- Las fronteras con los modelos cuestan lo mismo con cualquier opción (envolver un puntero), y el
  preprocesado de los modelos crea una imagen nueva de todas formas. Elegir `cv::Mat` no ahorra nada.
- Descartado `cv::Mat`: tipo comprobado al ejecutar (`at<float>` sobre `uint16` lee basura), las
  copias comparten datos sin avisar (peligroso con threads), y mete OpenCV en un dominio que debe ser puro.
- Descartado Eigen: indexa `(fila, columna)` = `(v, u)`, al revés que `at(u, v)` (error silencioso);
  un `struct Rgb` como elemento exige `NumTraits`; ofrece álgebra lineal que no tiene sentido en una imagen.
  Sus operaciones vectorizadas siguen disponibles envolviendo el bloque con `Eigen::Map`.
- Template y no una clase por tipo: un solo código, tipo fijado al compilar.

## Consecuencias
- El dominio no depende de OpenCV. Los tests crean imágenes en memoria, sin ficheros.
- `std::vector<bool>` está prohibido para máscaras: guarda bits, no tiene un bloque de bytes envolvible.
- Regla anti-crecimiento: `Image<T>` no lleva operaciones de imagen (redimensionar, filtrar). Si pasa de
  ~100 líneas, replantear.
- Avisos para los adaptadores de modelos (del análisis del Python): el resize bilineal con antialias de
  torchvision no da los mismos píxeles que `cv2.resize`; el margen de la caja en
  `segment_utils.py:161-169` solo se limita por izquierda y arriba (el slicing de Python lo tapa), en
  C++ hay que recortar los cuatro lados.
- Si la semántica pasa a GPU, las imágenes vivirán en memoria de GPU: lo resuelve el adaptador CUDA.
