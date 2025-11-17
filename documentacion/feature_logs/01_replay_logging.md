# Feature Log: 01 - Implementación de Replay Logging

**Objetivo:** Modificar `scripts/extract_slam_baseline.py` para ejecutar el pipeline de OVO y grabar un log detallado de la salida del SLAM. Este log es el pilar de nuestra metodología de "Replay Híbrido".

## Datos a Registrar por Frame:
*   Pose de la cámara (`estimated_c2w`).
*   La nube de puntos del mapa (`map_data`).
*   La señal de cierre de bucle (`map_updated`).
*   Los keyframes (`kfs`) relevantes en un cierre de bucle.

---
*Este log se actualizará a medida que se avance en la implementación.*
