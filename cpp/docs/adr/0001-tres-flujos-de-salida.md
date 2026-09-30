# ADR-0001: Telemetría, estado del mapa y consultas como tres puertos separados

- **Estado:** aceptado
- **Fecha:** 2026-09-30

## Contexto
En OVO Python, la visualización (Rerun/Open3D), el logging (wandb, prints) y las consultas
open-vocabulary viajan mezclados por la misma cola multiproceso (`mpqueue`) y una pipe, con
`if self.stream and self.rerun_mode in (...)` repartidos por `ovomapping.py`. Queremos un sistema
de telemetría para entender qué pasa durante un experimento, sin acoplar el dominio a Rerun ni a wandb.

## Opciones
1. **Un único puerto de salida** para todo (eventos + mapa + consultas).
2. **Dos puertos:** observabilidad (telemetría + mapa) y consultas.
3. **Tres puertos separados por intención:** telemetría, estado del mapa, consultas.

## Decisión
Opción 3.

| Puerto | Dirección | Peso | Política |
|---|---|---|---|
| Telemetría (eventos, métricas, trazas) | salida | ligero | nunca bloquea; pérdida aceptable |
| Estado del mapa (puntos, instancias, poses) | salida | pesado | limitado (1 de cada N); pérdida aceptable |
| Consultas open-vocabulary | entrada | petición/respuesta | debe responder |

Un mismo adaptador (Rerun) puede implementar los tres.

## Por qué
- Políticas opuestas (no bloquear / limitar / responder): un puerto único obliga a ramificar por tipo de mensaje.
- Consumidores distintos: la evaluación solo necesita estado del mapa; wandb solo telemetría.
- Interfaces pequeñas (ISP). Fusionar puertos después es fácil; separar uno mezclado, no.

## Consecuencias
- El dominio no conoce Rerun, wandb ni ficheros.
- Pendiente: ADR de telemetría (formato de eventos, trazas por thread, Tracy) y de gestión de errores.
