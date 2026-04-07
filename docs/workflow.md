# Workflow de desarrollo OVO

Documento vivo. Refleja cómo se organiza el trabajo en este proyecto: fases, herramientas, agentes, scripts y convenciones.

---

## Principios generales

- Todo trabajo repetitivo va en un script Python. Los comandos/skills orquestan, no ejecutan lógica.
- La metodología de implementación es siempre **TDD**: primero tests, luego implementación en ciclos cortos hasta que pasan.
- Cada tarea con código tiene su propio **worktree** y **rama** desde `develop` (salvo indicación contraria).
- El **session log** es el único canal de estado persistente entre sesiones. Lo escribe el agente de forma incremental.
- La orquestación entre instancias de Claude es **asíncrona por disco**: el orquestador lee logs, escribe instrucciones en un inbox si necesita comunicarse.
- La integración de subtareas paralelas la hace el humano de momento.

---

## Fases del ciclo de vida de una tarea

### FASE 0 — Discusión de scope *(HitL obligatorio)*

Conversación entre el usuario y Claude hasta acordar:
- ¿Una tarea o varias? ¿Con subtareas?
- Interfaz esperada / contrato entre módulos
- Riesgos técnicos conocidos

**Output obligatorio**: acuerdo escrito en `description.md` antes de pasar a fase 1. Sin esto la fase 1 se construye sobre acuerdos volátiles.

---

### FASE 1 — Creación de tareas *(script)*

Mecánico, sin discusión. Crea carpeta, archivos vacíos, actualiza índice.

```
python .agents_mapper/scripts/new_task.py "<título>"
```

Estado resultante: `PENDING PLANNING`

Archivos creados:
```
active_tasks/{id}_{slug}/
├── description.md
├── strategy.md       (vacío)
├── tests.md          (vacío)
├── session_log.md    (vacío)
└── findings/
```

---

### FASE 2 — Investigación *(agentes, opcional)*

Solo si hay código desconocido o interfaces poco claras. Se lanza con `/start-research-task`.

Agentes `file-context-extractor` en paralelo → `research-consolidator` → `research.md` en la carpeta de la tarea.

El output queda en disco. No se carga en el contexto principal.

Estado resultante: `RESEARCHING` → `PENDING PLANNING`

---

### FASE 3 — Planificación + criterios de verificación *(HitL obligatorio)*

Claude lee `description.md` y `research.md` (si existe) y propone:
- Estrategia de implementación → `strategy.md`
- Tests que definen "hecho" → `tests.md`

Discusión con el usuario hasta aprobación. Estas dos fases van juntas en la misma sesión — separararlas artificialmente genera fricción.

**Los tests en `tests.md` son la hoja de ruta del agente de implementación. No se improvisan tests durante la implementación.**

Estado resultante: `READY`

---

### FASE 4 — Activación *(script + skill)*

Crea el worktree y la rama. Sin discusión.

```
python .agents_mapper/scripts/start_task.py {id}
```

- Rama: `task/{id}-{slug}` desde `develop`
- Worktree: `.claude/worktrees/{slug}`
- Actualiza `_index.json` con `branch` y `worktree_path`

Estado resultante: `IN_PROGRESS`

---

### FASE 5 — Implementación *(instancia Claude independiente)*

Se lanza una instancia separada de Claude Code apuntando al worktree. No es un subagente — es un proceso independiente que sobrevive al cierre de la sesión principal.

El agente:
1. Lee `strategy.md` y `tests.md`
2. Trabaja en ciclos TDD: escribe test → falla → implementa mínimo → pasa → siguiente
3. Actualiza `session_log.md` incrementalmente (qué se hizo, en qué archivos, qué inesperado si lo hay)
4. Para cuando todos los tests de `tests.md` pasan

El criterio de "listo para revisión" es objetivo: **todos los tests definidos pasan**.

Estado resultante: `IN_REVIEW`

---

### FASE 6 — Revisión *(HitL obligatorio)*

El usuario revisa los cambios sabiendo que los tests ya pasan. Aprueba o pide correcciones.

---

### FASE 7 — Cierre *(script + skill)*

```
python .agents_mapper/scripts/complete_task.py {id}
```

- Merge worktree → rama → develop
- Limpieza del worktree
- Mueve carpeta a `completed_tasks/`
- Actualiza `_index.json`

Estado resultante: `COMPLETED`

---

## Estados de una tarea

```
PENDING PLANNING → RESEARCHING → PENDING PLANNING → READY → IN_PROGRESS → IN_REVIEW → COMPLETED
                                                  ↑
                                        (si no hay investigación,
                                         se salta directamente)
```

---

## Session log

Formato del entry en `session_log.md`:

```markdown
## YYYY-MM-DD HH:MM — {fase}
**Hecho:** X en `archivo:línea`, Y en `archivo:línea`
**Siguiente:** A, B
**Inesperado:** (solo si hay algo relevante)
```

Conciso. Sin relleno.

---

## Índice de tareas (_index.json)

Cada tarea activa incluye:

```json
{
  "id": "16",
  "title": "...",
  "status": "IN_PROGRESS",
  "branch": "task/16-slug",
  "worktree_path": ".claude/worktrees/slug"
}
```

---

## Convención de nombres

| Elemento | Convención | Ejemplo |
|----------|-----------|---------|
| Rama | `task/{id}-{slug}` | `task/16-instance-graphs` |
| Worktree | `.claude/worktrees/{slug}` | `.claude/worktrees/instance-graphs` |
| Carpeta tarea | `{id}_{slug}` | `16_instance_graphs` |

---

## Herramientas a crear

### Scripts Python (`.agents_mapper/scripts/`)

| Script | Estado | Función |
|--------|--------|---------|
| `new_task.py` | Rehacer | Crear tarea: carpeta, archivos, índice |
| `get_next_task_id.py` | Rehacer | Siguiente ID disponible |
| `get_task_info.py` | Rehacer | Devuelve JSON con info de una tarea |
| `start_task.py` | Nuevo | Crear worktree + rama desde develop |
| `complete_task.py` | Rehacer | Merge + cleanup + archivar |
| `log_session.py` | Nuevo | Append estructurado al session log |
| `task_status.py` | Nuevo | Estado legible de todas las tareas activas |

### Skills / Commands (`.claude/commands/`)

| Command | Estado | Función |
|---------|--------|---------|
| `/new-task` | Actualizar | Añadir paso de scope a description.md |
| `/start-research-task` | OK | Lanzar agentes de investigación |
| `/plan-task` | Nuevo | Cargar contexto para sesión de planificación HitL |
| `/start-task` | Nuevo | Activar tarea: script + lanzar instancia implementadora |
| `/resume-task` | Nuevo | Leer session log + strategy para retomar contexto |
| `/task-status` | Nuevo | Estado de tareas activas + último entry de cada log |
| `/complete-task` | Reescribir | Añadir merge + limpieza al cierre |

### Agentes (`.claude/agents/`)

| Agente | Estado | Función |
|--------|--------|---------|
| `file-context-extractor` | OK | Extrae contexto de archivos para research |
| `research-consolidator` | OK | Consolida findings en research.md |
| `tdd-implementor` | Nuevo | Implementa en worktree siguiendo ciclos TDD, actualiza session log |

---

## Pendiente por resolver

- Comunicación entre instancias Claude: verificar si `claude --resume <id>` u otro mecanismo permite enviar mensajes a una instancia en ejecución.
