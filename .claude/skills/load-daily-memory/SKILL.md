---
name: load-daily-memory
description: "Load and cherry-pick specific sections from Obsidian daily memory logs. Use when: you need context from a specific date; analyzing past decisions/actions; retrieving linked concepts; searching by date range or tags; gathering memory for task verification."
user-invocable: true
disable-model-invocation: false
allowed-tools:
  - Bash
---

# load-daily-memory — Cherry-pick Obsidian memory

Extract relevant information from daily memory logs at `/home/padidavid/Documentos/Obsidian Vault/daily/YYYY-MM-DD.md`.

The script parses **multiple sessions per day** (each session is indexed by timestamp) and lets you query them selectively.

---

## How to Use

Run via `uv`:
```bash
uv run .claude/scripts/load-daily-memory.py <date> [--options]
```

---

## Arguments

### Date (required)
- `today`, `yesterday`, `3-days-ago`, `week-ago`, `month-ago`  
- `2026-04-08` or `2026-04-08.md` (absolute date)

### Options (optional)

| Option | Values | Example |
|--------|--------|---------|
| `--sections` | Comma-separated list | `"What happened, Actions completed"` |
| `--tags` | Space-separated | `fusion slam reconstruction` |
| `--summary` | `full` \| `brief` \| `outline` | `--summary brief` |
| `--open-tasks` | (flag) | Extract only "Open tasks" from all sessions |
| `--sessions-summary` | (flag) | Show all sessions without timestamps |
| `--sessions` | Comma-separated indices | `0,2` (0-based) |

---

## Daily File Format (Expected)

The parser expects a daily file like `YYYY-MM-DD.md` with one or more session blocks.

### Session delimiter
Each session starts with a callout header line:
```text
> [!abstract]- Session End · HH:MM · `uuid`
```
or
```text
> [!abstract]- PreCompact · HH:MM · `uuid`
```

### Session title
Usually a bold line right after the delimiter:
```text
**My Session Title**
```

### Canonical sections (recognized)
- `**What happened**` or `**What happened**:`
- `**Decisions & reasoning**` or `**Decisions & reasoning**:`
- `**Actions completed**` or `**Actions completed**:`
- `**Open tasks**` or `**Open tasks**:`
- `**Connected to**` or `**Connected to**:`

Only these section names are parsed as content sections.

### Minimal example
```markdown
> [!abstract]- Session End · 16:43 · `...`
**Some session title**

**What happened**:
Summary text...

**Decisions & reasoning**
- Decision 1: rationale

**Actions completed**:
- path/to/file.py: change

**Open tasks**:
- Pending task

**Connected to**:
[[OVO]], [[Python]]

#tag1 #tag2
```

### Read-only behavior
This workflow is read-only. It only reads daily markdown files and prints output.
It does not create, modify, or delete files in the vault.

---

## Quick Examples

```bash
# All sessions from today
uv run .claude/scripts/load-daily-memory.py today

# Yesterday's open tasks only
uv run .claude/scripts/load-daily-memory.py yesterday --open-tasks

# Decisions & actions from 3 days ago (OVO-related, brief format)
uv run .claude/scripts/load-daily-memory.py 3-days-ago \
  --sections "Decisions & reasoning, Actions completed" \
  --tags OVO fusion --summary brief

# Session #0 and #2 from yesterday, full format
uv run .claude/scripts/load-daily-memory.py yesterday --sessions 0,2

# All sessions titled, no timestamps (compact)
uv run .claude/scripts/load-daily-memory.py today --sessions-summary

# Specific date (absolute)
uv run .claude/scripts/load-daily-memory.py 2026-04-05 --summary outline
```

---

## Output Format

Each session shows:
```
📅 16:43 | Session Title
──────────────────────────
**What happened**
...

**Decisions & reasoning**
...

**Actions completed**
...
==============================
```

With `--sessions-summary`: No timestamps, just titles  
With `--summary brief`: Only first line of each section  
With `--summary outline`: Bullet points only  

---

## Session Selection

Your daily notes may have multiple sessions. By default, all are shown.

To extract specific sessions:
```bash
# Sessions 0 and 1 (0-based indexing)
uv run .claude/scripts/load-daily-memory.py yesterday --sessions 0,1

# All sessions tagged with #fusion
uv run .claude/scripts/load-daily-memory.py yesterday --tags fusion

# All "Open tasks" across all sessions
uv run .claude/scripts/load-daily-memory.py yesterday --open-tasks
```

---

## Common Workflows

| Scenario | Command |
|----------|---------|
| What was open when I left? | `uv run ... yesterday --open-tasks --summary outline` |
| Why did we choose this approach? | `uv run ... 5-days-ago --tags fusion --sections "Decisions & reasoning"` |
| Review context for PR | `uv run ... yesterday --summary full` |
| Resume after a break | `uv run ... 3-days-ago --sections "What happened, Open tasks" --summary brief` |

---

## Implementation

Script lives in: `.claude/scripts/load-daily-memory.py`

**Key features:**
- Parses multiple sessions per file (by `> [!abstract]` delimiter)
- Extracts timestamps from each session header
- Handles canonical sections with or without trailing `:`
- Filters by section name, tags (`#tag`), or session index
- Adjusts output density (full/brief/outline)

**Dependencies:** Python stdlib only (pathlib, re, argparse, datetime)
