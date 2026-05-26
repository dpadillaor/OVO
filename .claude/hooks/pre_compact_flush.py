import sys
import json
import os
import re
import subprocess
import traceback
from pathlib import Path
from datetime import datetime, date

# --- CONFIGURATION ---
# Use the user's home Documents Obsidian Vault (migrated hooks come from another machine)
# Support both Spanish 'Documentos' and English 'Documents'. Remove accidental trailing '$'.
_VAULT_CANDIDATES = [
    Path.home() / "Documentos" / "Obsidian Vault",
    Path.home() / "Documents" / "Obsidian Vault",
]
# Pick the first existing candidate, otherwise default to the Spanish path (first candidate)
VAULT = next((p for p in _VAULT_CANDIDATES if p.exists()), _VAULT_CANDIDATES[0])
SCRIPT = Path(__file__).resolve()
LOG_FILE = SCRIPT.parent.parent / "logs" / "hooks_debug.log"
LOG_SOURCE = "PRECOMPACT"
import shutil

# Gemini CLI: prefer an env override, fall back to a system lookup. This makes the
# hooks portable between Windows (where the original path lived) and Unix systems.
GEMINI_CMD = os.environ.get("GEMINI_CMD", "gemini")
_which = shutil.which(GEMINI_CMD)
if _which:
    GEMINI_CMD = _which
GEMINI_MODEL = "gemini-3.1-pro-preview"
GEMINI_TMP_DIR = SCRIPT.parent.parent / "tmp"

REQUIRED_SECTIONS = (
    "**What happened**",
    "**Decisions & reasoning**",
    "**Actions completed**",
    "**Open tasks**",
    "**Connected to**",
)

# --- PROMPT DESIGN ---
# Instructions + transcript are written together into a temp file
# and passed to Gemini via @file reference — no shell escaping issues,
# no stdin/arg conflict, no size limits.
GEMINI_PROMPT_HEADER = r"""You are an expert conversation analyst for an Obsidian-based second brain. Your job is to deeply understand each conversation and extract the most important information for a daily log entry. Prioritize clear synthesis of key ideas, decisions, completed actions, and open tasks. Use Obsidian wikilinks ([[...]]), when relevant, to preserve traceability between notes.

STRICT FORMAT RULES:
1. DO NOT USE MARKDOWN HEADINGS (#, ##, ###). They break the Obsidian callout.
2. Use **bold text** for section titles.
3. Start the summary with a bold title: **[One-line title describing the session]**
4. Use [[wikilinks]] ONLY for durable concepts: tools, technologies, projects, people. NEVER wikilink file paths — file changes are already tracked in git.
5. Include at least 3 #tags at the very end.
6. Write in the SAME LANGUAGE as the conversation (Spanish/English).
7. Be concise but thorough. Document every file change and key decision.
8. You must include ALL the REQUIRED SECTIONS listed below. If any section is not applicable, include it with a note like "N/A" or "None".
9. DO NOT include your own reasoning process, plans, or first-person meta text (examples forbidden: "I will...", "I'll...", "I think...").

REQUIRED SECTIONS (Use EXACTLY these names):

**What happened**
[Summary of the session goal and outcome in 2-4 sentences]

**Decisions & reasoning**
- [Decision]: [Reasoning]

**Actions completed**
- `file_path`: [Description of change]

**Open tasks**
- [Unfinished task]

**Connected to**
[[ProjectName]], [[Concept]], [[Technology]]

#tag1 #tag2 #tag3

---
EXAMPLE (IN SPANISH):

**Refactorización de hooks y mejora de logs**

**What happened**
Se ha mejorado la robustez de los hooks de Obsidian mediante la implementación de un proceso trabajador (worker) independiente. Esto evita que los hooks se cierren prematuramente y asegura que los logs se escriban correctamente.

**Decisions & reasoning**
- Usar `subprocess.Popen` con flags de proceso despegado: garantiza que el script sobreviva al cierre de la sesión de Claude/Gemini.
- Implementar validación estricta de formato: evita que notas mal formadas ensucien el vault.

**Actions completed**
- `.claude/hooks/session_end_flush.py`: implementación del patrón worker y mejora de prompts.
- `.claude/hooks/pre_compact_flush.py`: sincronización con la lógica de fin de sesión.
- `.claude/logs/hooks_debug.log`: añadido registro detallado de errores de Gemini.

**Open tasks**
- Probar el hook de PreCompact manualmente.

**Connected to**
[[Python]], [[Obsidian]], [[Gemini CLI]], [[Hooks]]

#refactor #decision #security

---
NOW PROCESS THE TRANSCRIPT BELOW. Output ONLY the formatted note. Do not output analysis or internal reasoning.

=== TRANSCRIPT BEGIN ===
"""

GEMINI_PROMPT_FOOTER = "\n=== TRANSCRIPT END ===\n"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg: str):
    """Safe logging to a local file."""
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{ts}] [{LOG_SOURCE}] {msg}\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Transcript loading
# ---------------------------------------------------------------------------

def load_transcript(transcript_path: str) -> str:
    """Read JSONL transcript and keep messages since the last compaction boundary."""
    path = Path(transcript_path)
    if not path.exists():
        return ""

    messages = []
    last_compact_idx = -1

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry_type = entry.get("type", "")

            # Boundary detection
            if entry_type == "system" and entry.get("subtype") == "compact_boundary":
                last_compact_idx = len(messages)
                continue

            if entry_type not in ("user", "assistant"):
                continue
            if entry.get("isCompactSummary", False):
                last_compact_idx = len(messages)
                continue
            if entry.get("isMeta", False):
                continue

            message = entry.get("message", {})
            content = message.get("content", "")

            if isinstance(content, list):
                text_parts = [
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                content = " ".join(text_parts).strip()

            if not content:
                continue
            if "<local-command" in content or "<command-name>" in content:
                continue

            label = "Human" if entry_type == "user" else "Claude"
            messages.append(f"[{label}]: {content}")

    if last_compact_idx > 0:
        messages = messages[last_compact_idx:]

    return "\n\n".join(messages)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def is_valid_summary(summary: str) -> bool:
    """Basic structural validation of the generated summary."""
    if not summary or not summary.startswith("**"):
        return False
    for section in REQUIRED_SECTIONS:
        if section not in summary:
            return False
    if not re.search(r"(?m)^#\w+", summary):
        return False
    return True


# ---------------------------------------------------------------------------
# Gemini invocation
# ---------------------------------------------------------------------------

def extract_with_gemini(conversation: str) -> str:
    """
    Invoke Gemini CLI to process the transcript.

    - shell=False          → no Windows shell escaping issues
    - @file reference      → single unambiguous input, no arg length limits
    - tmp in .claude/tmp   → within workspace, accepted by Gemini's path policy
    - --output-format json → clean .response extraction
    - --approval-mode auto_edit → no interactive prompts in headless mode
    - stdin=DEVNULL        → no ambiguity about input source
    """
    log(f"Calling Gemini CLI... ({len(conversation)} chars)")

    tmp_path = None
    # Allow overriding the timeout from the environment for slow connections or large prompts
    try:
        GEMINI_TIMEOUT = int(os.environ.get("GEMINI_TIMEOUT", "180"))
    except Exception:
        GEMINI_TIMEOUT = 180
    try:
        full_input = GEMINI_PROMPT_HEADER + conversation + GEMINI_PROMPT_FOOTER

        GEMINI_TMP_DIR.mkdir(parents=True, exist_ok=True)
        tmp_file = GEMINI_TMP_DIR / f"gemini_hook_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.txt"
        with open(tmp_file, "w", encoding="utf-8") as tmp:
            tmp.write(full_input)
            tmp_path = str(tmp_file)

        log(f"Temp file written: {tmp_path} ({len(full_input)} chars)")

        tmp_path_for_prompt = tmp_file.resolve().as_posix()
        prompt_arg = f"Read and process the instructions and transcript in @{tmp_path_for_prompt}"

        cmd = [
            GEMINI_CMD,
            "-m", GEMINI_MODEL,
            "-p", prompt_arg,
            "--output-format", "json",
            "--approval-mode", "auto_edit",
        ]

        # Log the full command for debugging; the prompt argument may be long so we also
        # include the temp file path separately.
        log(f"Running command: {' '.join(cmd)}")
        log(f"Prompt file for Gemini: {tmp_path_for_prompt}")

        # On Windows we can hide the console window using CREATE_NO_WINDOW.
        # On Unix-like systems, creationflags is not supported; omit it.
        popen_kwargs = dict(
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
        )
        if os.name == "nt":
            CREATE_NO_WINDOW = 0x08000000
            popen_kwargs["creationflags"] = CREATE_NO_WINDOW

        process = subprocess.Popen(cmd, **popen_kwargs)
        try:
            stdout, stderr = process.communicate(timeout=GEMINI_TIMEOUT)
        except subprocess.TimeoutExpired:
            log(f"Gemini TIMEOUT ({GEMINI_TIMEOUT}s)")
            try:
                process.kill()
            except Exception:
                pass
            try:
                stdout, stderr = process.communicate(timeout=5)
            except Exception:
                stdout, stderr = "", ""
            # Log partial outputs to help debugging
            log(f"Gemini stdout (partial): {str(stdout)[:2000]}")
            log(f"Gemini stderr (partial): {str(stderr)[:2000]}")
            return "[Gemini error]: timeout"

        log(f"Gemini rc={process.returncode}, stdout_len={len(stdout)}, stderr_len={len(stderr)}")

        if process.returncode != 0:
            log(f"Gemini stderr: {stderr[:1000]}")
            return f"[Gemini error]: {stderr[:500]}"

        raw = stdout.strip()
        try:
            data = json.loads(raw)
            response = data.get("response", "").strip()
            if response:
                return response
            log(f"JSON parsed but .response empty. Keys: {list(data.keys())}")
            return f"[Gemini error]: empty response field. Full JSON: {raw[:500]}"
        except json.JSONDecodeError:
            log(f"JSON parse failed. Raw output: {raw[:500]}")
            lines = raw.splitlines()
            lines = [
                l for l in lines
                if not l.startswith("Loaded cached")
                and not l.startswith("✓")
                and not l.startswith("◆")
                and l.strip()
            ]
            cleaned = "\n".join(lines).strip()
            if cleaned:
                return cleaned
            return f"[Gemini error]: unparseable output: {raw[:300]}"

    except Exception as e:
        log(f"Gemini EXCEPTION: {e}\n{traceback.format_exc()}")
        return f"[Gemini error]: {e}"
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
                log(f"Temp file removed: {tmp_path}")
            except Exception as cleanup_err:
                log(f"Could not remove temp file {tmp_path}: {cleanup_err}")


# ---------------------------------------------------------------------------
# Vault writing
# ---------------------------------------------------------------------------

def write_to_vault(summary: str, session_id: str):
    """Append the summary to the Obsidian vault's daily log."""
    if not VAULT.exists():
        log(f"ERROR: Vault not found: {VAULT}")
        return
    today_log = VAULT / "daily" / f"{date.today()}.md"
    today_log.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%H:%M")
    entry = (
        f"\n\n---\n\n"
        f"> [!abstract]- PreCompact · {timestamp} · `{session_id}`\n"
        f"{summary}\n"
    )
    with open(today_log, "a", encoding="utf-8") as f:
        f.write(entry)
    log(f"Vault written: {today_log}")


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def debug_dump(transcript_path: str):
    """Print the exact cleaned transcript that would be sent to Gemini."""
    conversation = load_transcript(transcript_path)
    if not conversation:
        print("[debug-dump] Empty conversation after filtering.")
        return
    print(f"[debug-dump] chars={len(conversation)}")
    print("=== GEMINI INPUT BEGIN ===")
    print(conversation)
    print("=== GEMINI INPUT END ===")


def debug_with_prompt(transcript_path: str):
    """Print the full content that would be written to the temp file."""
    conversation = load_transcript(transcript_path)
    if not conversation:
        print("[debug-with-prompt] Empty conversation after filtering.")
        return
    full_input = GEMINI_PROMPT_HEADER + conversation + GEMINI_PROMPT_FOOTER
    print(f"[debug-with-prompt] model={GEMINI_MODEL}, "
          f"transcript={len(conversation)} chars, "
          f"total_to_gemini={len(full_input)} chars")
    print("=== FULL GEMINI FILE BEGIN ===")
    print(full_input)
    print("=== FULL GEMINI FILE END ===")


def debug_run(transcript_path: str):
    """Run the full Gemini call and print the raw output without writing to vault."""
    conversation = load_transcript(transcript_path)
    if not conversation:
        print("[debug-run] Empty conversation after filtering.")
        return
    print(f"[debug-run] Sending {len(conversation)} chars to Gemini...")
    summary = extract_with_gemini(conversation)
    print("=== GEMINI OUTPUT BEGIN ===")
    print(summary)
    print("=== GEMINI OUTPUT END ===")
    print(f"\n[debug-run] Valid: {is_valid_summary(summary)}")


# ---------------------------------------------------------------------------
# Hook entry point & worker
# ---------------------------------------------------------------------------

def hook_entry():
    """Main entry point for the hook, spawns a detached worker process."""
    raw_stdin = sys.stdin.read()
    try:
        data = json.loads(raw_stdin)
    except Exception:
        try:
            data = json.loads(raw_stdin.replace("\\", "/"))
        except Exception:
            sys.exit(0)

    transcript_path = data.get("transcript_path", "")
    session_id = data.get("session_id", "unknown")

    if not transcript_path:
        sys.exit(0)

    try:
        # Use platform-appropriate flags when spawning a detached worker.
        popen_args = [sys.executable, str(SCRIPT), "--worker", transcript_path, session_id]
        popen_kwargs = dict(close_fds=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.name == "nt":
            # Windows: create a new process group and hide the window
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            CREATE_NO_WINDOW = 0x08000000
            popen_kwargs["creationflags"] = CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
        else:
            # Unix: start a new session so the child is detached from the parent
            popen_kwargs["start_new_session"] = True

        subprocess.Popen(popen_args, **popen_kwargs)
        log(f"PreCompact hook spawned worker. session={session_id}")
    except Exception:
        log(f"Failed to spawn worker for session={session_id}")
    sys.exit(0)


def worker(transcript_path: str, session_id: str):
    """Background process that handles processing and vault update."""
    try:
        log(f"PreCompact worker started. session={session_id}")
        conversation = load_transcript(transcript_path)

        if not conversation:
            log("PreCompact empty conversation, exiting.")
            return

        summary = extract_with_gemini(conversation)
        is_valid = is_valid_summary(summary)
        log(f"PreCompact validation result: {is_valid}")

        if summary and not summary.startswith("[Gemini error]") and is_valid:
            log("PreCompact writing to vault...")
            write_to_vault(summary, session_id)
            log("PreCompact finished. OK.")
        else:
            log(f"PreCompact Gemini failed/empty/invalid format. Summary length: {len(summary)}")
            if summary:
                log(f"Full output was:\n{summary}")
    except Exception:
        log(f"PRECOMPACT WORKER EXCEPTION:\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--debug-dump":
        debug_dump(sys.argv[2])
    elif len(sys.argv) >= 3 and sys.argv[1] == "--debug-with-prompt":
        debug_with_prompt(sys.argv[2])
    elif len(sys.argv) >= 3 and sys.argv[1] == "--debug-run":
        debug_run(sys.argv[2])
    elif len(sys.argv) >= 4 and sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3])
    elif len(sys.argv) > 1:
        print("Usage:")
        print(f"  {SCRIPT.name} --debug-dump <transcript_path>        # show cleaned transcript")
        print(f"  {SCRIPT.name} --debug-with-prompt <transcript_path>  # show full file sent to Gemini")
        print(f"  {SCRIPT.name} --debug-run <transcript_path>          # run Gemini and print output")
        print(f"  {SCRIPT.name} --worker <transcript_path> <session_id>")
    else:
        hook_entry()
