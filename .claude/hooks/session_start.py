import sys
import json
from pathlib import Path
from datetime import datetime, date

VAULT = Path.home() / "Documentos" / "Obsidian Vault$"
SCRIPT = Path(__file__).resolve()
LOG_FILE = SCRIPT.parent.parent / "logs" / "hooks_debug.log"
LOG_SOURCE = "SESSION_START"


def log(msg: str):
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{ts}] [{LOG_SOURCE}] {msg}\n")
    except Exception:
        pass


session_id = "unknown"
try:
    raw_stdin = sys.stdin.read().strip()
    if raw_stdin:
        payload = json.loads(raw_stdin)
        session_id = payload.get("session_id", "unknown")
except Exception:
    # Keep SessionStart resilient even if stdin is malformed.
    session_id = "unknown"

files = ["SOUL.md", "USER.md", "MEMORY.md"]
context_parts = []

for f in files:
    p = VAULT / f
    if p.exists():
        context_parts.append(f"## {f}\n{p.read_text(encoding='utf-8')}")

today_log = VAULT / "daily" / f"{date.today()}.md"
if today_log.exists():
    context_parts.append(f"## Today's Log\n{today_log.read_text(encoding='utf-8')}")

additional_context = "\n\n".join(context_parts)
log(
    f"SessionStart fired. session={session_id}, "
    f"sections={len(context_parts)}, context_chars={len(additional_context)}"
)

output = {
    "hookSpecificOutput": {
        "hookEventName": "SessionStart",
        "additionalContext": additional_context
    }
}
print(json.dumps(output))
sys.exit(0)
