#!/usr/bin/env python3
# worktree_create.py — WorktreeCreate hook.
# Creates the git worktree and symlinks shared gitignored dirs (data/, thirdParty/).

import sys
import json
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

SCRIPT = Path(__file__).resolve()
LOG_FILE = SCRIPT.parent.parent / "logs" / "hooks_debug.log"
LOG_SOURCE = "WORKTREE_CREATE"

SHARED_DIRS = ["data", "thirdParty"]


def log(msg: str):
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{ts}] [{LOG_SOURCE}] {msg}\n")
    except Exception:
        pass


def main():
    raw = sys.stdin.read().strip()
    payload = json.loads(raw) if raw else {}
    name = payload.get("name", "").strip()

    if not name:
        log("ERROR: 'name' not found in hook input")
        sys.exit(1)

    repo_root = Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    )
    worktree_path = Path.home() / ".claude" / "worktrees" / name

    # Create the worktree
    log(f"Creating worktree '{name}' at {worktree_path}")
    subprocess.run(["git", "worktree", "add", str(worktree_path)], check=True, cwd=repo_root)

    # Symlink shared gitignored dirs
    for dir_name in SHARED_DIRS:
        target = repo_root / dir_name
        link = worktree_path / dir_name

        if not target.is_dir():
            log(f"  {dir_name}/ not found in repo root, skipping")
            continue

        if link.is_symlink():
            log(f"  {dir_name}/ already a symlink, skipping")
        else:
            if link.exists():
                shutil.rmtree(link)
            link.symlink_to(target)
            log(f"  Linked {dir_name}/ -> {target}")

    print(str(worktree_path))
    log(f"Done. Worktree ready at {worktree_path}")


if __name__ == "__main__":
    main()