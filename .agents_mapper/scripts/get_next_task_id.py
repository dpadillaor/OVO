#!/usr/bin/env python3
"""
Script to get the next available task ID.
"""

import json
import sys
from pathlib import Path


def get_next_id() -> str:
    """Find the maximum ID in active and completed tasks and return next ID."""
    base_path = Path(__file__).parent.parent / "tasks"
    index_path = base_path / "_index.json"

    if not index_path.exists():
        return "01"

    try:
        with open(index_path, "r") as f:
            index_data = json.load(f)
    except json.JSONDecodeError:
        return "01"

    max_id = 0
    
    # Check active tasks
    for task in index_data.get("active", []):
        try:
            tid = int(task.get("id", "0"))
            if tid > max_id:
                max_id = tid
        except ValueError:
            pass
            
    # Check completed tasks
    for task in index_data.get("completed", []):
        try:
            tid = int(task.get("id", "0"))
            if tid > max_id:
                max_id = tid
        except ValueError:
            pass
            
    next_id = max_id + 1
    return f"{next_id:02d}"


def main():
    print(get_next_id())


if __name__ == "__main__":
    main()
