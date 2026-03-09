#!/usr/bin/env python3
"""
Script to get task information by ID and validate its status.
Usage: python get_task_info.py <task_id>

Returns task info as JSON if valid, or error message.
"""

import json
import sys
from pathlib import Path


def get_task_info(task_id: str) -> dict:
    """
    Find a task by ID and return its information.

    Returns dict with:
      - success: bool
      - If success: id, title, status, folder_path
      - If error: error message
    """
    base_path = Path(__file__).parent.parent / "tasks"
    index_path = base_path / "_index.json"

    if not index_path.exists():
        return {"success": False, "error": "Index file not found"}

    try:
        with open(index_path, "r") as f:
            index_data = json.load(f)
    except json.JSONDecodeError:
        return {"success": False, "error": "Invalid index file"}

    # Normalize task_id (remove leading zeros for comparison, but keep original)
    search_id = task_id.lstrip("0") or "0"

    # Search in active tasks
    for task in index_data.get("active", []):
        tid = task.get("id", "").lstrip("0") or "0"
        if tid == search_id:
            title = task.get("title", "")
            status = task.get("status", "")
            # Build folder name: {id}_{title_snake_case}
            title_snake = title.lower().replace(" ", "_")
            folder_name = f"{task.get('id')}_{title_snake}"
            folder_path = base_path / "active_tasks" / folder_name

            return {
                "success": True,
                "id": task.get("id"),
                "title": title,
                "status": status,
                "folder_path": str(folder_path),
                "findings_path": str(folder_path / "findings")
            }

    # Search in completed tasks
    for task in index_data.get("completed", []):
        tid = task.get("id", "").lstrip("0") or "0"
        if tid == search_id:
            return {
                "success": False,
                "error": f"Task {task_id} ('{task.get('title')}') is in completed tasks with status '{task.get('status')}'"
            }

    return {"success": False, "error": f"Task with ID '{task_id}' not found"}


def main():
    if len(sys.argv) != 2:
        print(json.dumps({"success": False, "error": "Usage: python get_task_info.py <task_id>"}))
        sys.exit(1)

    task_id = sys.argv[1]
    result = get_task_info(task_id)
    print(json.dumps(result, indent=2))

    if not result["success"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
