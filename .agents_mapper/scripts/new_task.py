#!/usr/bin/env python3
"""
Script to create a new task.
Calculates the next ID, adds it to the index with PENDING PLANNING status,
creates the directory, and initializes standard markdown files.
"""

import json
import sys
from pathlib import Path


def get_next_id(index_data: dict) -> str:
    """Find the maximum ID in active and completed tasks and return next ID."""
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


def get_folder_name(task_id: str, title: str) -> str:
    """Convert task id and title to folder name format: id_title_in_snake_case"""
    snake_title = title.lower().replace(" ", "_")
    # Keep only alphanumeric and underscores
    snake_title = "".join(c for c in snake_title if c.isalnum() or c == "_")
    return f"{task_id}_{snake_title}"


def create_task_files(folder_path: Path) -> None:
    """Creates the standard markdown files for the task."""
    
    # 1. strategy.md (Empty)
    (folder_path / "strategy.md").touch()
    
    # 2. tests.md (Empty)
    (folder_path / "tests.md").touch()
    
    # 3. description.md (Template)
    description_content = """# Purpose

[TODO: Add task description here]

## Task files
- `strategy.md`: Describes the architecture and the strategy to be implemented.
- `tests.md`: Summarizes the proposed TDD tests to validate the implementation.
"""
    with open(folder_path / "description.md", "w") as f:
        f.write(description_content)


def create_task(title: str) -> None:
    base_path = Path(__file__).parent.parent / "tasks"
    index_path = base_path / "_index.json"
    active_tasks_path = base_path / "active_tasks"

    # Read the index
    if not index_path.exists():
        print(f"Error: Index file not found at '{index_path}'")
        sys.exit(1)

    try:
        with open(index_path, "r") as f:
            index = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in index file: {e}")
        sys.exit(1)

    # Validate index structure
    if "active" not in index:
        index["active"] = []

    next_id = get_next_id(index)
    folder_name = get_folder_name(next_id, title)
    new_folder_path = active_tasks_path / folder_name

    # Create folder
    if new_folder_path.exists():
        print(f"Error: Target folder '{new_folder_path}' already exists.")
        sys.exit(1)
    
    try:
        new_folder_path.mkdir(parents=True, exist_ok=False)
        create_task_files(new_folder_path)
    except OSError as e:
        print(f"Error: Failed to create directory or files: {e}")
        sys.exit(1)

    # Add to index
    new_task = {
        "id": next_id,
        "title": title,
        "status": "PENDING PLANNING"
    }
    
    index["active"].append(new_task)

    # Save index
    try:
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
            f.write("\n")
    except OSError as e:
        print(f"Error: Failed to write index file: {e}")
        # Rollback directory creation
        try:
            import shutil
            shutil.rmtree(new_folder_path)
            print("Rolled back directory creation.")
        except OSError:
            print(f"Warning: Could not rollback directory creation at {new_folder_path}")
        sys.exit(1)

    print(f"Task created successfully!")
    print(f"ID: {next_id}")
    print(f"Title: {title}")
    print(f"Location: {new_folder_path}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python new_task.py <task_title>")
        print('Example: python new_task.py "My New Feature"')
        sys.exit(1)

    title = sys.argv[1]
    create_task(title)


if __name__ == "__main__":
    main()