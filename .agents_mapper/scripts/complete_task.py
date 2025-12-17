#!/usr/bin/env python3
"""
Script to complete a task that is in REVIEW status.
Moves the task from active to completed and relocates the folder.
"""

import json
import shutil
import sys
from pathlib import Path


def get_folder_name(task_id: str, title: str) -> str:
    """Convert task id and title to folder name format: id_title_in_snake_case"""
    snake_title = title.lower().replace(" ", "_")
    return f"{task_id}_{snake_title}"


def complete_task(task_id: str) -> None:
    """Complete a task that is in REVIEW status."""
    base_path = Path(__file__).parent.parent / "tasks"
    index_path = base_path / "_index.json"
    active_tasks_path = base_path / "active_tasks"
    completed_tasks_path = base_path / "completed_tasks"

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
    if "active" not in index or "completed" not in index:
        print("Error: Index file is missing 'active' or 'completed' keys.")
        sys.exit(1)

    # Find the task in active section
    task_to_complete = None
    task_index = None
    for i, task in enumerate(index["active"]):
        if task.get("id") == task_id:
            task_to_complete = task
            task_index = i
            break

    if task_to_complete is None:
        print(f"Error: Task with ID '{task_id}' not found in active tasks.")
        sys.exit(1)

    if task_to_complete.get("status") != "REVIEW":
        print(f"Error: Task '{task_id}' is not in REVIEW status. Current status: {task_to_complete.get('status')}")
        print("Only tasks with REVIEW status can be completed.")
        sys.exit(1)

    # Get folder name
    folder_name = get_folder_name(task_id, task_to_complete["title"])
    source_folder = active_tasks_path / folder_name
    dest_folder = completed_tasks_path / folder_name

    # Track if folder was moved (for rollback if needed)
    folder_moved = False

    # Check if source folder exists
    if not source_folder.exists():
        print(f"Warning: Source folder '{source_folder}' does not exist.")
        print("Proceeding with index update only...")
    else:
        # Move the folder
        if dest_folder.exists():
            print(f"Error: Destination folder '{dest_folder}' already exists.")
            sys.exit(1)
        try:
            shutil.move(str(source_folder), str(dest_folder))
            folder_moved = True
            print(f"Moved folder: {source_folder} -> {dest_folder}")
        except (shutil.Error, OSError) as e:
            print(f"Error: Failed to move folder: {e}")
            sys.exit(1)

    # Update the index
    # Remove from active
    index["active"].pop(task_index)

    # Add to completed with COMPLETED status
    task_to_complete["status"] = "COMPLETED"
    index["completed"].insert(0, task_to_complete)

    # Save the index
    try:
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
            f.write("\n")
    except OSError as e:
        print(f"Error: Failed to write index file: {e}")
        # Rollback folder move if it happened
        if folder_moved:
            try:
                shutil.move(str(dest_folder), str(source_folder))
                print("Rolled back folder move due to index write failure.")
            except (shutil.Error, OSError):
                print(f"Warning: Could not rollback folder move. Folder is at: {dest_folder}")
        sys.exit(1)

    print(f"Task '{task_id} - {task_to_complete['title']}' has been completed successfully!")
    print(f"Updated index at: {index_path}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python complete_task.py <task_id>")
        print("Example: python complete_task.py 11")
        sys.exit(1)

    task_id = sys.argv[1]
    complete_task(task_id)


if __name__ == "__main__":
    main()
