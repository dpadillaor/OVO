Initiate a new task based on the description provided in $ARGUMENTS.

Follow these steps interactively:

1.  **Analyze Context**: Execute `python .agents_mapper/scripts/get_next_task_id.py` to get the next available Task ID.
2.  **Propose Title**: Based on the user's prompt ($ARGUMENTS), suggest a concise, descriptive title for the task (e.g., "Feature Name" or "Refactor Logic").
3.  **Confirm**: Present the proposed ID and Title to the user. Ask for confirmation or a revised title. Repeat until the user agrees.
4.  **Execute**: Once confirmed, run the creation script with the agreed title:
    `python .agents_mapper/scripts/new_task.py "{Agreed Title}"`

    This script will automatically:
    - Assign the new ID.
    - Create the folder in `active_tasks`.
    - Create empty `strategy.md` and `tests.md` files.
    - Create `description.md` with a template.
    - Create a `/findings/` directory.
    - Add the task to `_index.json` with status `PENDING PLANNING`.

5.  **Initialize Description**:
    - Identify the newly created task folder path (from the script output).
    - Read the `description.md` file in that folder.
    - Replace the `[TODO: Add task description here]` placeholder with a clear, specific paragraph describing the purpose of the task. Use the information from the user's original prompt ($ARGUMENTS) and your understanding of the project.
    - **Do NOT** ask for permission for this specific edit; consider it part of the task initialization process you are authorized to complete.

6.  **Report**: Confirm the completion to the user and show the location of the new task files.
