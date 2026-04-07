Create a new task from the description in $ARGUMENTS.

## Steps

1. Run `python .agents_mapper/scripts/get_next_task_id.py` to get the next ID.

2. Based on $ARGUMENTS, propose a concise task title (3-6 words, title case).
   Show the user: proposed ID and title. Ask for confirmation or a revised title.
   Repeat until confirmed.

3. Run `python .agents_mapper/scripts/new_task.py "<confirmed title>"`

4. Read the `description.md` in the newly created folder.
   Replace the `[TODO: Add task description here]` placeholder with a clear paragraph
   describing the task purpose, using $ARGUMENTS as the source. Do not ask permission
   for this edit — it is part of the initialization.

5. Show the user the task folder path and confirm the task is ready for planning.
