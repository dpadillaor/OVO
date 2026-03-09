Run the task completion script to mark task $ARGUMENTS as completed.

Execute: `python .agents_mapper/scripts/complete_task.py $ARGUMENTS`

This script will:
1. Verify the task with ID $ARGUMENTS exists and is in REVIEW status
2. Move the task folder from `active_tasks` to `completed_tasks`
3. Update the `_index.json` to move the task from active to completed section
4. Change the task status to COMPLETED

Report the result to me.
