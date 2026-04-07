Complete task $ARGUMENTS: push branch, open PR to develop, remove worktree, and archive.

## Steps

1. Run `python .agents_mapper/scripts/get_task_info.py $ARGUMENTS` and parse the JSON.
   - If `success` is false: show the error and STOP.
   - If `status` is not `IN_REVIEW`: tell the user and STOP.

2. Show a summary of what is about to happen:
   - Task title and ID
   - Branch to push
   - Worktree to remove (if any)

   Ask the user to confirm before proceeding.

3. Run `python .agents_mapper/scripts/complete_task.py $ARGUMENTS`

4. Report the result and show the PR URL.
