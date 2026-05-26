Resume work on task $ARGUMENTS. Restores context from the session log and strategy so the session can continue where it left off.

## Steps

1. Run `python .agents_mapper/scripts/get_task_info.py $ARGUMENTS` and parse the JSON.
   - If `success` is false: show the error and STOP.
   - If `status` is not `IN_PROGRESS` or `IN_REVIEW`: tell the user and STOP.

2. Load:
   - `strategy.md` from `folder_path`
   - `tests.md` from `folder_path`
   - `session_log.md` from `folder_path` — read the last 3 entries only

3. Present a concise summary:
   - Task title, status, branch, worktree path
   - What was last done (from session log)
   - What comes next (from session log)
   - Which tests are still pending (infer from session log and tests.md)

4. Ask the user: continue in this session, or generate the agent command for a new terminal?
   - If this session: proceed directly with the implementation following TDD, updating the session log after each cycle.
   - If new terminal: output the `claude` invocation command as defined in `/start-task`, using the existing worktree.
