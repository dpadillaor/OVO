Activate task $ARGUMENTS and launch a subagent to implement it following TDD.

## Steps

1. Run `python .agents_mapper/scripts/get_task_info.py $ARGUMENTS` and parse the JSON.
   - If `success` is false: show the error and STOP.
   - If `status` is not `READY`: tell the user the current status and STOP.
   - Read `strategy.md` and `tests.md` from `folder_path`. If either is empty: warn the user and STOP — run `/plan-task $ARGUMENTS` first.

2. If `worktree_path` is null, run: `python .agents_mapper/scripts/start_task.py $ARGUMENTS`
   - This requires being on `develop`. If the script errors, show the message and STOP.
   - Re-run `get_task_info.py` to get the updated `worktree_path` and `branch`.

3. Read the contents of `strategy.md` and `tests.md`.

4. Launch a subagent (type: general-purpose) with the following prompt, substituting all placeholders:

---
You are implementing task {id}: {title}.

Your working directory is: {worktree_path}
Branch: {branch}
Session log: {folder_path}/session_log.md

## Strategy
{contents of strategy.md}

## Tests
{contents of tests.md}

## Instructions
- Work strictly in TDD cycles: write one failing test → implement the minimum code to pass it → verify it passes → move to the next test.
- Only modify files within the scope defined in Strategy. Do not refactor or change anything outside that scope.
- After each passing test, append a session log entry:
  `python .agents_mapper/scripts/log_session.py {id} "IMPLEMENTATION" "<what was done and in which files>" "<next test or step>"`
- If you encounter something unexpected that changes the approach, log it with the optional unexpected argument.
- Ask the user if you are unsure about scope or approach — do not assume.
- Stop when all tests defined in the Tests section pass. Run the full test suite one final time to confirm.
---

5. When the subagent finishes, report its result to the user and remind them to review the changes and run `/complete-task $ARGUMENTS` when ready.
