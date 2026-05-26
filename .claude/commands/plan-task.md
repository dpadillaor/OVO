Start the planning phase for task $ARGUMENTS. This is a HitL session: propose strategy and tests, discuss with the user, and write the approved plan.

## Steps

1. Run `python .agents_mapper/scripts/get_task_info.py $ARGUMENTS` and parse the JSON.
   - If `success` is false: show the error and STOP.
   - If `status` is not `PENDING PLANNING`: tell the user the current status and STOP.

2. Load context:
   - Read `description.md` from `folder_path`
   - If `research.md` exists in `folder_path`, read it too

3. Based on the description and research (if any), propose:
   - **Strategy**: a concrete implementation plan. Which files to touch, in what order,
     what each change does. Be specific — this is what the implementation agent will follow.
   - **Tests**: the verification criteria. Each test should be independently runnable and
     have a clear pass/fail condition. These are the agent's definition of done.

   Present both to the user and discuss until approved. Iterate as needed.

4. Once the user approves:
   - Write the agreed strategy to `strategy.md`
   - Write the agreed tests to `tests.md`
   - Run `python .agents_mapper/scripts/update_task_status.py $ARGUMENTS READY`

5. Confirm to the user that the task is ready to activate with `/start-task $ARGUMENTS`.
