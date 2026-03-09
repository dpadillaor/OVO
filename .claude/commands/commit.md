Analyze the current git state and create one or more commits to leave the history clean and well-organized.

Follow these steps:

1. **Inspect the working tree**: Run `git status` and `git diff` (including untracked files) to see everything that has changed.

2. **Decide how many commits are needed**: Group changes by logical concern. A single commit is fine when all changes belong to the same feature or fix. Split into multiple commits when changes are clearly independent (e.g. a bug fix + a new feature, or code changes + documentation). Err on the side of fewer commits unless separation genuinely improves clarity.

3. **Write the commit message(s)**:
   - Be descriptive but concise: two or three sentences at most.
   - The first line is a short summary (imperative mood, no period, ~60 chars max).
   - If needed, add a blank line followed by one or two sentences of extra context.
   - Never mention Claude or any AI tool in the message.
   - Do not use bullet points or lists inside the message.

4. **Stage and commit**: For each commit, stage only the relevant files explicitly (never `git add -A` blindly) and create the commit.

5. **Confirm**: After all commits are done, run `git log --oneline -5` and show the result to the user.
