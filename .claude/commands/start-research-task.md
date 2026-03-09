Start the research phase for a task. $ARGUMENTS must be a task ID (e.g., "03" or "3").

Follow these steps:

## 1. Validate the Task

Execute:
```
python .agents_mapper/scripts/get_task_info.py $ARGUMENTS
```

This returns JSON with task info or an error.

- **If `success: false`**: Show the error message and STOP.
- **If `success: true` but `status` is NOT "PENDING PLANNING"**: Report "Task '{title}' has status '{status}'. Research phase requires status 'PENDING PLANNING'." and STOP.
- **If valid**: Continue to step 2.

## 2. Load Task Context

- Read `description.md` from the `folder_path` returned by the script
- Understand the task objective

## 3. Confirm Task

Present to the user:
```
Task Found:
  ID: {id}
  Title: {title}
  Status: {status}
  Description: {first 2-3 sentences from description.md}

Do you want to start the research phase for this task?
```

**Wait for user confirmation before proceeding.**

## 4. Plan and Launch Research Agents

Based on:
- The task description
- Your knowledge of the project structure from CLAUDE.md
- The codebase architecture (ovo/entities/, ovo/slam/, ovo/utils/, scripts/)

Identify which source files need to be analyzed. Group related files logically.

**Guidelines:**
- Maximum 2-3 file-context-extractor agents
- Each agent analyzes 1-3 related files
- Focus on files directly relevant to the task

Launch file-context-extractor agents **in parallel** using the Task tool.

For each agent, use this prompt format:
```
Task: {task_title} - {specific_focus_for_this_agent}
Files to analyze: {comma_separated_absolute_paths}
Findings directory: {findings_path from script}
Output filename: {name}.md
```

**IMPORTANT**: Launch all agents in a single message with multiple Task tool calls.

## 5. Monitor Progress

After launching:
- Confirm agents are running
- List what each agent is investigating
- Show the findings directory path where results will appear

**Wait for ALL file-context-extractor agents to complete before proceeding.**

## 6. Consolidate Research

Once all agents have finished, launch the **research-consolidator** agent to unify all findings:

```
Task: {task_title}
Task Folder Path: {folder_path from script}
Output Filename: research.md
```

This agent will:
- Read all `.md` files from the findings directory
- Synthesize them into a single `research.md` in the task folder root
- Preserve all file:line references for implementation

## 7. Final Report

After consolidation completes:
- Confirm research.md has been created
- Show the path to the consolidated research file
- Summarize what was discovered (2-3 sentences)
- Indicate the task is ready for planning phase
