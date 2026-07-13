"""The hand-written debugging tasks and their registry.

Three tasks, each probing a different failure mode of LLM debugging:

``easy``
    Can the agent read a stack trace? (off-by-one)
``medium``
    Does it trace a symptom to its cause, or blame the frame the error names?
    (red herring)
``hard``
    Does it know that a passing test suite is not proof of correctness?
    (race condition invisible to every sequential test)
"""

from agentdebugger.tasks.easy import TASK as EASY
from agentdebugger.tasks.hard import TASK as HARD
from agentdebugger.tasks.medium import TASK as MEDIUM
from agentdebugger.tasks.models import GroundTruth, Task

TASKS: dict[str, Task] = {task.task_id: task for task in (EASY, MEDIUM, HARD)}


def get_task(task_id: str) -> Task:
    """Look up a task by id, raising ``ValueError`` for unknown ids."""
    try:
        return TASKS[task_id]
    except KeyError:
        raise ValueError(
            f"Unknown task_id {task_id!r}. Available: {', '.join(TASKS)}"
        ) from None


def list_tasks() -> list[str]:
    """Return the available task ids, easiest first."""
    return list(TASKS)


__all__ = ["EASY", "HARD", "MEDIUM", "TASKS", "GroundTruth", "Task", "get_task", "list_tasks"]
