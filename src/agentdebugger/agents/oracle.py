"""The oracle agent: submits the known-good fix, first try.

It is not a model and it is not a baseline — it cannot generalise to anything,
because it simply reads the reference fix out of the task definition. It exists
for two honest reasons:

* **Solvability.** If the oracle cannot reach a green suite and a top score, the
  task is broken or the sandbox is rejecting a legitimate fix. The tests assert
  exactly that, and it is how the environment's own correctness is checked.
* **A score ceiling.** Every grader's maximum is whatever the oracle scores, so
  model results have something real to be compared against.

It is also what ``agentdebugger episode`` runs by default, which lets someone
with no GPU and no API key watch a full episode — sandbox, grader and all — in a
few seconds.
"""

from __future__ import annotations

from typing import Any

from agentdebugger.protocol import Action, Observation
from agentdebugger.tasks import get_task


class OracleAgent:
    """Submits the task's reference fix, with the reference diagnosis."""

    name = "oracle"

    def act(self, observation: Observation, info: dict[str, Any]) -> Action:
        ground_truth = get_task(observation.task_id).ground_truth

        # The oracle knows the answer, so a second attempt would mean the
        # environment rejected a correct fix. Give up rather than resubmit: that
        # turns a broken environment into a visibly bad score instead of a loop.
        if observation.previous_attempts:
            return Action(
                action_type="give_up",
                final_diagnosis=ground_truth.reference_hypothesis,
            )

        return Action(
            action_type="submit_fix",
            fixed_code=ground_truth.fixed_code,
            hypothesis=ground_truth.reference_hypothesis,
        )
