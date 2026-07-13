"""The multi-step debugging environment.

An episode is a conversation with a broken program. The agent submits a fix and
sees what the tests say, asks for context, or gives up. Every submission runs in
the sandbox; the agent's only channel to ground truth is execution output.

Step rewards are shaped so the *direction* of travel is rewarded immediately
(more tests passing than last step) rather than only at the end, and so the
behaviours the project exists to discourage — submitting a fix with no
hypothesis, thrashing without progress, breaking tests that used to pass — cost
something at the moment they happen.

The episode ends when the grader says the task is solved, when the attempt
budget runs out, when the agent gives up, or when the step budget is exhausted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any

from agentdebugger.protocol import (
    ACTION_TYPES,
    QUERY_TYPES,
    Action,
    FixAttempt,
    Observation,
    Reward,
    StepResult,
)
from agentdebugger.rewards.graders import Episode, get_grader
from agentdebugger.sandbox import execute
from agentdebugger.tasks import Task, get_task

#: Step-reward constants. Positive terms pay for progress; negative ones price
#: the failure modes the environment is designed to discourage.
SOLVE_BONUS = 0.50
PROGRESS_WEIGHT = 0.15  # scaled by the fraction of the suite newly passing
REGRESSION_WEIGHT = 0.10  # scaled by the fraction newly broken
STAGNATION_PENALTY = 0.05
TIMEOUT_PENALTY = 0.10
MISSING_HYPOTHESIS_PENALTY = 0.10
NO_ATTEMPTS_PENALTY = 0.15
INVALID_ACTION_PENALTY = 0.05
EXTRA_QUERY_PENALTY = 0.05
TRUNCATION_PENALTY = 0.20

_TESTS_PASSED = re.compile(r"(\d+) passed")


class EpisodeFinished(Exception):
    """Raised when an action arrives after the episode has already ended."""


@dataclass
class _EpisodeState:
    """Mutable state for one episode. Rebuilt from scratch by every ``reset``."""

    task: Task
    observation: Observation
    attempts: list[FixAttempt] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    cumulative_reward: float = 0.0
    step_number: int = 0
    queries_used: int = 0
    previous_tests_passed: int = 0
    done: bool = False


class TaskEnvironment:
    """A stateful, single-episode debugging environment over the hand-written tasks."""

    def __init__(self) -> None:
        self._state: _EpisodeState | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def reset(self, task_id: str = "easy") -> Observation:
        """Start a fresh episode and return the opening observation.

        The buggy code is run once here, so the agent's first observation
        includes the real failure output rather than a description of it.
        """
        task = get_task(task_id)
        result = execute(task.buggy_code, task.executable_tests, policy=task.policy)
        tests_passed = _parse_tests_passed(result.output, task.tests_total)

        observation = Observation(
            task_id=task.task_id,
            task_description=task.description,
            buggy_code=task.buggy_code,
            test_suite=task.test_suite,
            initial_error_output=result.output,
            current_code=task.buggy_code,
            current_error_output=result.output,
            tests_passed=tests_passed,
            tests_total=task.tests_total,
            previous_attempts=(),
            attempts_remaining=task.max_attempts,
            max_attempts=task.max_attempts,
            step_number=0,
            max_steps=task.max_steps,
            done=False,
            hint_used=False,
        )
        self._state = _EpisodeState(
            task=task,
            observation=observation,
            previous_tests_passed=tests_passed,
        )
        return observation

    def step(self, action: Action) -> StepResult:
        """Apply one action and return the resulting observation, reward and info."""
        state = self._require_episode()
        if state.done:
            raise EpisodeFinished("Episode is over. Call reset() to start a new one.")

        state.step_number += 1
        if state.step_number > state.task.max_steps:
            return self._truncate(state)

        if action.action_type == "submit_fix":
            return self._submit_fix(state, action)
        if action.action_type == "query_context":
            return self._query_context(state, action)
        if action.action_type == "give_up":
            return self._give_up(state, action)

        return self._respond(
            state,
            step_reward=-INVALID_ACTION_PENALTY,
            info={
                "error": (
                    f"Unknown action_type {action.action_type!r}. "
                    f"Valid: {', '.join(sorted(ACTION_TYPES))}."
                )
            },
        )

    @property
    def observation(self) -> Observation:
        """The current observation."""
        return self._require_episode().observation

    def state(self) -> dict[str, Any]:
        """A snapshot of internal episode state, for debugging and the ``/state`` endpoint."""
        if self._state is None:
            return {"active": False}

        state = self._state
        return {
            "active": True,
            "task_id": state.task.task_id,
            "step_number": state.step_number,
            "attempts_used": len(state.attempts),
            "tests_passed": state.observation.tests_passed,
            "tests_total": state.task.tests_total,
            "best_tests_passed": max((a.tests_passed for a in state.attempts), default=0),
            "hypotheses": list(state.hypotheses),
            "cumulative_reward": round(state.cumulative_reward, 4),
            "queries_used": state.queries_used,
            "done": state.done,
        }

    # ── actions ───────────────────────────────────────────────────────────────

    def _submit_fix(self, state: _EpisodeState, action: Action) -> StepResult:
        hypothesis = (action.hypothesis or "").strip()
        if not hypothesis:
            # The central constraint of the environment: no hypothesis, no run.
            # The fix is not even executed, so guessing costs a step and pays nothing.
            return self._respond(
                state,
                step_reward=-MISSING_HYPOTHESIS_PENALTY,
                info={"error": "submit_fix requires a hypothesis. The fix was not executed."},
            )

        if state.observation.attempts_remaining <= 0:
            return self._respond(
                state,
                step_reward=-NO_ATTEMPTS_PENALTY,
                info={"error": "No attempts remaining. Use query_context or give_up."},
            )

        task = state.task
        code = action.fixed_code or ""
        result = execute(code, task.executable_tests, policy=task.policy)
        tests_passed = _parse_tests_passed(result.output, task.tests_total)

        attempt = FixAttempt(
            attempt_number=len(state.attempts) + 1,
            hypothesis=hypothesis,
            code_submitted=code,
            execution_output=result.output,
            tests_passed=tests_passed,
            tests_total=task.tests_total,
            execution_time_ms=result.duration_ms,
            timed_out=result.timed_out,
        )
        state.attempts.append(attempt)
        state.hypotheses.append(hypothesis)

        solved = get_grader(task.task_id).is_solved(task, attempt)
        step_reward = self._progress_reward(state, tests_passed, result.timed_out, solved)
        if solved:
            step_reward += SOLVE_BONUS

        attempts_remaining = task.max_attempts - len(state.attempts)
        state.observation = replace(
            state.observation,
            current_code=code,
            current_error_output=result.output,
            tests_passed=tests_passed,
            previous_attempts=tuple(state.attempts),
            attempts_remaining=attempts_remaining,
        )
        state.previous_tests_passed = tests_passed

        info = {
            "tests_passed": tests_passed,
            "tests_total": task.tests_total,
            "execution_time_ms": result.duration_ms,
            "timed_out": result.timed_out,
            "blocked": result.blocked,
            "solved": solved,
        }
        if solved or attempts_remaining <= 0:
            return self._finish(state, step_reward, info)
        return self._respond(state, step_reward, info)

    def _query_context(self, state: _EpisodeState, action: Action) -> StepResult:
        if action.query_type not in QUERY_TYPES:
            return self._respond(
                state,
                step_reward=-INVALID_ACTION_PENALTY,
                info={
                    "error": (
                        f"Invalid query_type {action.query_type!r}. "
                        f"Valid: {', '.join(sorted(QUERY_TYPES))}."
                    )
                },
            )

        answer = self._answer_query(state, action.query_type, action.query_target)

        # The first question is free; asking instead of thinking is not.
        first_query = state.queries_used == 0
        state.queries_used += 1
        if first_query:
            state.observation = replace(state.observation, hint_used=True)

        return self._respond(
            state,
            step_reward=0.0 if first_query else -EXTRA_QUERY_PENALTY,
            info={"query_result": answer, "free_query": first_query},
        )

    def _give_up(self, state: _EpisodeState, action: Action) -> StepResult:
        if action.final_diagnosis:
            state.hypotheses.append(action.final_diagnosis)
        return self._finish(state, step_reward=0.0, info={"gave_up": True})

    # ── reward and termination ────────────────────────────────────────────────

    def _progress_reward(
        self, state: _EpisodeState, tests_passed: int, timed_out: bool, solved: bool
    ) -> float:
        """Pay for movement towards a green suite, charge for movement away from it."""
        total = state.task.tests_total
        previous = state.previous_tests_passed
        reward = -TIMEOUT_PENALTY if timed_out else 0.0

        if tests_passed > previous:
            reward += PROGRESS_WEIGHT * (tests_passed - previous) / total
        elif tests_passed < previous:
            reward -= REGRESSION_WEIGHT * (previous - tests_passed) / total
        elif not solved:
            # Stagnation costs, but only when the agent is actually stuck. On the
            # hard task the test count cannot move — the suite is green before and
            # after — so a winning submission must not be charged for standing still.
            reward -= STAGNATION_PENALTY

        return reward

    def _truncate(self, state: _EpisodeState) -> StepResult:
        return self._finish(
            state,
            step_reward=-TRUNCATION_PENALTY,
            info={"error": "Step budget exhausted. Episode truncated."},
        )

    def _finish(self, state: _EpisodeState, step_reward: float, info: dict[str, Any]) -> StepResult:
        """End the episode and run the grader."""
        state.done = True
        grader = get_grader(state.task.task_id)
        grader_score = grader.score(
            state.task,
            Episode(attempts=tuple(state.attempts), hypotheses=tuple(state.hypotheses)),
        )

        keywords = state.task.ground_truth.hypothesis_keywords
        info = dict(info)
        info["hypothesis_matched_bug"] = any(
            keyword.lower() in hypothesis.lower()
            for hypothesis in state.hypotheses
            for keyword in keywords
        )
        return self._respond(state, step_reward, info, grader_score=grader_score)

    def _respond(
        self,
        state: _EpisodeState,
        step_reward: float,
        info: dict[str, Any],
        grader_score: float = 0.0,
    ) -> StepResult:
        state.cumulative_reward += step_reward
        state.observation = replace(
            state.observation,
            step_number=state.step_number,
            done=state.done,
        )
        reward = Reward(
            step_reward=round(step_reward, 4),
            cumulative_reward=round(state.cumulative_reward, 4),
            grader_score=round(grader_score, 4),
        )
        return StepResult(
            observation=state.observation,
            reward=reward,
            done=state.done,
            info={"step_number": state.step_number, **info},
        )

    # ── context queries ───────────────────────────────────────────────────────

    def _answer_query(self, state: _EpisodeState, query_type: str, target: str | None) -> str:
        task = state.task

        if query_type == "function_signature":
            signatures = [
                line.strip()
                for line in task.buggy_code.splitlines()
                if line.strip().startswith("def ")
            ]
            if target:
                signatures = [s for s in signatures if target in s] or signatures
            return "Function signatures:\n" + "\n".join(f"  {s}" for s in signatures)

        if query_type == "related_code":
            return f"Full source:\n{task.buggy_code}"

        if query_type == "error_explanation":
            return (
                f"Output of the test suite against the current code:\n"
                f"{state.observation.current_error_output}\n\n"
                f"Each FAILED line is an assertion that did not hold; each ERROR line is an "
                f"exception raised before the assertion ran."
            )

        if query_type == "test_details":
            return f"Full test suite:\n{task.test_suite}"

        # test_suggestion: the one hint that points at the *shape* of the bug
        # without naming it. This is what makes the hard task solvable at all.
        return _HINTS[task.task_id]

    def _require_episode(self) -> _EpisodeState:
        if self._state is None:
            raise EpisodeFinished("No episode in progress. Call reset() first.")
        return self._state


_HINTS = {
    "easy": "Look closely at the loop's termination condition and its boundaries.",
    "medium": (
        "Do not trust the function named in the error message. Trace the data backwards "
        "and check what each function it calls actually returns."
    ),
    "hard": (
        "Every test here passes, and the bug is real. Ask what the suite does not "
        "exercise: all of these tests call the counter from a single thread. "
        "Consider what happens when many threads call it at once."
    ),
}


def _parse_tests_passed(output: str, tests_total: int) -> int:
    """Read the test count out of sandbox output.

    The last match wins: the code under test is free to print, and a fix that
    happens to print "3 passed" should not be able to fake its own score.
    """
    matches = _TESTS_PASSED.findall(output)
    if not matches:
        return 0
    return min(int(matches[-1]), tests_total)
