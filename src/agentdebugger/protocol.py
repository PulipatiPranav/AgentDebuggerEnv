"""The wire format between an agent and the environments.

Two shapes of interaction live here, because the project has two of them:

* **Tool-call style** (:class:`Action` / :class:`Observation`) — the multi-step
  HTTP environment. The agent submits fixes, asks for context, or gives up, and
  gets execution output back.
* **Structured-text style** (:class:`StructuredAgentOutput`) — the single-turn
  format used for GRPO training, where the whole response is one block of text
  the model must lay out correctly. Format compliance is itself part of the
  reward, so parsing has to be forgiving about whitespace and case but strict
  about which fields exist.

Everything here is a plain dataclass: the training path must not have to import
a web framework.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Confidence = Literal["low", "medium", "high"]

#: Actions the multi-step environment accepts.
ACTION_TYPES: frozenset[str] = frozenset({"submit_fix", "query_context", "give_up"})

#: Context an agent may ask the multi-step environment for.
QUERY_TYPES: frozenset[str] = frozenset(
    {"function_signature", "related_code", "error_explanation", "test_details", "test_suggestion"}
)

#: Actions the structured-text format accepts. ``invalid`` is what an
#: unparseable ACTION field collapses to; it is never a legal choice.
STRUCTURED_ACTIONS: frozenset[str] = frozenset(
    {"inspect_lines", "run_tests", "propose_fix", "request_context", "give_up"}
)

#: Minimum lengths for a structured response to count as well-formed. They exist
#: to reject a model that emits the field names with nothing behind them.
MIN_OBSERVATION_CHARS = 6
MIN_HYPOTHESIS_CHARS = 11


@dataclass(frozen=True)
class Action:
    """One agent move in the multi-step environment."""

    action_type: str
    # submit_fix
    fixed_code: str | None = None
    hypothesis: str | None = None
    # query_context
    query_type: str | None = None
    query_target: str | None = None
    # give_up
    final_diagnosis: str | None = None


@dataclass(frozen=True)
class FixAttempt:
    """A recorded ``submit_fix``, kept in the observation so the agent can see its history."""

    attempt_number: int
    hypothesis: str
    code_submitted: str
    execution_output: str
    tests_passed: int
    tests_total: int
    execution_time_ms: int
    timed_out: bool


@dataclass(frozen=True)
class Observation:
    """Everything the agent can see about the episode right now."""

    task_id: str
    task_description: str
    buggy_code: str
    test_suite: str
    initial_error_output: str

    current_code: str
    current_error_output: str
    tests_passed: int
    tests_total: int
    previous_attempts: tuple[FixAttempt, ...]

    attempts_remaining: int
    max_attempts: int
    step_number: int
    max_steps: int
    done: bool
    hint_used: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Reward:
    """Reward for one step of the multi-step environment.

    ``grader_score`` is 0.0 until the episode ends; on the terminal step it holds
    the task grader's verdict in [0, 1].
    """

    step_reward: float
    cumulative_reward: float
    grader_score: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class StepResult:
    """What :meth:`agentdebugger.envs.task_env.TaskEnvironment.step` returns."""

    observation: Observation
    reward: Reward
    done: bool
    info: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation.as_dict(),
            "reward": self.reward.as_dict(),
            "done": self.done,
            "info": self.info,
        }


@dataclass(frozen=True)
class StructuredAgentOutput:
    """A parsed OBSERVATION/HYPOTHESIS/CONFIDENCE/ACTION/DETAIL response.

    ``valid`` means the response is well-formed, not that it is correct.
    """

    observation: str
    hypothesis: str
    confidence: Confidence
    action: str
    detail: str
    valid: bool
    raw_text: str


_FIELDS = ("OBSERVATION", "HYPOTHESIS", "CONFIDENCE", "ACTION", "DETAIL")
_FIELD_PATTERNS = {
    name: re.compile(
        rf"{name}\s*:\s*(.*?)(?=\n\s*(?:{'|'.join(_FIELDS)})\s*:|$)",
        re.IGNORECASE | re.DOTALL,
    )
    for name in _FIELDS
}


def parse_agent_output(raw_text: str) -> StructuredAgentOutput:
    """Parse a structured response, tolerating whitespace and case.

    A missing or unrecognised field never raises: it degrades. An unknown ACTION
    becomes ``"invalid"`` and an unreadable CONFIDENCE becomes ``"low"``, and the
    response is marked ``valid=False`` so the reward function can price the
    formatting failure.
    """
    values = {name: _extract(raw_text, name) for name in _FIELDS}

    confidence_raw = values["CONFIDENCE"].lower()
    confidence: Confidence = confidence_raw if confidence_raw in {"low", "medium", "high"} else "low"

    action_raw = values["ACTION"].lower()
    action = action_raw if action_raw in STRUCTURED_ACTIONS else "invalid"

    valid = (
        len(values["OBSERVATION"]) >= MIN_OBSERVATION_CHARS
        and len(values["HYPOTHESIS"]) >= MIN_HYPOTHESIS_CHARS
        and confidence_raw in {"low", "medium", "high"}
        and action in STRUCTURED_ACTIONS
        and len(values["DETAIL"]) > 0
    )

    return StructuredAgentOutput(
        observation=values["OBSERVATION"],
        hypothesis=values["HYPOTHESIS"],
        confidence=confidence,
        action=action,
        detail=values["DETAIL"],
        valid=valid,
        raw_text=raw_text,
    )


def _extract(text: str, name: str) -> str:
    match = _FIELD_PATTERNS[name].search(text)
    return match.group(1).strip() if match else ""
