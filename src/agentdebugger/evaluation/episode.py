"""Drive an agent through the multi-step task environment and record what happened."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from agentdebugger.agents.base import Agent
from agentdebugger.envs.task_env import TaskEnvironment
from agentdebugger.protocol import Action
from agentdebugger.tasks import list_tasks

#: Called after every step with (step_number, action, step_result). The CLI uses
#: it to render an episode live; evaluation ignores it.
StepHook = Callable[[int, Action, Any], None]


@dataclass(frozen=True)
class TurnRecord:
    """One step of an episode, kept for the transcript."""

    step: int
    action_type: str
    hypothesis: str | None
    step_reward: float
    tests_passed: int
    tests_total: int


@dataclass(frozen=True)
class EpisodeResult:
    """What an agent achieved on one task."""

    task_id: str
    agent: str
    grader_score: float
    cumulative_reward: float
    steps_taken: int
    attempts_used: int
    tests_passed: int
    tests_total: int
    solved: bool
    turns: tuple[TurnRecord, ...] = field(default=())

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationReport:
    """An agent's results across several tasks."""

    agent: str
    episodes: tuple[EpisodeResult, ...]

    @property
    def mean_score(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.grader_score for e in self.episodes) / len(self.episodes)

    @property
    def solve_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.solved for e in self.episodes) / len(self.episodes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "mean_score": round(self.mean_score, 4),
            "solve_rate": round(self.solve_rate, 4),
            "episodes": [episode.as_dict() for episode in self.episodes],
        }


def run_episode(
    agent: Agent,
    task_id: str = "easy",
    on_step: StepHook | None = None,
) -> EpisodeResult:
    """Run one full episode of ``task_id`` with ``agent``.

    The environment's own step budget terminates the episode, so a misbehaving
    agent cannot loop forever.
    """
    if hasattr(agent, "reset"):
        agent.reset()

    env = TaskEnvironment()
    observation = env.reset(task_id)
    info: dict[str, Any] = {}
    turns: list[TurnRecord] = []
    result = None

    while True:
        action = agent.act(observation, info)
        result = env.step(action)

        turns.append(
            TurnRecord(
                step=result.observation.step_number,
                action_type=action.action_type,
                hypothesis=action.hypothesis or action.final_diagnosis,
                step_reward=result.reward.step_reward,
                tests_passed=result.observation.tests_passed,
                tests_total=result.observation.tests_total,
            )
        )
        if on_step is not None:
            on_step(result.observation.step_number, action, result)

        observation, info = result.observation, result.info
        if result.done:
            break

    return EpisodeResult(
        task_id=task_id,
        agent=agent.name,
        grader_score=result.reward.grader_score,
        cumulative_reward=result.reward.cumulative_reward,
        steps_taken=observation.step_number,
        attempts_used=observation.max_attempts - observation.attempts_remaining,
        tests_passed=observation.tests_passed,
        tests_total=observation.tests_total,
        solved=bool(info.get("solved", False)),
        turns=tuple(turns),
    )


def evaluate_agent(
    agent: Agent,
    task_ids: Sequence[str] | None = None,
    on_step: StepHook | None = None,
) -> EvaluationReport:
    """Run ``agent`` on every task and collect the results."""
    task_ids = list(task_ids) if task_ids else list_tasks()
    episodes = tuple(run_episode(agent, task_id, on_step=on_step) for task_id in task_ids)
    return EvaluationReport(agent=agent.name, episodes=episodes)
