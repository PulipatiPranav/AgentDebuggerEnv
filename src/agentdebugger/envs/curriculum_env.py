"""The curriculum environment: one structured response, scored densely.

This is the environment GRPO trains against. The agent sees a buggy function and
must reply in the OBSERVATION/HYPOTHESIS/CONFIDENCE/ACTION/DETAIL format; the
response is parsed, any proposed fix is executed against the bug's test cases,
and the dense reward in :mod:`agentdebugger.rewards.turn` prices the result.

Which bugs the agent sees depends on the training step, via
:class:`~agentdebugger.config.CurriculumSchedule`. Training on the full
distribution from step zero collapses the policy onto the easiest bug type; the
schedule holds tiers 2 and 3 back until the format and localization behaviours
have stabilised.

:func:`score_response` is the whole scoring path with no state attached, so the
GRPO reward function and the evaluation harness can call it directly instead of
driving an environment object.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from agentdebugger.config import DEFAULT_CURRICULUM, CurriculumSchedule, SandboxLimits
from agentdebugger.dataset import Bug, load_bugs
from agentdebugger.protocol import StructuredAgentOutput, parse_agent_output
from agentdebugger.rewards.turn import (
    GroundTruth,
    RewardBreakdown,
    TurnRewardCalculator,
)
from agentdebugger.sandbox import SandboxPolicy, TestResults, run_test_cases

#: Curriculum bugs are single functions with a handful of tiny test cases, so a
#: solution runs in milliseconds. A short deadline keeps a buggy function that
#: loops forever (several of them do) from costing the full default timeout on
#: every rollout — which matters when GRPO scores thousands of them.
CURRICULUM_POLICY = SandboxPolicy(limits=SandboxLimits(wall_clock_seconds=3.0, cpu_seconds=3))

#: Baseline test outcomes per bug id, so `newly_broken` can be computed without
#: re-running the buggy code on every turn. Bug records are immutable, so this
#: is safe to keep for the life of the process.
_BASELINES: dict[str, TestResults] = {}


def baseline_results(bug: Bug) -> TestResults:
    """Run the *buggy* code against its own test cases, and remember the outcome."""
    if bug.id not in _BASELINES:
        _BASELINES[bug.id] = run_test_cases(
            bug.buggy_code,
            bug.function_name,
            [case.as_dict() for case in bug.test_cases],
            policy=CURRICULUM_POLICY,
        )
    return _BASELINES[bug.id]


@dataclass(frozen=True)
class TurnOutcome:
    """Everything one structured response produced."""

    output: StructuredAgentOutput
    tests: TestResults
    reward: RewardBreakdown
    newly_broken: int = 0

    @property
    def solved(self) -> bool:
        """True when the response proposed a fix that passes every test case."""
        return self.reward.fix_quality >= TurnRewardCalculator.SOLVED_THRESHOLD


def score_response(
    bug: Bug,
    raw_text: str,
    turn_number: int = 0,
    calculator: TurnRewardCalculator | None = None,
) -> TurnOutcome:
    """Parse a response, run any fix it proposes, and score it. No environment needed.

    This is the single scoring path: the trainer, the evaluator and the
    environment all go through it, so a reward reported during training means the
    same thing as a reward reported during evaluation.
    """
    calculator = calculator or TurnRewardCalculator()
    output = parse_agent_output(raw_text)

    cases = [case.as_dict() for case in bug.test_cases]
    if output.action == "propose_fix" and cases:
        tests = run_test_cases(output.detail, bug.function_name, cases, policy=CURRICULUM_POLICY)
        broken = tests.newly_broken(baseline_results(bug))
    else:
        tests = TestResults(total=0)
        broken = 0

    reward = calculator.compute_turn_reward(
        agent_output=output,
        ground_truth=GroundTruth.from_bug(bug.as_dict()),
        test_results={**tests.as_dict(), "newly_broken": broken},
        turn_number=turn_number,
    )
    return TurnOutcome(output=output, tests=tests, reward=reward, newly_broken=broken)


@dataclass
class _Turn:
    """One recorded turn of an episode."""

    turn: int
    outcome: TurnOutcome

    @property
    def reward(self) -> RewardBreakdown:
        return self.outcome.reward


class CurriculumEnvironment:
    """Samples bugs according to the curriculum and scores structured responses."""

    def __init__(
        self,
        step: int = 0,
        schedule: CurriculumSchedule = DEFAULT_CURRICULUM,
        calculator: TurnRewardCalculator | None = None,
        seed: int | None = None,
    ) -> None:
        self.schedule = schedule
        self.calculator = calculator or TurnRewardCalculator()
        self._random = random.Random(seed)
        self._step = 0
        self.bugs: tuple[Bug, ...] = ()
        self.advance_to(step)

        self.bug: Bug | None = None
        self.trajectory: list[_Turn] = []
        self.turn_number = 0

    # ── curriculum ────────────────────────────────────────────────────────────

    @property
    def step(self) -> int:
        """The training step the curriculum is currently positioned at."""
        return self._step

    @property
    def active_tiers(self) -> tuple[int, ...]:
        """Tiers the environment is currently sampling from."""
        return self.schedule.tiers_at(self._step)

    def advance_to(self, step: int) -> tuple[int, ...]:
        """Move the curriculum to ``step`` and reload the bug pool if the tiers changed."""
        tiers = self.schedule.tiers_at(step)
        if not self.bugs or tiers != self.schedule.tiers_at(self._step):
            self.bugs = load_bugs(tiers)
        self._step = step
        return tiers

    # ── episode ───────────────────────────────────────────────────────────────

    def reset(self, bug: Bug | None = None) -> dict[str, Any]:
        """Start an episode on ``bug``, or on a random bug from the active tiers."""
        if bug is None:
            if not self.bugs:
                raise RuntimeError("No bugs loaded for the active curriculum tiers.")
            bug = self._random.choice(self.bugs)

        self.bug = bug
        self.trajectory = []
        self.turn_number = 0
        return self._observation(tests=None)

    def step_turn(self, raw_text: str) -> dict[str, Any]:
        """Score one structured response and advance the episode."""
        if self.bug is None:
            raise RuntimeError("No episode in progress. Call reset() first.")

        outcome = score_response(
            bug=self.bug,
            raw_text=raw_text,
            turn_number=self.turn_number,
            calculator=self.calculator,
        )
        self.trajectory.append(_Turn(turn=self.turn_number, outcome=outcome))
        self.turn_number += 1

        done = (
            outcome.solved
            or outcome.output.action == "give_up"
            or self.turn_number >= self.calculator.max_turns
        )
        return {
            "observation": self._observation(tests=outcome.tests),
            "reward": outcome.reward.total,
            "done": done,
            "info": {
                "reward_breakdown": outcome.reward.as_dict(),
                "solved": outcome.solved,
                "newly_broken": outcome.newly_broken,
                "action": outcome.output.action,
                "turn_number": self.turn_number,
                "bug_id": self.bug.id,
                "bug_tier": self.bug.tier,
            },
        }

    def episode_reward(self) -> float:
        """Discounted total reward for the episode so far."""
        return self.calculator.compute_episode_reward(
            [{"reward": turn.reward} for turn in self.trajectory]
        )

    def episode_metrics(self) -> dict[str, float]:
        """Mean per-component rewards for the episode, ready for W&B."""
        return self.calculator.mean_components(
            [{"reward": turn.reward} for turn in self.trajectory]
        )

    def _observation(self, tests: TestResults | None) -> dict[str, Any]:
        assert self.bug is not None
        return {
            "bug_id": self.bug.id,
            "buggy_code": self.bug.buggy_code,
            "error_message": self.bug.initial_error,
            "test_results": (
                tests.as_dict() if tests else {"passed": 0, "failed": 0, "total": len(self.bug.test_cases)}
            ),
            "turn_number": self.turn_number,
            "history": [
                {
                    "turn": turn.turn,
                    "action": turn.outcome.output.action,
                    "reward": turn.reward.total,
                }
                for turn in self.trajectory
            ],
        }
