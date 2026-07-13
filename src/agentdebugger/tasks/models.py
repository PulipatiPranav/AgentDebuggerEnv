"""The task model shared by every hand-written debugging task."""

from __future__ import annotations

from dataclasses import dataclass, field

from agentdebugger.config import SandboxLimits
from agentdebugger.sandbox import SandboxPolicy


@dataclass(frozen=True)
class GroundTruth:
    """What the environment knows about a bug and the agent does not."""

    bug_location: str
    bug_type: str
    #: A hypothesis counts as correct if it mentions any of these.
    hypothesis_keywords: tuple[str, ...]
    #: The reference fix. Used by the oracle agent and by the solvability tests;
    #: never shown to a learning agent.
    fixed_code: str
    #: A human's diagnosis of the bug, in the words a good agent would use.
    #: The oracle agent states this; it is also the worked example in the docs.
    reference_hypothesis: str
    #: A term that names the *symptom's* location rather than the bug's. A
    #: hypothesis that blames only this has followed a red herring.
    red_herring_keyword: str | None = None


@dataclass(frozen=True)
class Task:
    """A single debugging task: broken code, its tests, and how it is scored."""

    task_id: str
    name: str
    difficulty: str
    description: str
    buggy_code: str
    #: The test suite as the agent reads it.
    test_suite: str
    #: A harness appended after the suite that runs it and prints
    #: ``"<n> passed, <m> failed"``. Kept separate so the agent is not shown
    #: environment plumbing.
    test_runner: str
    tests_total: int
    max_attempts: int
    max_steps: int
    ground_truth: GroundTruth
    #: Modules this task's code is allowed to import beyond the default denylist.
    allowed_imports: tuple[str, ...] = field(default=())
    limits: SandboxLimits = field(default_factory=SandboxLimits)

    @property
    def policy(self) -> SandboxPolicy:
        """The sandbox policy this task's code runs under."""
        return SandboxPolicy(limits=self.limits).allowing(*self.allowed_imports)

    @property
    def executable_tests(self) -> str:
        """Suite plus harness, ready to append to a candidate fix."""
        return f"{self.test_suite}\n\n{self.test_runner}"
