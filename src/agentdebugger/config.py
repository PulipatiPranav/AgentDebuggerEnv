"""Central configuration: sandbox limits and the curriculum schedule.

Every tunable that more than one module needs lives here, so that the training
loop, the environments and the tests can never disagree about a threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Difficulty tiers shipped in the bug dataset.
TIERS: tuple[int, ...] = (1, 2, 3)


@dataclass(frozen=True)
class SandboxLimits:
    """Resource ceilings applied to every sandboxed execution.

    ``wall_clock_seconds`` is enforced by the parent process (it kills the whole
    child process group). ``cpu_seconds`` and ``memory_mb`` are enforced by the
    kernel via ``setrlimit`` in the child, so they hold even if the child stops
    responding to signals.
    """

    wall_clock_seconds: float = 10.0
    cpu_seconds: int = 10
    memory_mb: int = 256
    #: Maximum bytes the child may write to any file. 0 forbids file writes
    #: outright; stdout/stderr are pipes and are not affected by RLIMIT_FSIZE.
    max_file_write_bytes: int = 0
    #: stdout+stderr captured from the child are truncated to this many chars.
    max_output_chars: int = 20_000
    #: When set, the child's GIL switch interval (seconds). Shrinking it forces
    #: frequent thread preemption, so a non-atomic read-modify-write loses
    #: updates reliably instead of once in a hundred runs. Used by the
    #: concurrency grader; user code cannot undo it (``sys`` is not importable).
    switch_interval: float | None = None


@dataclass(frozen=True)
class CurriculumStage:
    """One stage of the training curriculum.

    The stage is active from ``start_step`` (inclusive) until the next stage's
    ``start_step``. ``tiers`` is the set of bug tiers sampled during the stage.
    """

    start_step: int
    tiers: tuple[int, ...]


@dataclass(frozen=True)
class CurriculumSchedule:
    """Which bug tiers are sampled at a given training step.

    The default is the schedule the published training run used: tier 1 only
    until step 150, tiers 1-2 until step 350, then all three tiers.
    """

    stages: tuple[CurriculumStage, ...] = field(
        default=(
            CurriculumStage(start_step=0, tiers=(1,)),
            CurriculumStage(start_step=150, tiers=(1, 2)),
            CurriculumStage(start_step=350, tiers=(1, 2, 3)),
        )
    )

    def __post_init__(self) -> None:
        steps = [stage.start_step for stage in self.stages]
        if not self.stages:
            raise ValueError("A curriculum needs at least one stage.")
        if steps[0] != 0:
            raise ValueError("The first curriculum stage must start at step 0.")
        if steps != sorted(steps) or len(set(steps)) != len(steps):
            raise ValueError("Curriculum stages must have strictly increasing start_step.")

    def tiers_at(self, step: int) -> tuple[int, ...]:
        """Return the tiers that are active at ``step``."""
        if step < 0:
            raise ValueError(f"step must be non-negative, got {step}")
        active = self.stages[0]
        for stage in self.stages:
            if step >= stage.start_step:
                active = stage
            else:
                break
        return active.tiers

    def advances_at(self) -> tuple[int, ...]:
        """Steps at which the active tier set changes (excluding step 0)."""
        return tuple(stage.start_step for stage in self.stages[1:])


DEFAULT_LIMITS = SandboxLimits()
DEFAULT_CURRICULUM = CurriculumSchedule()
