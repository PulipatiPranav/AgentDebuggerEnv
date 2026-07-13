"""Check that every bug in the dataset is actually a bug.

Two invariants have to hold for a record to be usable as a training signal, and
they are easy to break by hand-editing JSONL:

1. ``original_code`` passes **every** test case. Otherwise the reference fix is
   wrong and the reward function is chasing a target that does not exist.
2. ``buggy_code`` fails **at least one** test case. Otherwise the "bug" is a
   no-op and the model is rewarded for changing nothing.

Both are checked by execution, in the sandbox, so a malformed record cannot
break the checker.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentdebugger.config import TIERS, SandboxLimits
from agentdebugger.dataset.loader import load_tier
from agentdebugger.dataset.models import Bug
from agentdebugger.sandbox import SandboxPolicy, run_test_cases

#: Dataset bugs are tiny; a reference fix that needs more than a few seconds is a
#: bug in the dataset, not a slow test. A short deadline also stops a buggy record
#: that loops forever from stalling validation of the other 89.
_VALIDATION_POLICY = SandboxPolicy(limits=SandboxLimits(wall_clock_seconds=3.0, cpu_seconds=3))


@dataclass(frozen=True)
class BugReport:
    """What validation found wrong with one bug. Empty ``problems`` means it is sound."""

    bug_id: str
    tier: int
    problems: tuple[str, ...] = field(default=())

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass(frozen=True)
class ValidationReport:
    """The result of validating one or more tiers."""

    reports: tuple[BugReport, ...]

    @property
    def ok(self) -> bool:
        return all(report.ok for report in self.reports)

    @property
    def failures(self) -> tuple[BugReport, ...]:
        return tuple(report for report in self.reports if not report.ok)

    @property
    def total(self) -> int:
        return len(self.reports)


def validate_bug(bug: Bug) -> BugReport:
    """Execute a bug's reference fix and its buggy code against its test cases."""
    problems: list[str] = []

    if not bug.test_cases:
        problems.append("has no test cases")
        return BugReport(bug_id=bug.id, tier=bug.tier, problems=tuple(problems))

    cases = [case.as_dict() for case in bug.test_cases]

    reference = run_test_cases(
        bug.original_code, bug.function_name, cases, policy=_VALIDATION_POLICY
    )
    if not reference.all_passed:
        problems.append(
            f"original_code fails {reference.failed}/{reference.total} of its own test cases"
        )

    buggy = run_test_cases(bug.buggy_code, bug.function_name, cases, policy=_VALIDATION_POLICY)
    if buggy.all_passed:
        problems.append("buggy_code passes every test case, so there is nothing to fix")

    return BugReport(bug_id=bug.id, tier=bug.tier, problems=tuple(problems))


def validate_tiers(tiers: tuple[int, ...] = TIERS) -> ValidationReport:
    """Validate every bug in the given tiers."""
    reports = [validate_bug(bug) for tier in tiers for bug in load_tier(tier)]
    return ValidationReport(reports=tuple(reports))
