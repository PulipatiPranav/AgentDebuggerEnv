"""Run a candidate function against input/expected-output test cases.

This is the single place where "does this fix actually work?" is answered. Every
caller — the environments, the GRPO reward function, the dataset validator —
goes through :func:`run_test_cases`, so they cannot drift apart.

All cases run inside one sandboxed process. Spawning a process per case is both
slower and less faithful: a fix with import-time side effects should be paid for
once, exactly as it would be in a real test run.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agentdebugger.sandbox.policy import SandboxPolicy
from agentdebugger.sandbox.runner import ExecutionResult, execute

_RESULT_LINE = re.compile(r"^SANDBOX-RESULTS ([01]*)$", re.MULTILINE)


@dataclass(frozen=True)
class TestResults:
    """How a candidate fix scored against a bug's test cases.

    ``outcomes`` holds one flag per case, in order, which is what makes it
    possible to tell a fix that *broke* a passing test apart from one that
    merely failed to fix a broken one.
    """

    total: int
    outcomes: tuple[bool, ...] = field(default=())
    output: str = ""
    timed_out: bool = False
    blocked: bool = False

    @property
    def passed(self) -> int:
        return sum(self.outcomes)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def all_passed(self) -> bool:
        return self.total > 0 and self.passed == self.total

    def newly_broken(self, baseline: TestResults) -> int:
        """How many cases passed in ``baseline`` but fail here.

        A fix that trades one passing test for another has done real damage that
        a pass *count* alone would hide.
        """
        return sum(
            was_passing and not is_passing
            for was_passing, is_passing in zip(baseline.outcomes, self.outcomes, strict=False)
        )

    def as_dict(self, baseline: TestResults | None = None) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failed": self.failed,
            "total": self.total,
            "newly_broken": self.newly_broken(baseline) if baseline else 0,
            "timed_out": self.timed_out,
            "blocked": self.blocked,
        }


_HARNESS = '''
_CASES = {cases!r}
_OUTCOMES = []
for _args, _expected in _CASES:
    try:
        _ok = {function}(*_args) == _expected
    except BaseException:
        _ok = False
    _OUTCOMES.append("1" if _ok else "0")
print("SANDBOX-RESULTS " + "".join(_OUTCOMES))
'''


def run_test_cases(
    code: str,
    function: str,
    cases: Sequence[Mapping[str, Any]],
    policy: SandboxPolicy | None = None,
) -> TestResults:
    """Execute ``code`` and call ``function`` once per case, comparing results.

    Each case is a mapping with an ``input`` list (splatted as positional
    arguments) and an ``expected_output`` value. A case counts as failed if the
    call raises, returns the wrong value, or never runs because the process was
    killed.
    """
    if not code.strip() or not function or not cases:
        return TestResults(total=len(cases), outcomes=(False,) * len(cases))

    payload = [(tuple(case["input"]), case["expected_output"]) for case in cases]
    harness = _HARNESS.format(cases=payload, function=function)
    result = execute(code, harness, policy=policy)

    return TestResults(
        total=len(cases),
        outcomes=_parse_outcomes(result, len(cases)),
        output=result.output,
        timed_out=result.timed_out,
        blocked=result.blocked,
    )


def _parse_outcomes(result: ExecutionResult, total: int) -> tuple[bool, ...]:
    """Read the harness result line. A crashed, blocked or killed process passed nothing."""
    matches = _RESULT_LINE.findall(result.output)
    if not matches or len(matches[-1]) != total:
        return (False,) * total
    return tuple(flag == "1" for flag in matches[-1])
