"""The bug record: one buggy function, its reference fix, and its test cases."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TestCase:
    """One call of the buggy function and the value it should return."""

    #: Positional arguments, splatted into the call.
    input: tuple[Any, ...]
    expected_output: Any

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> TestCase:
        return cls(input=tuple(raw["input"]), expected_output=raw["expected_output"])

    def as_dict(self) -> dict[str, Any]:
        return {"input": list(self.input), "expected_output": self.expected_output}


@dataclass(frozen=True)
class BugLocation:
    """Where the bug is. Used to score localization; never shown to the agent."""

    function: str = ""
    line_start: int = -1


@dataclass(frozen=True)
class Bug:
    """A single curriculum bug."""

    id: str
    tier: int
    bug_type: str
    function_name: str
    buggy_code: str
    #: The known-good implementation. Reference for semantic similarity, and the
    #: control in dataset validation — it must pass every test case.
    original_code: str
    initial_error: str
    location: BugLocation
    test_cases: tuple[TestCase, ...] = field(default=())

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Bug:
        """Build a bug from one JSONL record."""
        location = raw.get("bug_location") or {}
        return cls(
            id=raw["id"],
            tier=int(raw["difficulty"]),
            bug_type=raw["bug_type"],
            function_name=raw["function_name"],
            buggy_code=raw["buggy_code"],
            original_code=raw["original_code"],
            initial_error=raw.get("initial_error", "Some tests are failing."),
            location=BugLocation(
                function=location.get("function", ""),
                line_start=int(location.get("line_start", -1)),
            ),
            test_cases=tuple(TestCase.from_dict(case) for case in raw.get("test_cases", [])),
        )

    def as_dict(self) -> dict[str, Any]:
        """The shape the reward function and training loop consume."""
        return {
            "id": self.id,
            "difficulty": self.tier,
            "bug_type": self.bug_type,
            "function_name": self.function_name,
            "buggy_code": self.buggy_code,
            "original_code": self.original_code,
            "initial_error": self.initial_error,
            "bug_location": {
                "function": self.location.function,
                "line_start": self.location.line_start,
            },
            "test_cases": [case.as_dict() for case in self.test_cases],
        }
