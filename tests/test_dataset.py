"""The bug dataset backs the reported solve rates, so its integrity is a claim to defend.

``test_every_bug_is_a_real_bug`` is the important one: it executes every record's
reference fix and buggy code in the sandbox and asserts the fix passes and the
bug fails. If it is green, the 90 numbers in the results file are measuring
something real.
"""

from __future__ import annotations

import pytest

from agentdebugger.config import TIERS
from agentdebugger.dataset import load_bugs, load_tier, tier_counts, validate_bug, validate_tiers


def test_the_dataset_ships_ninety_bugs_across_three_tiers():
    counts = tier_counts()
    assert counts == {1: 40, 2: 30, 3: 20}
    assert sum(counts.values()) == 90


def test_every_bug_parses_and_carries_test_cases():
    for bug in load_bugs():
        assert bug.id
        assert bug.function_name
        assert bug.buggy_code and bug.original_code
        assert bug.test_cases, f"{bug.id} has no test cases"


def test_bug_ids_are_unique():
    ids = [bug.id for bug in load_bugs()]
    assert len(ids) == len(set(ids))


def test_the_loader_works_from_package_data_not_a_relative_path(tmp_path, monkeypatch):
    """load_bugs() must not depend on the current working directory."""
    monkeypatch.chdir(tmp_path)
    assert len(load_tier(1)) == 40


@pytest.mark.slow
@pytest.mark.parametrize("tier", TIERS)
def test_every_bug_is_a_real_bug(tier):
    """Reference fix passes all its cases; buggy code fails at least one. Checked by execution."""
    report = validate_tiers((tier,))
    assert report.ok, "\n".join(
        f"{f.bug_id}: {'; '.join(f.problems)}" for f in report.failures
    )


def test_validation_catches_a_fix_that_does_not_fix():
    from agentdebugger.dataset.models import Bug

    broken = Bug.from_dict(
        {
            "id": "synthetic",
            "difficulty": 1,
            "bug_type": "test",
            "function_name": "f",
            "buggy_code": "def f(x):\n    return x + 1",  # already correct → nothing to fix
            "original_code": "def f(x):\n    return x + 1",
            "bug_location": {"function": "f", "line_start": 1},
            "test_cases": [{"input": [1], "expected_output": 2}],
        }
    )
    report = validate_bug(broken)
    assert not report.ok
    assert any("nothing to fix" in problem for problem in report.problems)
