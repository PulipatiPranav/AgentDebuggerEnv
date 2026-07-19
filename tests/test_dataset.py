"""The bug dataset backs the reported solve rates, so its integrity is a claim to defend.

``test_every_bug_is_a_real_bug`` is the important one: it executes every record's
reference fix and buggy code in the sandbox and asserts the fix passes and the
bug fails. If it is green, the 90 numbers in the results file are measuring
something real.
"""

from __future__ import annotations

import pytest

from agentdebugger.config import TIERS
from agentdebugger.dataset import (
    load_bugs,
    load_split,
    load_tier,
    tier_counts,
    validate_bug,
    validate_tiers,
)


def test_the_dataset_ships_one_hundred_eighty_bugs_across_three_tiers():
    counts = tier_counts()
    assert counts == {1: 60, 2: 60, 3: 60}
    assert sum(counts.values()) == 180


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
    assert len(load_tier(1)) == 60


def test_train_and_heldout_splits_partition_the_dataset_without_overlap():
    """Every bug is in exactly one of train/held-out, and together they are the whole set."""
    train = set(load_split("train"))
    heldout = set(load_split("heldout"))
    everything = {bug.id for bug in load_bugs()}

    assert train.isdisjoint(heldout), "a bug may not be in both train and held-out"
    assert train | heldout == everything, "the split must cover every bug"
    assert load_split("all") is None  # 'all' means no filter


def test_loading_a_split_restricts_the_bugs_returned():
    heldout_ids = load_split("heldout")
    heldout_bugs = load_bugs(split="heldout")
    assert len(heldout_bugs) == len(heldout_ids)
    assert all(bug.id in heldout_ids for bug in heldout_bugs)
    # train + held-out recover the full set
    assert len(load_bugs(split="train")) + len(heldout_bugs) == len(load_bugs())


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
