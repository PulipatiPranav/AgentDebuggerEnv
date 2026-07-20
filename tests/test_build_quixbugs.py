"""Unit tests for the QuixBugs adapter's pure logic (Phase A item 4).

These deliberately avoid the network: they exercise ``scripts/build_quixbugs.py``
against small, synthetic files instead of a real QuixBugs checkout, so they run
in the same fast, offline suite as everything else. Anything that needs an
actual clone (the end-to-end adaptation) was verified manually against a real
checkout; see the module docstring in ``scripts/build_quixbugs.py`` for why each
structural exclusion exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_quixbugs import (  # needs the sys.path tweak above
    _EXCLUDED,
    _drop_pathologically_slow_cases,
    _function_source,
)


def test_function_source_extracts_only_the_named_function(tmp_path):
    """The trailing problem-statement docstring QuixBugs embeds must be dropped."""
    program = tmp_path / "example.py"
    program.write_text(
        "def example(n):\n"
        "    return n + 1\n"
        "\n"
        '"""\n'
        "Example\n"
        "Input:\n"
        "    n: an int\n"
        '"""\n'
    )
    source = _function_source(program, "example")
    assert source.strip() == "def example(n):\n    return n + 1"
    assert "Input:" not in source


def test_function_source_keeps_needed_top_level_imports(tmp_path):
    """`to_base`-style programs need a module-level import in scope."""
    program = tmp_path / "example.py"
    program.write_text(
        "import string\n\ndef example(i):\n    return string.digits[i]\n\n" '"""doc"""\n'
    )
    source = _function_source(program, "example")
    assert "import string" in source
    namespace: dict = {}
    exec(source, namespace)
    assert namespace["example"](3) == "3"


def test_drop_pathologically_slow_cases_drops_a_hanging_case():
    correct_code = "def slow(should_hang):\n    while should_hang:\n        pass\n    return should_hang\n"
    cases = [([False], False), ([True], True)]
    kept = _drop_pathologically_slow_cases(correct_code, "slow", cases)
    # Only 1 case survives (< _MIN_CASES=3), so the whole bug is rejected.
    assert kept is None


def test_drop_pathologically_slow_cases_keeps_enough_fast_cases():
    correct_code = "def fast(x):\n    return x + 1\n"
    cases = [([i], i + 1) for i in range(5)]
    kept = _drop_pathologically_slow_cases(correct_code, "fast", cases)
    assert kept == cases


def test_drop_pathologically_slow_cases_does_not_mutate_the_callers_cases():
    """A program that mutates its argument in place (e.g. next_palindrome) must
    not leave that mutation behind in the case object the build script reuses
    for the real record — that was a real bug caught while building this set.
    """
    correct_code = "def mutate_in_place(items):\n    items[0] = 999\n    return items\n"
    original_case = ([1, 2, 3], [999, 2, 3])
    cases = [original_case, ([4, 5, 6], [999, 5, 6]), ([7, 8, 9], [999, 8, 9])]
    _drop_pathologically_slow_cases(correct_code, "mutate_in_place", cases)
    assert cases[0][0] == [1, 2, 3]  # untouched, despite the probe call


def test_excluded_programs_have_non_empty_reasons():
    for name, reason in _EXCLUDED.items():
        assert name and reason


@pytest.mark.parametrize("name", ["flatten", "kheapsort", "hanoi", "sqrt"])
def test_specific_structural_exclusions_are_present(name):
    assert name in _EXCLUDED
