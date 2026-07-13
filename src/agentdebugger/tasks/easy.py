"""Easy task — an off-by-one bug in binary search.

One function, one bug, a failing test that names the symptom precisely. An agent
that reads the error output should solve this in one attempt; it exists to
establish a floor, not to discriminate between strong models.
"""

from agentdebugger.tasks.harness import build_test_runner
from agentdebugger.tasks.models import GroundTruth, Task

BUGGY_CODE = '''def binary_search(arr: list, target: int) -> int:
    """Return the index of target in sorted arr, or -1 if not found."""
    left, right = 0, len(arr) - 1
    while left < right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
'''

FIXED_CODE = '''def binary_search(arr: list, target: int) -> int:
    """Return the index of target in sorted arr, or -1 if not found."""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
'''

TEST_SUITE = '''def test_finds_first_element():
    assert binary_search([1, 3, 5, 7, 9], 1) == 0

def test_finds_middle_element():
    assert binary_search([1, 3, 5, 7, 9], 5) == 2

def test_finds_last_element():
    assert binary_search([1, 3, 5, 7, 9], 9) == 4

def test_returns_minus_one_for_missing():
    assert binary_search([1, 3, 5, 7, 9], 4) == -1

def test_single_element_found():
    assert binary_search([42], 42) == 0

def test_single_element_not_found():
    assert binary_search([42], 7) == -1

def test_empty_list():
    assert binary_search([], 5) == -1

def test_finds_second_to_last():
    assert binary_search([2, 4, 6, 8, 10], 8) == 3
'''

TEST_NAMES = (
    "test_finds_first_element",
    "test_finds_middle_element",
    "test_finds_last_element",
    "test_returns_minus_one_for_missing",
    "test_single_element_found",
    "test_single_element_not_found",
    "test_empty_list",
    "test_finds_second_to_last",
)

TASK = Task(
    task_id="easy",
    name="Off-by-one in binary search",
    difficulty="easy",
    description=(
        "A data-processing utility module contains a binary search over a sorted list. "
        "It should return the index of the target, or -1 when the target is absent. "
        "Some tests fail. Find the root cause, state a hypothesis, and fix it."
    ),
    buggy_code=BUGGY_CODE,
    test_suite=TEST_SUITE,
    test_runner=build_test_runner(TEST_NAMES),
    tests_total=len(TEST_NAMES),
    max_attempts=5,
    max_steps=8,
    ground_truth=GroundTruth(
        bug_location="binary_search",
        bug_type="off_by_one",
        hypothesis_keywords=(
            "left <= right",
            "termination",
            "last element",
            "off by one",
            "off-by-one",
            "<=",
        ),
        fixed_code=FIXED_CODE,
        reference_hypothesis=(
            "The loop condition is `left < right`, so the loop exits while left == right "
            "without ever examining that final candidate index. Any target that binary "
            "search narrows down to a single remaining slot — the last element, or a "
            "one-element list — is reported as missing. The termination condition should "
            "be `left <= right`, which is the classic off-by-one in this algorithm."
        ),
    ),
)
