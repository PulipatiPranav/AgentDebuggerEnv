"""Episode graders rank models, so their scores have to be earned and deterministic.

The load-bearing claims:

* An agent that submits nothing scores zero — even on the hard task, where the
  buggy code already passes every sequential test.
* The red-herring grader pays for naming the real cause and refuses to pay for
  following the symptom.
* The concurrency grader only calls a fix solved if it survives the stress test,
  and it never runs agent code in this process.
"""

from __future__ import annotations

from agentdebugger.protocol import FixAttempt
from agentdebugger.rewards import get_grader
from agentdebugger.rewards.graders import ConcurrencyGrader, Episode, RedHerringGrader
from agentdebugger.tasks import get_task


def attempt(code="", hypothesis="", tests_passed=0, number=1, tests_total=8):
    return FixAttempt(
        attempt_number=number,
        hypothesis=hypothesis,
        code_submitted=code,
        execution_output="",
        tests_passed=tests_passed,
        tests_total=tests_total,
        execution_time_ms=10,
        timed_out=False,
    )


# ── the invariant every grader shares ─────────────────────────────────────────


def test_submitting_nothing_scores_zero_on_every_task():
    for task_id in ("easy", "medium", "hard"):
        task = get_task(task_id)
        score = get_grader(task_id).score(task, Episode())
        assert score == 0.0, f"{task_id} paid out for an empty episode"


def test_scores_are_bounded_to_the_unit_interval():
    task = get_task("easy")
    oracle = Episode(
        attempts=(attempt(task.ground_truth.fixed_code, "off-by-one", 8),),
        hypotheses=("binary_search off-by-one, use left <= right",),
    )
    score = get_grader("easy").score(task, oracle)
    assert 0.0 <= score <= 1.0


def test_grading_is_deterministic():
    task = get_task("hard")
    episode = Episode(
        attempts=(attempt(task.ground_truth.fixed_code, "race condition", 8),),
        hypotheses=("a race condition; the read-modify-write needs a lock",),
    )
    scores = {get_grader("hard").score(task, episode) for _ in range(3)}
    assert len(scores) == 1


# ── the reference fix defines the ceiling ─────────────────────────────────────


def test_the_reference_fix_scores_near_the_top_on_every_task():
    for task_id in ("easy", "medium", "hard"):
        task = get_task(task_id)
        episode = Episode(
            attempts=(attempt(task.ground_truth.fixed_code, task.ground_truth.reference_hypothesis, task.tests_total, tests_total=task.tests_total),),
            hypotheses=(task.ground_truth.reference_hypothesis,),
        )
        score = get_grader(task_id).score(task, episode)
        assert score >= 0.9, f"{task_id} reference fix only scored {score}"


# ── the red herring ───────────────────────────────────────────────────────────


def test_naming_the_root_cause_beats_following_the_red_herring():
    task = get_task("medium")
    grader = RedHerringGrader()

    root_cause = grader._score_hypothesis(
        task, "the bug is in hash_password, which wraps hexdigest in str(bytes(...))"
    )
    red_herring = grader._score_hypothesis(
        task, "authenticate_user is returning False when it should return True"
    )
    generic = grader._score_hypothesis(task, "there is a mismatch somewhere in the comparison")

    assert root_cause == 1.0
    assert red_herring == 0.0
    assert red_herring < generic < root_cause


def test_naming_the_cause_without_detail_earns_partial_credit():
    task = get_task("medium")
    score = RedHerringGrader()._score_hypothesis(task, "the problem is in hash_password")
    assert score == 0.5


# ── the concurrency grader ────────────────────────────────────────────────────


def test_a_locked_counter_survives_the_stress_test_and_a_racy_one_does_not():
    task = get_task("hard")
    grader = ConcurrencyGrader()

    assert grader.survives_stress_test(task.ground_truth.fixed_code, task) is True
    assert grader.survives_stress_test(task.buggy_code, task) is False


def test_passing_the_sequential_suite_is_not_enough_to_be_solved():
    """The buggy code passes all eight tests; it must not count as solved."""
    task = get_task("hard")
    grader = ConcurrencyGrader()

    buggy_but_green = attempt(task.buggy_code, "looks fine to me", tests_passed=8)
    truly_fixed = attempt(task.ground_truth.fixed_code, "race condition", tests_passed=8)

    assert grader.is_solved(task, buggy_but_green) is False
    assert grader.is_solved(task, truly_fixed) is True


def test_a_fix_that_only_passes_sequential_tests_scores_below_a_real_fix():
    task = get_task("hard")
    grader = ConcurrencyGrader()

    sequential_only = grader.score(
        task,
        Episode(
            attempts=(attempt(task.buggy_code, "no race here", 8),),
            hypotheses=("the counter looks correct",),
        ),
    )
    thread_safe = grader.score(
        task,
        Episode(
            attempts=(attempt(task.ground_truth.fixed_code, "race condition, needs a lock", 8),),
            hypotheses=("a race condition in the read-modify-write; add a lock",),
        ),
    )
    assert sequential_only < thread_safe
