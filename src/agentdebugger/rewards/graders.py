"""Episode graders for the hand-written tasks.

A grader answers one question at the end of an episode: how well did the agent
debug this task, in [0, 1]? Graders are deterministic — the same episode always
scores the same — because they are used to rank models.

One rule underpins all three: **only code the agent actually submitted counts.**
An agent that submits nothing scores 0.0, even on the hard task where the buggy
code already passes every sequential test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from agentdebugger.protocol import FixAttempt
from agentdebugger.sandbox import execute
from agentdebugger.tasks import Task

#: Threads and per-thread increments used by the concurrency stress test. Sized
#: so a racy counter loses updates on every run (see tasks/hard.py) while the
#: whole run still finishes in well under the sandbox's wall-clock limit.
STRESS_THREADS = 16
STRESS_INCREMENTS_PER_THREAD = 2_000
#: How many times the stress test is repeated before a fix is called thread-safe.
STRESS_REPEATS = 3

_STRESS_TEST = f"""
import threading

threading.stack_size(131072)

_THREADS = {STRESS_THREADS}
_PER_THREAD = {STRESS_INCREMENTS_PER_THREAD}
_counter = ConnectionCounter()


def _hammer():
    for _ in range(_PER_THREAD):
        _counter.increment()


_threads = [threading.Thread(target=_hammer) for _ in range(_THREADS)]
for _thread in _threads:
    _thread.start()
for _thread in _threads:
    _thread.join()

_expected = _THREADS * _PER_THREAD
_actual = _counter.get_count()
if _actual == _expected:
    print("CONCURRENT PASS")
else:
    print("CONCURRENT FAIL: counted %d of %d" % (_actual, _expected))
"""


@dataclass(frozen=True)
class Episode:
    """The record a grader scores."""

    attempts: tuple[FixAttempt, ...] = field(default=())
    hypotheses: tuple[str, ...] = field(default=())

    @property
    def attempts_used(self) -> int:
        return len(self.attempts)

    @property
    def best_tests_passed(self) -> int:
        """Best result across *submitted* fixes. Zero if the agent never submitted one."""
        return max((a.tests_passed for a in self.attempts), default=0)

    @property
    def best_attempt(self) -> FixAttempt | None:
        """The submission that passed the most tests; ties go to the later attempt."""
        if not self.attempts:
            return None
        return max(self.attempts, key=lambda a: (a.tests_passed, a.attempt_number))


class Grader:
    """Base grader: test pass ratio, efficiency, hypothesis accuracy, early-solve bonus."""

    TEST_WEIGHT = 0.60
    EFFICIENCY_WEIGHT = 0.20
    HYPOTHESIS_WEIGHT = 0.15
    EARLY_SOLVE_BONUS = 0.05

    def score(self, task: Task, episode: Episode) -> float:
        """Return this episode's score in [0, 1]."""
        pass_ratio = _ratio(episode.best_tests_passed, task.tests_total)
        solved = episode.best_tests_passed == task.tests_total and task.tests_total > 0

        total = (
            pass_ratio * self.TEST_WEIGHT
            + self._efficiency(task, episode, solved) * self.EFFICIENCY_WEIGHT
            + self._hypothesis_accuracy(task, episode) * self.HYPOTHESIS_WEIGHT
            + (self.EARLY_SOLVE_BONUS if solved and self._solved_early(task, episode) else 0.0)
        )
        return _clamp(total)

    def is_solved(self, task: Task, attempt: FixAttempt) -> bool:
        """Has this submission solved the task?

        The environment asks the grader rather than comparing test counts itself,
        because "all tests pass" is not the same as "solved" on every task — see
        :class:`ConcurrencyGrader`.
        """
        return attempt.tests_passed == task.tests_total and task.tests_total > 0

    def _efficiency(self, task: Task, episode: Episode, solved: bool) -> float:
        """Fraction of the attempt budget left unspent — but only once the task is solved.

        Paying for unused attempts on an unsolved task would reward doing
        nothing: an agent that never submits a fix has spent no attempts at all.
        """
        if not solved or task.max_attempts <= 0:
            return 0.0
        unused = task.max_attempts - episode.attempts_used
        return max(0.0, unused / task.max_attempts)

    def _solved_early(self, task: Task, episode: Episode) -> bool:
        return episode.attempts_used <= math.ceil(task.max_attempts / 3)

    def _hypothesis_accuracy(self, task: Task, episode: Episode) -> float:
        """Mean accuracy over every hypothesis the agent stated."""
        if not episode.hypotheses:
            return 0.0
        scores = [self._score_hypothesis(task, h) for h in episode.hypotheses]
        return sum(scores) / len(scores)

    def _score_hypothesis(self, task: Task, hypothesis: str) -> float:
        """Credit a hypothesis that names any ground-truth keyword."""
        lowered = hypothesis.lower()
        keywords = task.ground_truth.hypothesis_keywords
        return 1.0 if any(kw.lower() in lowered for kw in keywords) else 0.0


class RedHerringGrader(Grader):
    """Grader for the medium task, where *what* the agent blames is the whole point.

    Naming ``hash_password`` and saying why earns full credit. Naming it without
    explaining anything earns half. Blaming only ``authenticate_user`` — the
    function every error message points at, and the one function that is
    correct — earns nothing.
    """

    def _score_hypothesis(self, task: Task, hypothesis: str) -> float:
        lowered = hypothesis.lower()
        truth = task.ground_truth
        root_cause = truth.bug_location.lower()
        red_herring = (truth.red_herring_keyword or "").lower()

        names_root_cause = root_cause in lowered
        supporting = [kw for kw in truth.hypothesis_keywords if kw.lower() != root_cause]
        has_detail = any(kw.lower() in lowered for kw in supporting)

        if names_root_cause and has_detail:
            return 1.0
        if names_root_cause:
            return 0.5
        if red_herring and red_herring in lowered:
            return 0.0
        return 0.1  # generic but not actively wrong


class ConcurrencyGrader(Grader):
    """Grader for the hard task: passing the sequential suite is not enough.

    The sequential tests are worth less than the concurrency check, because the
    buggy code already passes all of them. Efficiency is only paid out to an
    agent that actually fixed the race — otherwise a confident wrong answer in
    one attempt would outscore a correct answer in three.
    """

    SEQUENTIAL_WEIGHT = 0.40
    CONCURRENT_WEIGHT = 0.30
    HYPOTHESIS_WEIGHT = 0.20
    EFFICIENCY_BONUS = 0.10
    EFFICIENT_ATTEMPTS = 5

    def is_solved(self, task: Task, attempt: FixAttempt) -> bool:
        """Passing all eight sequential tests is not enough — the buggy code does that.

        A submission counts as solved only if it also survives the concurrency
        stress test. This is what stops the environment from paying out its
        solve bonus for a fix that changed nothing.
        """
        if attempt.tests_passed != task.tests_total:
            return False
        return self.survives_stress_test(attempt.code_submitted, task)

    def score(self, task: Task, episode: Episode) -> float:
        sequential = _ratio(episode.best_tests_passed, task.tests_total) * self.SEQUENTIAL_WEIGHT
        concurrent = self._concurrency_score(task, episode)
        hypothesis = self._hypothesis_accuracy(task, episode) * self.HYPOTHESIS_WEIGHT

        thread_safe = concurrent == self.CONCURRENT_WEIGHT
        efficient = episode.attempts_used <= self.EFFICIENT_ATTEMPTS
        efficiency = self.EFFICIENCY_BONUS if thread_safe and efficient else 0.0

        return _clamp(sequential + concurrent + hypothesis + efficiency)

    def _concurrency_score(self, task: Task, episode: Episode) -> float:
        """Full credit only for a fix that survives every repeat of the stress test."""
        best = episode.best_attempt
        if best is None or not best.code_submitted.strip():
            return 0.0

        passes = sum(
            self.survives_stress_test(best.code_submitted, task) for _ in range(STRESS_REPEATS)
        )
        if passes == STRESS_REPEATS:
            return self.CONCURRENT_WEIGHT
        if passes > 0:
            return self.CONCURRENT_WEIGHT / 2  # partially synchronised, still loses updates
        return 0.0

    @staticmethod
    def survives_stress_test(code: str, task: Task) -> bool:
        """Hammer the submitted counter from many threads; True if no update is lost.

        Runs through the sandbox like any other submission — grading never
        executes agent code in this process.
        """
        result = execute(code, _STRESS_TEST, policy=task.policy)
        if result.timed_out or result.blocked:
            return False
        return "CONCURRENT PASS" in result.output and "CONCURRENT FAIL" not in result.output


GRADERS: dict[str, Grader] = {
    "easy": Grader(),
    "medium": RedHerringGrader(),
    "hard": ConcurrencyGrader(),
}


def get_grader(task_id: str) -> Grader:
    """Return the grader for a task id."""
    try:
        return GRADERS[task_id]
    except KeyError:
        raise ValueError(f"No grader for task_id {task_id!r}") from None


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
