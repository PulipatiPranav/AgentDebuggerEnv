"""Hard task — a race condition that every sequential test misses.

``ConnectionCounter.increment`` reads ``self.count``, calls ``_apply`` to clamp
the result, then writes the value back. Under CPython's GIL a thread can be
preempted at a function call, so two threads can read the same count and write
back the same value: one increment is lost.

The point of the task is that **all eight tests pass on the buggy code**. An
agent that trusts a green test suite scores nothing here. To solve it, the agent
has to notice that a passing sequential suite is not evidence of thread safety,
and guard the read-modify-write with a lock.

Note on the code shape: a bare ``self.count += 1`` would *not* race on CPython
3.12+, because the interpreter only checks for a thread switch at calls and
backward jumps, never between the load and the store of an in-place add. The
call to ``_apply`` is what makes the window real — and it is exactly the kind of
innocuous helper that hides a race in production code.
"""

from agentdebugger.config import SandboxLimits
from agentdebugger.tasks.harness import build_test_runner
from agentdebugger.tasks.models import GroundTruth, Task

BUGGY_CODE = '''class ConnectionCounter:
    """Tracks the number of open connections for a web server."""

    def __init__(self):
        self.count = 0

    def _apply(self, current: int, delta: int) -> int:
        """Apply a delta to the count, clamping at zero."""
        return max(0, current + delta)

    def increment(self):
        """Record a new connection."""
        self.count = self._apply(self.count, 1)

    def decrement(self):
        """Record a closed connection."""
        self.count = self._apply(self.count, -1)

    def get_count(self) -> int:
        """Return the number of open connections."""
        return self.count

    def reset(self):
        """Drop the count back to zero."""
        self.count = 0
'''

FIXED_CODE = '''import threading


class ConnectionCounter:
    """Tracks the number of open connections for a web server."""

    def __init__(self):
        self.count = 0
        self._lock = threading.Lock()

    def _apply(self, current: int, delta: int) -> int:
        """Apply a delta to the count, clamping at zero."""
        return max(0, current + delta)

    def increment(self):
        """Record a new connection."""
        with self._lock:
            self.count = self._apply(self.count, 1)

    def decrement(self):
        """Record a closed connection."""
        with self._lock:
            self.count = self._apply(self.count, -1)

    def get_count(self) -> int:
        """Return the number of open connections."""
        with self._lock:
            return self.count

    def reset(self):
        """Drop the count back to zero."""
        with self._lock:
            self.count = 0
'''

TEST_SUITE = '''def test_initial_count_is_zero():
    assert ConnectionCounter().get_count() == 0

def test_single_increment():
    counter = ConnectionCounter()
    counter.increment()
    assert counter.get_count() == 1

def test_multiple_increments():
    counter = ConnectionCounter()
    for _ in range(5):
        counter.increment()
    assert counter.get_count() == 5

def test_increment_then_decrement():
    counter = ConnectionCounter()
    counter.increment()
    counter.increment()
    counter.decrement()
    assert counter.get_count() == 1

def test_decrement_clamps_at_zero():
    counter = ConnectionCounter()
    counter.decrement()
    assert counter.get_count() == 0

def test_multiple_decrements():
    counter = ConnectionCounter()
    for _ in range(3):
        counter.increment()
    for _ in range(2):
        counter.decrement()
    assert counter.get_count() == 1

def test_get_count_returns_int():
    assert isinstance(ConnectionCounter().get_count(), int)

def test_reset_works():
    counter = ConnectionCounter()
    for _ in range(5):
        counter.increment()
    counter.reset()
    assert counter.get_count() == 0
'''

TEST_NAMES = (
    "test_initial_count_is_zero",
    "test_single_increment",
    "test_multiple_increments",
    "test_increment_then_decrement",
    "test_decrement_clamps_at_zero",
    "test_multiple_decrements",
    "test_get_count_returns_int",
    "test_reset_works",
)

TASK = Task(
    task_id="hard",
    name="Race condition in a connection counter",
    difficulty="hard",
    description=(
        "A web server tracks open connections with a ConnectionCounter. Under production "
        "load the reported count drifts below the real number of connections, but the "
        "whole test suite passes and nobody can reproduce it locally. All eight tests "
        "pass on the code below. Work out what the tests do not cover, state a "
        "hypothesis, and fix it."
    ),
    buggy_code=BUGGY_CODE,
    test_suite=TEST_SUITE,
    test_runner=build_test_runner(TEST_NAMES),
    tests_total=len(TEST_NAMES),
    max_attempts=10,
    max_steps=25,
    ground_truth=GroundTruth(
        bug_location="increment and decrement",
        bug_type="race_condition",
        hypothesis_keywords=(
            "race condition",
            "race",
            "atomic",
            "lock",
            "read-modify-write",
            "thread-safe",
            "thread safe",
            "interleav",
            "synchroniz",
        ),
        fixed_code=FIXED_CODE,
        reference_hypothesis=(
            "Every test calls the counter from a single thread, so none of them can see "
            "the bug. increment() is a read-modify-write: it reads self.count, calls "
            "_apply, then writes the result back. That call is a point where CPython can "
            "switch threads, so two threads can read the same count and store the same "
            "value, losing an update — which is exactly the downward drift seen in "
            "production. This is a race condition; the read-modify-write in increment, "
            "decrement, get_count and reset must be made atomic with a threading.Lock."
        ),
    ),
    allowed_imports=("threading",),
    # Threads reserve address space for their stacks, so the concurrency stress
    # test needs more headroom than the default 256MB. The small switch interval
    # makes the lost-update race surface on every run rather than one in a
    # hundred; user code cannot undo it, since `sys` is not importable.
    limits=SandboxLimits(memory_mb=512, switch_interval=1e-6),
)
