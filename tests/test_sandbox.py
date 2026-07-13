"""The sandbox must contain model-generated code, and must not get in its way.

Both halves matter. A sandbox that blocks `import os` but also blocks
`import hashlib` is not secure, it is broken — and that was a real bug in the
first version of this environment: nullifying `exec` in ``builtins`` to stop
code injection also broke CPython's import machinery, so any module that was not
already loaded failed to import. The medium and hard tasks were unsolvable as a
result. ``test_allows_the_stdlib_a_fix_legitimately_needs`` is the regression
test for that.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from agentdebugger.config import SandboxLimits
from agentdebugger.sandbox import SandboxPolicy, analyze, execute, run_test_cases
from agentdebugger.sandbox.runner import HAS_RLIMITS

# ── it runs ordinary code ─────────────────────────────────────────────────────


def test_runs_clean_code():
    result = execute("def add(a, b):\n    return a + b", "print(add(2, 3))")
    assert result.output == "5"
    assert not result.blocked
    assert not result.timed_out
    assert result.exit_code == 0


def test_surfaces_syntax_errors_as_output_not_as_a_security_event():
    result = execute("def broken(: pass")
    assert "SyntaxError" in result.output
    assert not result.blocked


def test_traceback_points_at_the_agents_own_line_numbers():
    """The agent has to be able to read the traceback, so sandbox frames are stripped."""
    result = execute("def f():\n    raise ValueError('boom')\n", "f()")
    assert "ValueError: boom" in result.output
    assert "bootstrap" not in result.output
    assert "runpy" not in result.output


@pytest.mark.parametrize("module", ["hashlib", "math", "collections", "itertools", "random", "re"])
def test_allows_the_stdlib_a_fix_legitimately_needs(module):
    """Regression: a sandbox that breaks `import hashlib` makes the medium task unsolvable."""
    result = execute(f"import {module}\nprint('imported', {module}.__name__)")
    assert f"imported {module}" in result.output
    assert not result.blocked


def test_allows_private_attributes_and_super():
    """`self._lock` and `super().__init__()` are ordinary Python, not escape attempts."""
    code = (
        "class Base:\n"
        "    def __init__(self):\n"
        "        self._value = 1\n"
        "class Child(Base):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self._doubled = self._value * 2\n"
    )
    result = execute(code, "print(Child()._doubled)")
    assert result.output == "2"
    assert not result.blocked


def test_threading_is_available_only_when_the_task_allows_it():
    code = "import threading\nprint('threads', threading.active_count())"

    assert execute(code).blocked

    permitted = execute(code, policy=SandboxPolicy().allowing("threading"))
    assert not permitted.blocked
    assert "threads" in permitted.output


# ── it blocks escapes ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "attack",
    [
        pytest.param("import os\nos.system('echo pwned')", id="import-os"),
        pytest.param("import subprocess\nsubprocess.run(['echo', 'pwned'])", id="import-subprocess"),
        pytest.param("from os import system\nsystem('echo pwned')", id="from-import"),
        pytest.param("import socket\nsocket.socket()", id="network"),
        pytest.param("import importlib\nimportlib.import_module('os')", id="importlib"),
        pytest.param("__import__('os').system('echo pwned')", id="dunder-import"),
        pytest.param("eval('__import__(\"os\").system(\"echo pwned\")')", id="eval"),
        pytest.param("exec('import os')", id="exec"),
        pytest.param("open('/etc/passwd').read()", id="open-read"),
        pytest.param("open('/tmp/pwned', 'w').write('x')", id="open-write"),
        pytest.param("().__class__.__base__.__subclasses__()", id="subclasses-walk"),
        pytest.param("(lambda: 0).__globals__['__builtins__']", id="globals-walk"),
        pytest.param("getattr(__builtins__, 'open')('/etc/passwd')", id="getattr-builtins"),
        pytest.param("import ctypes\nctypes.CDLL(None)", id="ctypes"),
        pytest.param("import pickle\npickle.loads(b'')", id="pickle"),
    ],
)
def test_escape_attempts_are_refused_before_execution(attack):
    result = execute(attack)
    assert result.blocked, f"not blocked: {attack}"
    assert result.exit_code is None, "blocked code must never reach a subprocess"
    assert "pwned" not in result.output


def test_a_blocked_program_is_never_executed():
    """Blocking has to happen before the process starts, not after the damage is done."""
    marker = os.path.join(tempfile.gettempdir(), "agentdebugger-escape-canary")
    if os.path.exists(marker):
        os.unlink(marker)

    result = execute(f"import os\nos.system('touch {marker}')\n")

    assert result.blocked
    assert not os.path.exists(marker), "sandboxed code touched the filesystem"


def test_violations_name_what_was_refused_and_where():
    violations = analyze("import os\nx = eval('1')\n", SandboxPolicy())
    rendered = [str(v) for v in violations]
    assert "line 1: import of blocked module 'os'" in rendered
    assert "line 2: use of blocked builtin 'eval'" in rendered


# ── it enforces limits ────────────────────────────────────────────────────────


def test_wall_clock_timeout_is_enforced(fast_policy):
    result = execute("while True:\n    pass", policy=fast_policy)
    assert result.timed_out
    assert "TIMEOUT" in result.output
    assert result.duration_ms < 6000, "the deadline must not overshoot by seconds"


def test_a_timeout_kills_the_whole_process_group(fast_policy):
    """A fix that spawns threads must not outlive the process that ran it."""
    policy = fast_policy.allowing("threading")
    code = (
        "import threading\n"
        "def spin():\n"
        "    while True:\n"
        "        pass\n"
        "for _ in range(4):\n"
        "    t = threading.Thread(target=spin, daemon=False)\n"
        "    t.start()\n"
        "while True:\n"
        "    pass\n"
    )
    result = execute(code, policy=policy)
    assert result.timed_out
    assert result.exit_code is not None and result.exit_code < 0, "expected a signal, not an exit"


@pytest.mark.skipif(not HAS_RLIMITS, reason="setrlimit is unavailable on this platform")
def test_memory_limit_is_enforced():
    policy = SandboxPolicy(limits=SandboxLimits(memory_mb=64))
    result = execute("data = bytearray(256 * 1024 * 1024)\nprint('allocated', len(data))", policy=policy)
    assert "allocated" not in result.output
    assert "MemoryError" in result.output
    assert not result.timed_out, "a memory limit should fail fast, not hang"


@pytest.mark.skipif(not HAS_RLIMITS, reason="setrlimit is unavailable on this platform")
def test_cpu_limit_backstops_a_process_that_ignores_signals():
    limits = SandboxLimits(wall_clock_seconds=30.0, cpu_seconds=1)
    result = execute("while True:\n    pass", policy=SandboxPolicy(limits=limits))
    assert not result.timed_out, "the CPU limit should fire before the wall clock"
    assert result.duration_ms < 10_000


def test_output_is_truncated_rather_than_unbounded():
    policy = SandboxPolicy(limits=SandboxLimits(max_output_chars=500))
    result = execute("for i in range(10000):\n    print('x' * 40)", policy=policy)
    assert len(result.output) < 700
    assert "truncated" in result.output


# ── it cleans up and stays deterministic ──────────────────────────────────────


def test_temporary_files_are_removed_after_execution():
    before = set(os.listdir(tempfile.gettempdir()))
    execute("print('hello')")
    execute("while True:\n    pass", policy=SandboxPolicy(limits=SandboxLimits(wall_clock_seconds=1)))
    leaked = {
        name
        for name in set(os.listdir(tempfile.gettempdir())) - before
        if name.startswith("agentdebugger-sandbox-")
    }
    assert not leaked, f"sandbox left files behind: {leaked}"


def test_executions_do_not_leak_state_into_each_other():
    execute("leaked = 'secret'", "print('first')")
    result = execute("", "try:\n    print(leaked)\nexcept NameError:\n    print('ISOLATED')")
    assert "ISOLATED" in result.output


def test_output_is_identical_across_runs():
    """Rewards are computed from this text, so it must not carry a random temp path."""
    outputs = {execute("raise ValueError('boom')").output for _ in range(3)}
    assert len(outputs) == 1
    assert "<sandbox>" in outputs.pop()


# ── the shared test-case runner ───────────────────────────────────────────────


def test_run_test_cases_reports_per_case_outcomes():
    cases = [
        {"input": [1, 2], "expected_output": 3},
        {"input": [2, 2], "expected_output": 5},  # wrong on purpose
        {"input": [0, 0], "expected_output": 0},
    ]
    results = run_test_cases("def add(a, b):\n    return a + b", "add", cases)
    assert results.outcomes == (True, False, True)
    assert (results.passed, results.failed, results.total) == (2, 1, 3)


def test_run_test_cases_counts_a_crash_as_a_failure_not_an_error():
    cases = [{"input": [1], "expected_output": 1}]
    results = run_test_cases("def f(x):\n    raise RuntimeError('boom')", "f", cases)
    assert results.passed == 0


def test_run_test_cases_counts_an_infinite_loop_as_a_failure():
    cases = [{"input": [1], "expected_output": 1}]
    policy = SandboxPolicy(limits=SandboxLimits(wall_clock_seconds=2.0, cpu_seconds=2))
    results = run_test_cases("def f(x):\n    while True:\n        pass", "f", cases, policy=policy)
    assert results.passed == 0
    assert results.timed_out


def test_newly_broken_counts_regressions_not_failures():
    """A fix that trades a passing test for a failing one has done damage a count would hide."""
    cases = [
        {"input": [1], "expected_output": 1},
        {"input": [2], "expected_output": 2},
    ]
    baseline = run_test_cases("def f(x):\n    return 1", "f", cases)  # passes case 1 only
    candidate = run_test_cases("def f(x):\n    return 2", "f", cases)  # passes case 2 only

    assert baseline.passed == candidate.passed == 1
    assert candidate.newly_broken(baseline) == 1
