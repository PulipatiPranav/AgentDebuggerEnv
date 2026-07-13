"""Turn a list of test function names into the runner appended to a submission.

The tasks deliberately do not use pytest: the sandbox has no site-packages and
the agent's fix must run against a plain interpreter. The generated runner keeps
the reporting format identical across tasks, so
:meth:`agentdebugger.envs.task_env.TaskEnvironment` has exactly one output
format to parse.
"""

from __future__ import annotations

from collections.abc import Sequence

#: The line every task's runner ends with. The task environment parses it.
SUMMARY_FORMAT = "{passed} passed, {failed} failed"


def build_test_runner(test_names: Sequence[str], setup: str = "") -> str:
    """Return runner source that executes each named test and prints a summary.

    ``setup`` is emitted before the tests run — the concurrency task uses it to
    size thread stacks.
    """
    if not test_names:
        raise ValueError("A task needs at least one test.")

    calls = "\n".join(f"_run_test({name!r}, {name})" for name in test_names)
    return f'''{setup}
_passed = 0
_failures = []


def _run_test(name, fn):
    global _passed
    try:
        fn()
        _passed += 1
    except AssertionError as exc:
        _failures.append("FAILED %s: %s" % (name, exc))
    except Exception as exc:
        _failures.append("ERROR %s: %s: %s" % (name, type(exc).__name__, exc))


{calls}

for _failure in _failures:
    print(_failure)
print("%d passed, %d failed" % (_passed, {len(test_names)} - _passed))
'''
