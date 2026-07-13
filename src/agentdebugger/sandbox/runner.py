"""Run untrusted code in a resource-limited, short-lived subprocess.

Three layers of defence, in the order they take effect:

1. **Static analysis** (:mod:`agentdebugger.sandbox.policy`) runs in *this*
   process and rejects the submission before a child is ever spawned.
2. **Kernel limits** (``setrlimit``) are applied in the child before it executes
   anything: address space, CPU time, and a zero file-write budget. These hold
   even if the child ignores signals.
3. **A wall-clock deadline** in the parent, which kills the child's entire
   process group — so a fix that spawns threads or grandchildren still dies.

The child is started with a scratch working directory, a scrubbed environment
and ``-I`` (isolated mode), so it cannot import anything from the caller's tree.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

from agentdebugger.config import SandboxLimits
from agentdebugger.sandbox.policy import SandboxPolicy, Violation, analyze

try:  # pragma: no cover - Windows has no `resource` module
    import resource

    HAS_RLIMITS = True
except ImportError:  # pragma: no cover
    HAS_RLIMITS = False

#: Placeholder substituted for the scratch directory, so identical code always
#: produces identical output. Rewards are computed from this text; it must not
#: vary run to run.
SANDBOX_PATH_PLACEHOLDER = "<sandbox>"


@dataclass(frozen=True)
class ExecutionResult:
    """Outcome of one sandboxed execution."""

    output: str
    timed_out: bool = False
    duration_ms: int = 0
    exit_code: int | None = None
    violations: tuple[Violation, ...] = field(default=())

    @property
    def blocked(self) -> bool:
        """True when the policy rejected the code and nothing was executed."""
        return bool(self.violations)


def _rejection_output(violations: tuple[Violation, ...]) -> str:
    lines = "\n".join(f"  {v}" for v in violations)
    return f"BLOCKED: code violates the sandbox policy and was not executed.\n{lines}"


#: Installed in the child before user code runs. Blocks dynamic imports that the
#: static pass cannot see (``importlib``, ``__import__`` reached via a builtin
#: reference that survived analysis). It only blocks imports issued *by the user
#: program*: stdlib modules import each other freely, so allowed modules that
#: internally depend on a blocked one (``random`` needs ``os``) keep working.
_BOOTSTRAP = '''\
import builtins
import runpy
import sys
import traceback

_PROGRAM = {program!r}
_BLOCKED = frozenset({blocked!r})
_SWITCH_INTERVAL = {switch_interval!r}
_real_import = builtins.__import__

if _SWITCH_INTERVAL is not None:
    sys.setswitchinterval(_SWITCH_INTERVAL)


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    importer = (globals or {{}}).get("__name__")
    if importer == "__main__" and name.split(".")[0] in _BLOCKED:
        raise ImportError(
            f"BLOCKED IMPORT: {{name!r}} is not available inside the sandbox."
        )
    return _real_import(name, globals, locals, fromlist, level)


builtins.__import__ = _guarded_import
sys.argv = ["program.py"]

try:
    runpy.run_path(_PROGRAM, run_name="__main__")
except SystemExit:
    raise
except BaseException as exc:
    # Drop this bootstrap and runpy's own frames so the traceback the agent
    # reads starts at the first line of its own program.
    tb = exc.__traceback__
    while tb is not None and tb.tb_frame.f_code.co_filename != _PROGRAM:
        tb = tb.tb_next
    traceback.print_exception(type(exc), exc, tb)
    sys.exit(1)
'''


def _limit_child(limits: SandboxLimits) -> None:  # pragma: no cover - runs post-fork
    """Apply kernel resource limits. Executed in the child between fork and exec."""
    os.setsid()
    memory_bytes = limits.memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds + 1))
    resource.setrlimit(
        resource.RLIMIT_FSIZE,
        (limits.max_file_write_bytes, limits.max_file_write_bytes),
    )
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def execute(
    code: str,
    test_code: str = "",
    policy: SandboxPolicy | None = None,
) -> ExecutionResult:
    """Execute ``code`` followed by ``test_code`` under ``policy``.

    Both fragments are concatenated into a single program, so the tests see the
    definitions from ``code`` exactly as a real module would. Nothing is executed
    if either fragment violates the policy.
    """
    policy = policy or SandboxPolicy()
    program = f"{code}\n\n{test_code}\n" if test_code else f"{code}\n"

    violations = tuple(analyze(program, policy))
    if violations:
        return ExecutionResult(output=_rejection_output(violations), violations=violations)

    workdir = tempfile.mkdtemp(prefix="agentdebugger-sandbox-")
    try:
        return _run(program, workdir, policy)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _run(program: str, workdir: str, policy: SandboxPolicy) -> ExecutionResult:
    limits = policy.limits
    program_path = os.path.join(workdir, "program.py")
    bootstrap_path = os.path.join(workdir, "bootstrap.py")
    with open(program_path, "w", encoding="utf-8") as handle:
        handle.write(program)
    with open(bootstrap_path, "w", encoding="utf-8") as handle:
        handle.write(
            _BOOTSTRAP.format(
                blocked=sorted(policy.blocked_imports),
                program=program_path,
                switch_interval=limits.switch_interval,
            )
        )

    started = time.monotonic()
    process = subprocess.Popen(
        [sys.executable, "-I", "-B", bootstrap_path],
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        env={"PATH": "/usr/bin:/bin", "HOME": workdir, "LC_ALL": "C.UTF-8"},
        preexec_fn=(lambda: _limit_child(limits)) if HAS_RLIMITS else None,
        start_new_session=not HAS_RLIMITS,  # _limit_child already calls setsid()
    )

    timed_out = False
    try:
        output, _ = process.communicate(timeout=limits.wall_clock_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_group(process)
        output, _ = process.communicate()
        output = (output or "") + (
            f"\nTIMEOUT: execution exceeded the {limits.wall_clock_seconds:g}s limit."
        )

    duration_ms = int((time.monotonic() - started) * 1000)
    return ExecutionResult(
        output=_clean(output, workdir, limits.max_output_chars),
        timed_out=timed_out,
        duration_ms=duration_ms,
        exit_code=process.returncode,
    )


def _kill_process_group(process: subprocess.Popen) -> None:
    """SIGKILL the child and everything it spawned."""
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):  # pragma: no cover - already gone
        process.kill()


def _clean(output: str, workdir: str, max_chars: int) -> str:
    """Make output stable across runs and bounded in size."""
    output = (output or "").replace(workdir, SANDBOX_PATH_PLACEHOLDER)
    if len(output) > max_chars:
        half = max_chars // 2
        output = f"{output[:half]}\n...[output truncated]...\n{output[-half:]}"
    return output.strip()
