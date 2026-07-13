"""Static policy for sandboxed code: what may be imported, called and reached.

The policy is enforced *before* execution by parsing the candidate source and
walking its AST. Doing the analysis in the parent process (rather than inside
the sandbox itself) keeps the check out of reach of the code being checked, and
lets the caller reject a submission without ever running it.

Threat model
------------
The sandbox defends against code that is *hostile by accident* and against the
naive escape attempts an LLM actually produces (``import os``, ``open(...)``,
``eval``, ``().__class__.__subclasses__()``). Combined with the kernel-enforced
limits in :mod:`agentdebugger.sandbox.runner` it is safe to run model output on
a workstation or CI box. It is **not** a substitute for a container or VM when
executing deliberately adversarial code from an untrusted third party.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace

from agentdebugger.config import DEFAULT_LIMITS, SandboxLimits

#: Top-level modules the sandbox refuses to import. Anything that reaches the
#: filesystem, the network, another process, the interpreter internals, or the
#: import system itself.
BLOCKED_IMPORTS: frozenset[str] = frozenset(
    {
        # Process, filesystem and OS surface
        "os", "posix", "nt", "sys", "io", "_io", "subprocess", "_posixsubprocess",
        "shutil", "pathlib", "glob", "tempfile", "fileinput", "fcntl", "termios",
        "pty", "signal", "resource", "mmap", "getpass", "platform", "webbrowser",
        # Network
        "socket", "_socket", "ssl", "http", "urllib", "ftplib", "smtplib",
        "telnetlib", "requests", "httpx", "aiohttp",
        # Concurrency (threading is unblocked per-task; see SandboxPolicy.allowing)
        "threading", "_thread", "multiprocessing", "concurrent", "asyncio",
        # Serialisation that executes code, and on-disk stores
        "pickle", "cPickle", "shelve", "marshal", "dbm", "sqlite3",
        # Interpreter internals / reflection / dynamic import
        "builtins", "__builtin__", "importlib", "imp", "runpy", "code", "codeop",
        "inspect", "ctypes", "cffi", "gc", "types", "atexit", "pdb", "bdb",
    }
)

#: Builtins the sandbox refuses to reference by name. These are the primitives
#: that turn "runs arbitrary arithmetic" into "runs arbitrary code".
BLOCKED_BUILTINS: frozenset[str] = frozenset(
    {
        "eval", "exec", "compile", "open", "input", "breakpoint", "help",
        "getattr", "setattr", "delattr", "globals", "locals", "vars",
        "memoryview", "__import__", "__builtins__", "__loader__", "__spec__",
    }
)

#: Dunder attributes that are reachable escape hatches (``().__class__``,
#: ``fn.__globals__``, ``obj.__reduce__``...). Dunders *not* on this list stay
#: legal, so ordinary object-oriented fixes — ``super().__init__()``,
#: ``def __repr__`` — are unaffected.
BLOCKED_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "__class__", "__base__", "__bases__", "__mro__", "__subclasses__",
        "__globals__", "__code__", "__closure__", "__func__", "__self__",
        "__dict__", "__builtins__", "__getattribute__", "__reduce__",
        "__reduce_ex__", "__import__", "__loader__", "__spec__", "__module__",
    }
)


@dataclass(frozen=True)
class SandboxPolicy:
    """An immutable set of sandbox rules plus the resource limits to run under."""

    blocked_imports: frozenset[str] = BLOCKED_IMPORTS
    blocked_builtins: frozenset[str] = BLOCKED_BUILTINS
    blocked_attributes: frozenset[str] = BLOCKED_ATTRIBUTES
    limits: SandboxLimits = DEFAULT_LIMITS

    def allowing(self, *modules: str) -> SandboxPolicy:
        """Return a copy of this policy with ``modules`` removed from the denylist.

        The concurrency task needs ``threading`` to be importable; nothing else
        in the repository relaxes the policy.
        """
        return replace(self, blocked_imports=self.blocked_imports - set(modules))


class PolicyViolation(Exception):
    """Raised when source code breaks the sandbox policy before execution."""


@dataclass(frozen=True)
class Violation:
    """A single policy breach, located in the analysed source."""

    kind: str  # "import" | "builtin" | "attribute"
    name: str
    line: int

    def __str__(self) -> str:
        reasons = {
            "import": f"import of blocked module {self.name!r}",
            "builtin": f"use of blocked builtin {self.name!r}",
            "attribute": f"access to blocked attribute {self.name!r}",
        }
        return f"line {self.line}: {reasons[self.kind]}"


def analyze(source: str, policy: SandboxPolicy) -> list[Violation]:
    """Return every policy violation in ``source``.

    Source that does not parse yields no violations: a ``SyntaxError`` is the
    agent's problem, not a security event, and the runner surfaces it as
    ordinary execution output.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    violations: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in policy.blocked_imports:
                    violations.append(Violation("import", root, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            # `from . import x` has module=None; relative imports cannot escape
            # a single-file sandbox, so only absolute imports are checked.
            root = (node.module or "").split(".")[0]
            if root in policy.blocked_imports:
                violations.append(Violation("import", root, node.lineno))
        elif isinstance(node, ast.Name) and node.id in policy.blocked_builtins:
            violations.append(Violation("builtin", node.id, node.lineno))
        elif isinstance(node, ast.Attribute) and node.attr in policy.blocked_attributes:
            violations.append(Violation("attribute", node.attr, node.lineno))

    violations.sort(key=lambda v: (v.line, v.kind, v.name))
    return violations
