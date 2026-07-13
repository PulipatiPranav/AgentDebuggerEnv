"""Shared fixtures."""

from __future__ import annotations

import pytest

from agentdebugger.config import SandboxLimits
from agentdebugger.sandbox import SandboxPolicy


@pytest.fixture
def policy() -> SandboxPolicy:
    """The default sandbox policy."""
    return SandboxPolicy()


@pytest.fixture
def fast_policy() -> SandboxPolicy:
    """A policy with a short deadline, so timeout tests do not take ten seconds."""
    return SandboxPolicy(limits=SandboxLimits(wall_clock_seconds=2.0, cpu_seconds=2))
