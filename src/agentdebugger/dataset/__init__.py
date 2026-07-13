"""The tiered bug dataset: 90 hand-checked Python bugs across three difficulty tiers."""

from agentdebugger.dataset.loader import (
    DatasetError,
    find_bug,
    load_bugs,
    load_tier,
    tier_counts,
)
from agentdebugger.dataset.models import Bug, BugLocation, TestCase
from agentdebugger.dataset.validate import (
    BugReport,
    ValidationReport,
    validate_bug,
    validate_tiers,
)

__all__ = [
    "Bug",
    "BugLocation",
    "BugReport",
    "DatasetError",
    "TestCase",
    "ValidationReport",
    "find_bug",
    "load_bugs",
    "load_tier",
    "tier_counts",
    "validate_bug",
    "validate_tiers",
]
