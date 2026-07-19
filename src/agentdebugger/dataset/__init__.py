"""The tiered bug dataset: 180 validated Python bugs across three difficulty tiers,
with a fixed 90/90 train/held-out split (``bugs/split.json``)."""

from agentdebugger.dataset.loader import (
    SPLITS,
    DatasetError,
    find_bug,
    load_bugs,
    load_split,
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
    "SPLITS",
    "Bug",
    "BugLocation",
    "BugReport",
    "DatasetError",
    "TestCase",
    "ValidationReport",
    "find_bug",
    "load_bugs",
    "load_split",
    "load_tier",
    "tier_counts",
    "validate_bug",
    "validate_tiers",
]
