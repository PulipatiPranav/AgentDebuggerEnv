"""Load the tiered bug dataset that ships with the package.

The JSONL files are package data, so ``load_bugs()`` works from any working
directory and from an installed wheel — no ``data/`` relative paths, no
"run this from the repo root" footgun.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from functools import cache
from importlib import resources

from agentdebugger.config import TIERS
from agentdebugger.dataset.models import Bug

_PACKAGE = "agentdebugger.dataset.bugs"


class DatasetError(Exception):
    """Raised when the bug dataset is missing or malformed."""


@cache
def load_tier(tier: int) -> tuple[Bug, ...]:
    """Load every bug in one difficulty tier."""
    if tier not in TIERS:
        raise DatasetError(f"Unknown tier {tier}. Available: {list(TIERS)}")

    source = resources.files(_PACKAGE).joinpath(f"bugs_tier{tier}.jsonl")
    try:
        text = source.read_text(encoding="utf-8")
    except FileNotFoundError as exc:  # pragma: no cover - packaging failure
        raise DatasetError(f"Bug data for tier {tier} is missing from the package.") from exc

    bugs = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            bugs.append(Bug.from_dict(json.loads(line)))
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise DatasetError(f"bugs_tier{tier}.jsonl line {lineno}: {exc}") from exc
    return tuple(bugs)


def load_bugs(tiers: Iterable[int] = TIERS) -> tuple[Bug, ...]:
    """Load the bugs from the given tiers, in tier order."""
    return tuple(bug for tier in sorted(set(tiers)) for bug in load_tier(tier))


def tier_counts() -> dict[int, int]:
    """How many bugs each tier holds."""
    return {tier: len(load_tier(tier)) for tier in TIERS}


def find_bug(bug_id: str, tiers: Sequence[int] = TIERS) -> Bug:
    """Look up a single bug by id."""
    for bug in load_bugs(tiers):
        if bug.id == bug_id:
            return bug
    raise DatasetError(f"No bug with id {bug_id!r}")
