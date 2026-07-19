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

#: The three recognised split names. ``"all"`` applies no filter — the historical
#: behaviour, and the safe default so nothing that omits ``split`` changes.
SPLITS = ("all", "train", "heldout")


class DatasetError(Exception):
    """Raised when the bug dataset is missing or malformed."""


@cache
def _read_tier(tier: int) -> tuple[Bug, ...]:
    """Load every bug in one difficulty tier, unfiltered."""
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


@cache
def load_split(split: str = "all") -> frozenset[str] | None:
    """Return the set of bug ids in ``split``, or ``None`` for the whole dataset.

    The split is a fixed, committed artifact (``bugs/split.json``): it is authored
    once and never reseeded, because every paired comparison in the experiment plan
    depends on every arm seeing the identical held-out set.
    """
    if split == "all":
        return None
    if split not in SPLITS:
        raise DatasetError(f"Unknown split {split!r}. Available: {list(SPLITS)}")

    source = resources.files(_PACKAGE).joinpath("split.json")
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:  # pragma: no cover - packaging failure
        raise DatasetError("split.json is missing from the package.") from exc

    ids = data.get(split)
    if not isinstance(ids, list):
        raise DatasetError(f"split.json has no {split!r} list.")
    return frozenset(ids)


def load_tier(tier: int, split: str = "all") -> tuple[Bug, ...]:
    """Load the bugs in one difficulty tier, optionally restricted to ``split``."""
    bugs = _read_tier(tier)
    ids = load_split(split)
    if ids is None:
        return bugs
    return tuple(bug for bug in bugs if bug.id in ids)


def load_bugs(tiers: Iterable[int] = TIERS, split: str = "all") -> tuple[Bug, ...]:
    """Load the bugs from the given tiers and split, in tier order."""
    return tuple(bug for tier in sorted(set(tiers)) for bug in load_tier(tier, split))


def tier_counts(split: str = "all") -> dict[int, int]:
    """How many bugs each tier holds in ``split``."""
    return {tier: len(load_tier(tier, split)) for tier in TIERS}


def find_bug(bug_id: str, tiers: Sequence[int] = TIERS) -> Bug:
    """Look up a single bug by id."""
    for bug in load_bugs(tiers):
        if bug.id == bug_id:
            return bug
    raise DatasetError(f"No bug with id {bug_id!r}")
