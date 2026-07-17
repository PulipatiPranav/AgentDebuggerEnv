"""Create the fixed train/held-out split for the v2 bug dataset.

The split is the single most important correctness artifact in the project
(research_plan.md §0 and §4.4): without it, training and evaluation share bugs
and every reported number measures memorisation, not debugging.

It is committed as an explicit list of bug ids — **not** a random seed that could
be silently rerun — stratified by tier, and once written it must never be
regenerated, or the pairing every statistical test depends on is destroyed.

Usage:
    python scripts/make_split.py                 # 50/50 per tier, seed 0
    python scripts/make_split.py --heldout-frac 0.5 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="data/v2", help="dataset directory")
    parser.add_argument("--heldout-frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="overwrite an existing split")
    args = parser.parse_args(argv)

    directory = Path(args.dir)
    out = directory / "split.json"
    if out.exists() and not args.force:
        raise SystemExit(
            f"{out} already exists. The split must be fixed once and never reseeded; "
            "pass --force only if you are certain no results depend on the current split."
        )

    rng = random.Random(args.seed)
    train: list[str] = []
    heldout: list[str] = []
    per_tier: dict[int, dict[str, int]] = {}

    for tier in (1, 2, 3):
        path = directory / f"bugs_tier{tier}.jsonl"
        ids = [
            json.loads(line)["id"]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        ids.sort()  # deterministic order before shuffling
        rng.shuffle(ids)
        n_heldout = round(len(ids) * args.heldout_frac)
        tier_heldout, tier_train = ids[:n_heldout], ids[n_heldout:]
        heldout.extend(tier_heldout)
        train.extend(tier_train)
        per_tier[tier] = {"train": len(tier_train), "heldout": len(tier_heldout)}

    payload = {
        "seed": args.seed,
        "heldout_frac": args.heldout_frac,
        "per_tier": per_tier,
        "train": sorted(train),
        "heldout": sorted(heldout),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote {out}")
    for tier, counts in per_tier.items():
        print(f"  tier {tier}: {counts['train']} train / {counts['heldout']} heldout")
    print(f"  total: {len(train)} train / {len(heldout)} heldout")
    print("\nThis split is now FIXED. Do not rerun without --force.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
