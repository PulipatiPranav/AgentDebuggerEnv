"""Reproduce the paired bootstrap CIs reported in the paper.

Unit of resampling: bug (n=90), not bug x seed.
For an RL arm, each bug's indicator is averaged across the three seeds before
resampling. For B0/B1 the indicator is the single greedy evaluation.

The fixed RNG seed and comparison order are part of reproducibility: changing
them changes percentile endpoints by small amounts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "primary"
DATASET = ROOT / "src" / "agentdebugger" / "dataset" / "bugs"
N_BOOT = 10_000
SEED = 204
SEEDS = (42, 123, 456)


def load(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {row["id"]: int(row["solved"]) for row in data["bugs"]}


def value(records: Mapping[str, int] | Sequence[Mapping[str, int]], bug_id: str) -> float:
    if isinstance(records, Mapping):
        return float(records[bug_id])
    return float(np.mean([record[bug_id] for record in records]))


def paired_difference(
    a: Mapping[str, int] | Sequence[Mapping[str, int]],
    b: Mapping[str, int] | Sequence[Mapping[str, int]],
    bug_ids: Sequence[str],
) -> np.ndarray:
    return np.asarray([value(a, bug) - value(b, bug) for bug in bug_ids], dtype=float)


def bootstrap_ci(diff: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    indices = rng.integers(0, len(diff), size=(N_BOOT, len(diff)))
    boot_means = diff[indices].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def main() -> None:
    b0 = load(RESULTS / "B0.json")
    b1 = load(RESULTS / "B1_700tok.json")
    arms = {
        arm: [load(RESULTS / f"{arm}_s{seed}.json") for seed in SEEDS]
        for arm in ("E1", "E3", "E4")
    }
    bug_ids = list(b0)
    all_rl = arms["E1"] + arms["E3"] + arms["E4"]

    assert len(b0) == 90, f"Expected 90 bugs, got {len(b0)}"
    for arm_list in arms.values():
        for run_dict in arm_list:
            assert set(run_dict) == set(b0), "Mismatch in bug IDs between runs"
    
    # Verify train/heldout disjointness and exact split counts
    split_file = DATASET / "split.json"
    assert split_file.exists(), f"Missing split file: {split_file}"
    split_data = json.loads(split_file.read_text(encoding="utf-8"))
    train_ids = set(split_data["train"])
    heldout_ids = set(split_data["heldout"])
    assert train_ids.isdisjoint(heldout_ids), "Train and heldout bug IDs overlap!"
    assert len(train_ids) == 90, f"Expected 90 train bugs, got {len(train_ids)}"
    assert len(heldout_ids) == 90, f"Expected 90 held-out bugs, got {len(heldout_ids)}"
    assert set(b0) == heldout_ids, "Evaluation bug IDs do not match the held-out split!"

    comparisons = (
        ("E1-E3", arms["E1"], arms["E3"]),
        ("E1-E4", arms["E1"], arms["E4"]),
        ("E3-E4", arms["E3"], arms["E4"]),
        ("E1-B0", arms["E1"], b0),
        ("E3-B0", arms["E3"], b0),
        ("E4-B0", arms["E4"], b0),
        ("ALL-RL-B0", all_rl, b0),
        ("E1-B1", arms["E1"], b1),
        ("B1-B0", b1, b0),
    )

    rng = np.random.default_rng(SEED)
    print("comparison,estimate_pp,ci_low_pp,ci_high_pp")
    for name, a, b in comparisons:
        diff = paired_difference(a, b, bug_ids)
        estimate, lo, hi = bootstrap_ci(diff, rng)
        print(f"{name},{estimate * 100:.1f},{lo * 100:.1f},{hi * 100:.1f}")


if __name__ == "__main__":
    main()
