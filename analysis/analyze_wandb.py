"""Summarise the exported W&B histories used by the paper audit.

The trainer creates a fresh GRPOTrainer at each curriculum boundary, so
``train/global_step`` resets at 150 and 350. This script reconstructs a
cumulative step axis [0, 500) using those committed boundaries.
"""
from pathlib import Path

import pandas as pd


def add_cumulative_step(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("_step").reset_index(drop=True).copy()
    local = df["train/global_step"].ffill().astype(int)
    stage = local.diff().lt(0).cumsum().astype(int)
    offsets = {0: 0, 1: 150, 2: 350}
    if stage.max() > 2:
        raise ValueError("Unexpected number of curriculum resets")
    df["curriculum_stage"] = stage
    df["cumulative_step"] = [offsets[int(s)] + int(v) for s, v in zip(stage, local, strict=True)]
    return df


def summarise(path: Path) -> dict[str, object]:
    df = add_cumulative_step(pd.read_csv(path))
    out: dict[str, object] = {
        "run": path.stem.removesuffix("_history"),
        "logged_reward_calls": int(df["group/degenerate_fraction"].notna().sum()),
        "cumulative_step_max": int(df["cumulative_step"].max()),
    }
    for metric in (
        "group/degenerate_fraction",
        "reward/mean",
        "reward/solve_rate",
        "reward/extraction_failure_rate",
    ):
        s = df[metric].dropna()
        out[f"{metric}_mean"] = float(s.mean())
        out[f"{metric}_final"] = float(s.iloc[-1]) if len(s) else None
    return out


def main(csv_dir: str | None = None) -> None:
    directory = Path(csv_dir) if csv_dir else Path(".")
    files = sorted(directory.glob("*_history.csv"))
    if not files:
        raise SystemExit("No *_history.csv files found. Pass the directory containing the W&B exports.")
    rows = [summarise(path) for path in files]
    result = pd.DataFrame(rows).sort_values("run")
    print(result.to_string(index=False))

    print("\nArm-level degenerate-group summary")
    result["arm"] = result["run"].str.split("_").str[0]
    grouped = result.groupby("arm")["group/degenerate_fraction_mean"]
    print(grouped.agg(["mean", "std"]).to_string())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarise W&B run histories for GRPO reward arms.")
    parser.add_argument(
        "--csv-dir",
        default=".",
        help="Directory containing exported W&B history CSVs (e.g., E1_s42_history.csv, ..., E4_s456_history.csv)",
    )
    args = parser.parse_args()
    main(args.csv_dir)
