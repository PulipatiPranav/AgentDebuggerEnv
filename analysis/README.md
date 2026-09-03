# Reproducible paper analysis

This directory contains the analysis code used to regenerate the statistics and W&B diagnostics cited by the paper.

## Final evaluation statistics

```bash
python analysis/bootstrap.py
```

The script resamples the 90 held-out bugs 10,000 times, using the seed-averaged solve indicator for each RL arm. The RNG is `numpy.random.default_rng(204)` and the comparison order is fixed so the percentile endpoints reproduce the paper.

## W&B diagnostics

Export the nine W&B histories as `*_history.csv`, then run:

```bash
python analysis/analyze_wandb.py --csv-dir /path/to/csvs
```

The trainer creates a new GRPOTrainer at curriculum boundaries, so `train/global_step` resets. The script reconstructs cumulative steps using the committed 150/350 boundaries.

`group/degenerate_fraction` is a per-reward-call fraction. In the supplied histories `group/count` is 1, so each logged value is 0 or 1: the single sampled prompt-group in that reward call was either degenerate or non-degenerate. Report aggregates as a descriptive telemetry statistic, not as an independent-sample confidence interval.
