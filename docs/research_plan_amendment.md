# Protocol Amendments to Preregistration

This document formally records all amendments and deviations from the original preregistration ([research_plan.md](research_plan.md)). All material deviations were established prior to the analysis of the final reward-ablation experimental matrix.

---

## Amendment 1 — Dataset Expansion and Partitioning

* **Preregistered Design (§4.4):** A single 90-bug dataset without an independent held-out evaluation partition (noted as an unresolved precondition in §4.4).
* **Executed Protocol:** Expanded the pool to 180 validated mutation bugs (stratified across Hard, Medium, and Easy tiers) and partitioned them into a fixed 90-bug training set and a strictly disjoint 90-bug held-out evaluation set (`src/agentdebugger/dataset/bugs/split.json`).
* **Rationale:** As mandated by the precondition in §4.4, conducting confirmatory evaluation required an independent held-out set to measure out-of-sample repair capability without train-set leakage.

---

## Amendment 2 — Primary Model Scale (3B instead of 7B)

* **Preregistered Design (§2.1, §4):** Primary experimental campaigns planned for 7B-parameter models (`Qwen2.5-Coder-7B-Instruct`).
* **Executed Protocol:** Primary training executed on `Qwen2.5-Coder-3B-Instruct` across all 9 matrix runs (3 reward configurations $\times$ 3 independent seeds: 42, 123, 456).
* **Rationale:** Compute constraints (approximately 75 total GPU-hours allocated on 24\,GB RTX 4090 hardware) precluded running the full multi-seed $3 \times 3$ matrix at the 7B scale.

---

## Amendment 3 — Omitted Secondary Hypotheses (H1 and H3)

* **Preregistered Design (§1, §3):**
  - **H1 (Structured Format):** Hypothesized that enforcing an explicit Observation $\to$ Hypothesis $\to$ Action schema constraint alone improves repair accuracy over unconstrained free-form generation.
  - **H3 (Curriculum Anti-Collapse):** Hypothesized that a tiered progressive curriculum prevents early policy collapse on Tier 3 bugs compared to uniform flat training.
* **Executed Protocol:** Dropped H1 and H3 from confirmatory hypothesis testing, focusing all compute and replication power exclusively on H2 (Reward Decomposition: R0 vs R1 vs R2).
* **Rationale:** Preserving statistical power and full triplicated replication for the central scientific question regarding reward shaping within the compute budget. Neither H1 nor H3 is presented as confirmatory evidence in the paper.

---

## Amendment 4 — External Transfer Evaluation (QuixBugs)

* **Preregistered Design:** External generalization transfer was not pre-registered as part of the core hypothesis matrix.
* **Executed Protocol:** Evaluated the 9 trained policies on 27 single-function programs adapted from QuixBugs (`data/quixbugs/`).
* **Status:** Reported strictly as an exploratory, descriptive evaluation. Because an untrained zero-shot base-model baseline (B0) was not evaluated on QuixBugs, these numbers do not establish an RL-induced generalization gain.

---

## Amendment 5 — Difficulty Gate Calibration Baseline

* **Preregistered Design (§2.2):** Difficulty gating calibration planned using external API calls (`gpt-4o-mini`).
* **Executed Protocol:** Difficulty gating calibrated locally with `meta-llama/Llama-3.1-8B-Instruct`.
* **Rationale:** Fully offline, reproducible calibration without dependencies on non-deterministic proprietary web APIs.
