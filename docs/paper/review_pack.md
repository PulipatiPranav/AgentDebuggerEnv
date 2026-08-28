# Review pack — companion to `paper.md`

Internal document. Everything here is derived from the repository and from statistics
recomputed from `results/*.json`; nothing is estimated or assumed unless marked **[A]**.

---

## 1. Research contribution summary

**Core research question.** Does decomposing an RL reward into hand-crafted components that
pay for debugging *reasoning* (hypothesis quality, fault localization) improve held-out bug-fix
rates over a terminal pass/fail-with-penalties reward, when everything else is held fixed?

**Main contribution.** A controlled, pre-registered negative result: at the 3B scale under GRPO,
it does not. Three reward configurations (R0 seven-component dense, R1 terminal-only, R2 dense
minus the two reasoning components) are statistically indistinguishable on 90 held-out bugs
across three seeds each, with the point estimate favouring the *simplest* reward. The paired
bootstrap bounds any dense-reward advantage at +2.6 points, versus a training effect of +9.7.

**Secondary contributions.**
1. A leakage-controlled testbed: 180 execution-validated single-mutation Python bugs from MBPP,
   a committed immutable 90/90 tier-stratified split, a three-layer execution sandbox, and one
   scoring function shared verbatim by training and evaluation (asserted by test).
2. An evaluation-methodology cautionary result: a hard-coded 300-token generation budget
   manufactured an 85-point apparent format effect (free-form 1.1% → 45.6% after correction).
3. A demonstration that a synthetic dataset's difficulty labels can be wrong in a way that
   invalidates a curriculum premise: tier 1 ("off-by-one") is empirically the *hardest* tier
   for both the 3B base model and Llama-3.3-70B.

**Novelty — stated honestly.** The novelty is *not* the environment, the reward, GRPO, or the
task. It is the **isolation**: the reward decomposition is implemented as component ceilings on
one scoring class, so R0/R1/R2 differ in nothing but a set of numbers — same model, data,
optimizer, prompts, curriculum, step budget, seeds and evaluation path. Published work that
compares dense against sparse rewards for code RL typically varies several factors at once.
This is a clean marginal-value measurement of a design choice practitioners make routinely,
plus a pre-registration that predicted the opposite of what happened.

**Strongest evidence.** Convergence of three independent analyses on the same conclusion:
(i) no per-seed McNemar test survives Holm correction (the only two raw p<0.05 both favour the
*less* shaped arm); (ii) at the all-pairs level R1 and R2 tie exactly (28 vs 28 discordant of
270); (iii) the paired bootstrap CI on R0−R1 is [−8.5, +2.6], i.e. the dense reward's plausible
advantage is smaller than the study's own training effect. All recomputed from committed
per-bug records.

**Biggest weakness.** Statistical power. 90 paired bugs detects ~16-point effects; a real
5-point benefit of dense shaping would be invisible. The claim is therefore "not detected, with
a +2.6-point upper bound under bug-resampling" — never "no effect."

**Biggest reviewer risk.** "Your group size is 2 — you disabled the mechanism you set out to
test." With `num_generations=2`, the group-relative advantage is essentially a sign bit, which
is exactly the regime where a dense reward's ability to *rank* completions within a group is
least useful. This is a legitimate scope limit on the result and is stated in §6 and §8; it
cannot be fixed without re-running the matrix at a larger group size (experiment P0-1).

---

## 2. Evidence audit

| # | Paper claim | Evidence | Verified? | Experiment needed? |
|---|---|---|---|---|
| 1 | Held-out solve rates: B0 37.8%, B1c 45.6%, E1 45.6%, E3 48.5%, E4 48.5% | `results/*.json`, 12 files × 90 per-bug records | ✅ recomputed from raw `solved` flags; matches every published summary | no |
| 2 | Zero train/held-out leakage | `split.json` (90/90, 0 overlap); all 12 eval files score exactly the 90 held-out ids, 0 train ids | ✅ verified programmatically | no |
| 3 | No per-seed McNemar survives Holm correction | exact-binomial McNemar on discordant pairs; 9 tests; min adjusted p = 0.28 | ✅ recomputed | no |
| 4 | E1−E3 bootstrap CI [−8.5, +2.6]; E1−E4 [−8.5, +2.6]; E3−E4 [−5.6, +5.6] | 10,000 paired resamples over bugs, seed-averaged indicator, seed 0 | ✅ recomputed | no |
| 5 | R1 and R2 tie exactly at all-pairs level (28/28, p=1.0) | 270 paired observations | ✅ recomputed | no |
| 6 | Training helps: pooled RL−B0 = +9.7pp, CI [+1.0, +18.5] | paired bootstrap over 9 RL runs vs B0 | ✅ recomputed | no |
| 7 | E1−B0 CI includes zero ([−0.7, +16.3]); E3−B0 and E4−B0 exclude it | same | ✅ recomputed | no |
| 8 | Extraction-failure 0.0% in every reported arm | `overall.extraction_failure_rate` in each file | ✅ verified | no |
| 9 | No run broke a previously-passing test (`newly_broken=0`) | per-bug `tests` records, spot-checked B0/E1_s42/E3_s456 | ✅ verified on 3 files | full sweep would strengthen; low cost |
| 10 | B1 truncation was the sole cause of the 1.1% result | 700-token rerun: 45.6%, 0% extraction failure, 0/90 truncated | ✅ the corrected run's `truncated` flags are all False | ideally: rerun at 300 tokens *with* truncation logging to show the flags fire (P1-3) |
| 11 | Tier labels don't track difficulty (B0: 30/43/40; Llama-70B: 67/57/80) | `results/B0.json`, `gate_llama70b_heldout.json` | ✅ recomputed | no |
| 12 | 20/90 bugs unsolved by all 10 configurations; boundary mutations dominate (13/20) | per-bug intersection across B0 + 9 RL runs, `bug_type` field | ✅ recomputed | no |
| 13 | Seed spread: E3 range 12.2pp (sd 6.5) vs E1 5.6pp (sd 2.9) | per-seed solve rates | ✅ recomputed | claim is descriptive only; a variance test needs ≥8 seeds (P1-1) |
| 14 | Llama-3.3-70B scores 67.8% (difficulty gate) | `gate_llama70b_heldout.json` | ✅ verified | no |
| 15 | Dataset validated by execution (reference passes, mutant fails) | `scripts/build_dataset.py` validates through the sandbox; `agentdebugger validate`; `tests/test_dataset.py` | ✅ code path confirmed; full-dataset test is the `slow`-marked test | run `pytest -m slow` before camera-ready |
| 16 | Training and eval share one scoring function | both call `score_response`; asserted in `tests/test_claims.py` | ✅ test passes | no |
| 17 | Reward sums to exactly 1.0 on perfect solve, floored at −0.5 | `rewards/turn.py`; asserted in `tests/test_rewards.py` | ✅ tests pass | no |
| 18 | Sandbox refuses 15 named escapes; rlimits + process-group deadline | `sandbox/`, `tests/test_sandbox.py` | ✅ 156 tests pass (1 skipped, 3 slow deselected) | no |
| 19 | Prompts length-matched within 15% by word count | `tests/test_prompts.py` invariant | ✅ test passes | no |
| 20 | Training hyperparameters (500 steps, lr 2e-5, temp 0.7, LoRA r8, group 2, 192 tokens) | `TrainingConfig`, `HardwareProfile.for_vram`, `scripts/run_matrix.sh` | ⚠️ **inferred**: the 24GB profile is what a 4090 selects and CONTEXT.md records dual-4090 training, but no run manifest is committed | **P0-2: emit and commit a run manifest** |
| 21 | "~40 s/step structured, ~90 s/step free-form" | `docs/CONTEXT.md` §2.7 narrative | ⚠️ **not independently verifiable** from the repo; used only to justify dropping E2, and reported as such | P2 |
| 22 | Degenerate-group fraction is the mechanism for E3's variance | logging code exists (`_log_batch_diagnostics`) but W&B output is not committed | ❌ **not evidenced** — the paper states this as an untested hypothesis, not a finding | P1-2: export the logged series |

**Claims deliberately NOT made:** clinical/production readiness; transfer to real repositories;
transfer to QuixBugs (adapted but never evaluated); that structure beats free-form (E2 not run);
that the curriculum helps (E5 not run); that GRPO beats PPO; any effect below the MDE.

---

## 3. Reproducibility checklist

**Present in the repository.**
- Source for environment, sandbox, reward, curriculum, trainer, evaluator; MIT licensed.
- Dataset (180 bugs, JSONL) + datacard + generator script + generation seed (0).
- Immutable split file, with a guard in `make_split.py` refusing to regenerate it.
- Exact reward configurations as code (`TurnRewardCalculator.full/terminal/no_reasoning`).
- Prompts for both formats, in one module shared by training and evaluation.
- Hyperparameters as typed defaults (`TrainingConfig`, `HardwareProfile`).
- The full experiment driver (`scripts/run_matrix.sh`) with arm→flag mapping and resume logic.
- All 12 evaluation reports with per-bug completions, actions, test results and reward breakdowns.
- 157-test suite covering the claim-critical properties; CI across Python 3.10–3.13.
- Pinned dependency ranges in `pyproject.toml`; `uv.lock` committed.

**Missing — must be added before camera-ready.**
1. **A run manifest per training run.** Nothing in `results/*.json` records which GPU, which
   `HardwareProfile`, which library versions, or the resolved batch geometry produced each
   checkpoint. The `model` field is just a local path (`./ckpt/E1_s42`). Emit a JSON manifest at
   train time (profile, VRAM, dtype, transformers/trl/peft/torch versions, git SHA, wall-clock).
   *This is the single largest reproducibility gap.*
2. **Training telemetry.** W&B was the only sink for reward curves, per-component means,
   degenerate-group fraction and extraction-failure rate. None is committed, so §6's variance
   discussion rests on end-point solve rates alone. Export the runs to CSV and commit.
3. **The trained adapters.** Only `./ckpt/...` paths are referenced; no adapter is published for
   the 9 matrix runs (only the earlier, pre-split `shashaank0707/AgentDebugger-trained`).
4. **The statistics script.** All numbers in the paper were recomputed ad hoc. Commit the
   analysis script (McNemar + Holm + Wilson + paired bootstrap, seed 0) as
   `scripts/analyze_results.py` so a reader can regenerate every table.
5. **Exact library versions used.** `pyproject.toml` gives ranges; the compat shims in
   `grpo.py` (for transformers 4.54+/4.57+ and trl 0.16.x) imply a narrow working window that is
   not pinned. Record the resolved versions.
6. **The `markdown-preview.pdf` scoping document** referenced by `CONTEXT.md` is not in the repo,
   so a reader cannot check the paper against the scoping authority it cites.
7. **Nothing — but fix a stale note.** `CONTEXT.md` §5 is headed "Code changes applied (not yet
   pushed to GitHub)" and lists five edits the corrected B1 result depends on (format-aware
   `load_generator`, `--max-new-tokens`, free-form completion floor, prompt changes,
   `parse_freeform_output` restore). **This heading is out of date: all five are present in
   `origin/main` at commit `91be258`, the working tree is clean, and local `HEAD` and
   `origin/main` are identical** (verified by `git show origin/main:<file>` for each edit and by
   `git fetch`). No code is unpushed. Edit that heading so it does not mislead a reader — or a
   reviewer — into thinking the B1 correction is unreproducible from the public repository.

---

## 4. Experiments still required

### P0 — necessary before submission

**P0-1. Group-size sweep on the R0 vs R1 contrast.**
- *Objective.* Determine whether the null result is an artifact of `num_generations=2`.
- *Hypothesis.* If dense shaping helps by ranking completions within a group, its benefit should
  grow with group size; at G=8 an R0−R1 gap should appear if the mechanism is real.
- *Design.* E1 and E3 only, G ∈ {2, 8}, 3 seeds each (G=2 arms already exist → 6 new runs).
  Hold total completions/step constant by halving batch×accumulation, so the comparison is
  group size, not throughput.
- *Metric.* Held-out solve rate, paired McNemar R0 vs R1 within each G; interaction reported
  descriptively.
- *Cost.* ~6 × 500 steps at ~40–80 s/step ≈ **40–70 GPU-hours** on a 24GB card. **[A]** step
  time scales roughly linearly in G.
- *Why P0.* It is the first objection any RL reviewer will raise, and the current paper can only
  answer it by acknowledgement.

**P0-2. Run manifests + committed analysis script.**
- Not an experiment: instrument `train()` to write a manifest, re-emit for existing checkpoints
  where recoverable, commit `scripts/analyze_results.py`. **Cost: CPU-only, hours.** Without it,
  reproducibility claims in §8 are overstated.

### P1 — strongly recommended

**P1-1. Seed expansion on E1 and E3 (3 → 8 seeds).**
- *Objective.* Turn §6's variance observation into a testable claim and tighten every CI.
- *Hypothesis.* R1's across-seed variance in final solve rate exceeds R0's (Levene, α=0.05).
- *Metric.* Across-seed sd of held-out solve rate; re-run the paired bootstrap with seed-level
  resampling as well as bug-level.
- *Cost.* 10 new runs ≈ **60 GPU-hours**. Highest value-per-hour item after P0-1: it directly
  attacks the "underpowered" objection and could convert a descriptive remark into a finding.

**P1-2. Export and analyse the degenerate-group series.**
- *Objective.* Test the pre-registered *mechanism*, which is currently unevidenced.
- *Hypothesis.* Per-step degenerate-group fraction is higher under R1 than R0 (Mann–Whitney U
  over steps).
- *Data.* Already logged by `_log_batch_diagnostics` during the completed runs — recover from
  W&B if the runs are still retained; otherwise re-run one seed per arm.
- *Cost.* Zero if W&B history survives; ~14 GPU-hours if two runs must be repeated.

**P1-3. QuixBugs transfer evaluation.**
- *Objective.* Test whether the null result holds off-distribution, on bugs we did not author.
- *Design.* Evaluate the existing 9 checkpoints + B0 on the 27 adapted QuixBugs programs. No
  training. Paired McNemar across arms.
- *Cost.* **~1 GPU-hour total** (270 greedy generations). Cheapest remaining experiment in the
  plan and it adds an external-validity axis the paper currently lacks; n=27 gives very wide
  intervals, so report it as descriptive.

**P1-4. Truncation-flag confirmation of the B1 root cause.**
- Re-run B1 at 300 tokens with the (now present) truncation logging, showing the flags fire on
  the same completions that previously failed extraction. Turns a narrative root-cause claim
  into a demonstrated one. **Cost: ~0.5 GPU-hour.**

### P2 — nice to have

**P2-1. E2 (free-form GRPO) at reduced budget** — 1 seed, 250 steps, to give H1 any evidence at
all; report as illustrative. ~7 GPU-hours.
**P2-2. Flat-vs-curriculum arm (E5)** — with the tier-label finding in §6, the interesting
version is now *re-tiering by measured base-model difficulty* and asking whether an
empirically-ordered curriculum beats the operator-category one. ~14 GPU-hours.
**P2-3. Weak-base-policy replication** — repeat R0 vs R1 on a 0.5B model, where terminal reward
should be near-zero and shaping should matter. This directly probes the paper's own explanation
for why terminal reward sufficed. ~10 GPU-hours.
**P2-4. Full `newly_broken` sweep** across all 12 result files to make the "no regressions"
claim complete rather than spot-checked. CPU-only, minutes.

---

## 5. Simulated peer review

**Scores (workshop scale, 1–10).**

| Dimension | Score | Reasoning |
|---|---|---|
| Novelty | 5 | The contribution is isolation and rigour, not a new method. Defensible for a workshop; would not clear a main track. |
| Technical soundness | 8 | Controls are genuinely tight: one scoring class, immutable split, shared scoring path, verified no leakage. Group size 2 is the soft spot. |
| Experimental rigor | 7 | Pre-registered, paired, corrected, three analysis levels, CIs everywhere, MDE stated. Held back by n=90 and 3 seeds. |
| Clarity | 8 | The claim is unambiguous and the limitations are load-bearing rather than decorative. |
| Reproducibility | 6 | Code/data/split/results all public; run manifests, telemetry and adapters missing (fixable pre-camera-ready). |
| Significance | 6 | A useful practitioner-facing negative result in a narrow regime. |
| Workshop fit | 8 | Strong for a negative-results / AI-for-code / RL-for-reasoning workshop. |

**The five most dangerous objections.**

1. **"Group size 2 disables the mechanism you're testing."** *Why raised:* GRPO's dense-reward
   advantage should come from ranking within a group; with G=2 the advantage is a sign.
   *Fixable?* Only by P0-1. *Action taken:* stated explicitly in §6 as one of three candidate
   explanations and in §8 as a scope limit, with the sweep named as the direct follow-up. This
   is the objection most likely to decide the review.

2. **"You're underpowered — absence of evidence isn't evidence of absence."** *Why raised:*
   n=90, 3 seeds, MDE ~16 points. *Fixable?* Partially. *Action taken:* the MDE is stated
   up-front; the claim is phrased as "not detected"; the bootstrap upper bound (+2.6pp) converts
   the null into a bounded interval, which is the strongest available answer. P1-1 would improve it.

3. **"Your dense reward components are bad heuristics — you disproved *your* reward, not reward
   decomposition."** *Why raised:* `hypothesis_quality` counts words, symbols and digits; it is
   not a reasoning judge. *Fixable?* Not within this study. *Action taken:* the components are
   printed in full in §3.4, the heuristic nature is stated in §3.4 and §6, and the title and
   abstract scope the claim to *reasoning-targeted decomposition of this kind* rather than to
   all dense rewards. R2's exact tie with R1 also shows the null is not carried by the two
   reasoning terms alone — the format/similarity/efficiency terms add nothing either, which
   makes the "bad heuristic" story less able to explain the whole result.

4. **"Synthetic single-mutation MBPP bugs are a toy; this says nothing about debugging."**
   *Why raised:* real bugs are not one AST edit inside one function. *Fixable?* Partially by
   P1-3 (QuixBugs). *Action taken:* stated as the first limitation; the paper claims a result
   about *reward design under GRPO*, not about debugging capability, and the difficulty gate
   (Llama-70B at 67.8%) shows the task is at least non-trivial.

5. **"How do we know the evaluation harness isn't producing this result too?"** *Why raised:*
   the paper itself reports an 85-point artifact from a one-line harness bug. *Fixable?* Yes,
   and answering it well is an asset. *Action taken:* every statistic recomputed from committed
   per-bug records; extraction-failure and truncation rates reported per arm (0.0% and 0/90);
   leakage verified programmatically; the artifact is reported rather than buried. Adding P0-2's
   analysis script closes this fully.

**Revisions made in response to this pass** (relative to the first draft): the title was scoped
from "reward decomposition" to "reasoning-targeted reward decomposition"; §4.1 (deviations from
pre-registration) was added rather than silently dropping H1/H3; the group-size explanation was
promoted into §6 instead of only appearing in limitations; the bootstrap upper bound was moved
into the abstract so the null is bounded at first read; and the tier-label finding was
reframed from a curriculum caveat into a substantive analysis result.
