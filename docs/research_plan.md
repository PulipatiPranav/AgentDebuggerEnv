# Research plan

This document states what AgentDebuggerEnv claims, in a form that can be shown to
be false, and specifies the smallest experiment that could show it. It is written
**before** any of these experiments are run, and is meant to be cited as a
pre-registration: if the results contradict it, the results win.

Nothing in this document reports a result. Every number below is either a
property of the shipped code (linked to its source) or a design parameter derived
from a stated assumption. Assumptions are marked **[A]**.

> ### Pre-registration Status & Execution Addendum (August 2026)
> 
> **Status:** Pre-registration authored July 2026; experimental campaign executed August 2026.
> 
> **1. Resolution of Preconditions (§4.4):** As mandated in §4.4, the initial 90-bug prototype lacked a train/held-out split. Prior to executing the experimental matrix, the dataset was expanded from 90 to 180 validated mutation bugs (`data/v2/`), creating a clean 90-bug training set (`data/v2/train.jsonl`) and a strictly disjoint 90-bug held-out evaluation set (`data/v2/held_out.jsonl`), stratified by tier. The historical note in §0 reflects the v0.1.0 codebase prior to this fix.
> 
> **2. Executed Experimental Matrix:** Due to compute constraints (75 GPU-hours on 24GB RTX 4090 hardware), the study executed the 3B model scale across the three core reward arms:
> - **E1 (R0 Full Dense):** 3 seeds (42, 123, 456)
> - **E3 (R1 Graded Outcome):** 3 seeds (42, 123, 456)
> - **E4 (R2 Dense No Reasoning):** 3 seeds (42, 123, 456)
> 
> All 9 runs were trained for 500 steps and evaluated on the 90 held-out bugs. Per-run JSON evaluation artifacts are committed at [`results/primary/`](../results/primary/). The historical pre-split prototype run was archived in [`results/historical/`](../results/historical/).
> 
> **3. Preregistration Deviations:**
> - *Scale:* Primary training executed at 3B scale rather than 7B due to hardware constraints.
> - *Hypotheses:* H1 (format) and H3 (curriculum collapse) were omitted from confirmatory testing to maintain full statistical power and replication on the central question (H2: reward decomposition).
> - *Calibration Baseline:* Local Llama-3.1-8B-Instruct gating was used instead of GPT-4o-mini API calls.
> 
> **4. Outcome Summary:** On the 90 held-out bugs, R0 achieved 45.6\% $\pm$ 2.9\%, R1 achieved 48.5\% $\pm$ 6.5\%, and R2 achieved 48.5\% $\pm$ 3.9\% solve rate. Paired bootstrap analysis (10,000 resamples, seed 204 in `analysis/bootstrap.py`) yielded a 95\% CI of [$-$8.5, $+$2.6] pp for R0 $-$ R1. The dense reward significantly reduced degenerate GRPO groups (16.2\% vs 61.9\%) but did not produce a detectable improvement in held-out solve rate. See the formal deviations in [docs/research_plan_amendment.md](research_plan_amendment.md) and the accompanying paper (`docs/paper_preprint.pdf`) for the full report.

---

## 0. What the environment originally measured, and the split precondition

Facts, from the code:

- The original dataset was **90 bugs**: 40 tier-1, 30 tier-2, 20 tier-3
  ([`dataset/bugs/`](../src/agentdebugger/dataset/bugs/)).
- The reward decomposes into six earned components and one penalty term, summing
  to exactly `1.0` on a perfect first-turn solve and floored at `-0.5`
  ([`rewards/turn.py`](../src/agentdebugger/rewards/turn.py)): format `0.10`,
  hypothesis `0.20`, localization `0.15`, fix `0.35`, semantic `0.10`, efficiency
  `0.10`, penalties `−0.55`.
- A bug counts as **solved** when every one of its test cases passes — the point
  at which `fix_quality` reaches its `0.35` ceiling
  ([`SOLVED_THRESHOLD`](../src/agentdebugger/rewards/turn.py)).
- The curriculum unlocks tier 2 at step 150 and tier 3 at step 350
  ([`config.py`](../src/agentdebugger/config.py)).
- Training and evaluation score a completion with the **same** `score_response`
  function, so a training curve and an eval number are commensurable
  (asserted in `tests/test_claims.py`).

**The split precondition (resolved in v0.2.0).** In the initial v0.1.0 release,
[`load_bugs()`](../src/agentdebugger/dataset/loader.py) returned every bug in the
requested tiers without a split. This design document identified that
**creating a train/held-out split was an absolute precondition** before running
any hypothesis test (§4.4). As documented in the addendum above, this was resolved
prior to running the matrix by expanding to 180 bugs in `data/v2/` and partitioning
into 90 train and 90 held-out bugs. The legacy un-split run was archived to
`results/historical/` and superseded by `results/primary/`.

---

## 1. Hypotheses

Throughout: the **unit of analysis is one bug**; the **primary dependent
variable** is the *held-out solve rate* — the fraction of held-out bugs for which
the policy's fix passes every test case. All comparisons are **paired on the bug**
(every arm sees the identical held-out set) and pooled across seeds, so the tests
below are McNemar / paired-bootstrap rather than two-sample tests. Significance is
α = 0.05, two-sided, Holm-corrected across the three primary comparisons (H1, H2,
H3).

No effect-size threshold is asserted anywhere. An acceptance criterion of the
form "improves by at least *k* points" would be an invented number; instead each
hypothesis is accepted when the **sign of the effect is as predicted and its 95%
confidence interval excludes zero**. What the design *does* have to commit to is
the smallest effect it could detect at all — see §4.5.

### H1 — Structured reasoning

> **Formal.** Let π_struct be a policy trained with GRPO in
> `CurriculumEnvironment`, required to emit the OBSERVATION / HYPOTHESIS /
> CONFIDENCE / ACTION / DETAIL format, and π_free an otherwise identical policy
> (same base weights, same bugs, same curriculum, same optimiser, same
> hyperparameters, same step budget, same reward) that emits a free-form response
> from which a fix is extracted. Then, on held-out bugs, π_struct attains a
> strictly higher solve rate than π_free.

- **H1₀ (null).** Solve rate(π_struct) − solve rate(π_free) ≤ 0.
- **Independent variable.** Response format required by the prompt and parser:
  structured vs. free-form. *One* variable — see the confound note below.
- **Dependent variables.** Primary: held-out solve rate. Secondary: mean total
  reward; regression rate (fraction of submissions with `newly_broken > 0`);
  localization accuracy (fraction naming the true buggy function); solve rate
  broken out by tier.
- **Metric.** Paired McNemar test on per-bug solve/not-solve across the two arms,
  pooled over 3 seeds; effect reported as a difference in proportions with a 95%
  paired-bootstrap CI over bugs.
- **Acceptance.** Reject H1₀ iff the paired difference is positive and its
  Holm-corrected 95% CI excludes 0.

**The confound, and how it is removed.** The naive comparison — the full system
vs. an unstructured baseline — changes *two* things at once, because four of the
seven reward components (format, hypothesis, localization, and the confidence
calibration inside hypothesis quality) are **undefined** without the structure.
So "structured" and "densely shaped" are entangled by construction. The design
breaks them apart by giving both arms the **same terminal-only reward** (R1,
§3): E3 (structured, R1) vs. E2 (free-form, R1) differ *only* in the format
requirement, and that is the primary test of H1. E1 vs. E2 — the full system vs.
the unstructured baseline — is reported as the *joint* effect of the whole
intervention, and is explicitly labelled as such rather than as evidence for
structure alone.

**Threats to validity.**
- *Parser asymmetry.* If the free-form arm's fix extractor is weaker than the
  structured parser, H1 is confirmed by a bad regex rather than by reasoning.
  Mitigation: the free-form extractor takes the last fenced code block, falling
  back to the whole response; its extraction-failure rate is reported as a
  diagnostic, and if it exceeds the structured arm's format-failure rate the
  comparison is void.
- *Prompt-length confound.* The structured prompt is longer and demonstrates a
  format. Mitigation: match token counts within ~10% **[A]** by giving the
  free-form arm an equally long instruction (a worked example, no schema).
- *Format-as-scaffold, not reasoning.* The model may benefit from any rigid
  output schema, not from hypothesising. This is a live alternative explanation
  the design cannot fully exclude at this cost; the honest statement is that H1
  tests "this structure helps", not "hypothesising per se helps". A future arm
  with a length-matched but semantically empty schema would separate them.
- *Ceiling on tier 1.* If tier-1 solve rates saturate, the pooled effect is
  carried entirely by tiers 2–3. Mitigation: report per-tier effects alongside
  the pooled one, and pre-register the pooled rate as primary.

### H2 — Reward decomposition

> **Formal.** Let π_dense be a policy trained with the full seven-component
> reward (R0) and π_mono an otherwise identical policy trained with a monolithic
> terminal reward (R1) that pays only for test outcome and penalties. Then, at
> equal step budget, π_dense attains a strictly higher held-out solve rate than
> π_mono; and the advantage is mediated by a lower rate of **degenerate GRPO
> groups** — sampled groups in which every completion receives an identical
> reward, so the group-relative advantage is exactly zero and the step
> contributes no gradient.

- **H2₀ (null).** Solve rate(π_dense) − solve rate(π_mono) ≤ 0.
- **Independent variable.** Reward configuration: R0 (full) vs. R1 (terminal
  only). Structure, curriculum, budget and hyperparameters held fixed.
- **Dependent variables.** Primary: held-out solve rate. Mediator (this is the
  *mechanism* the hypothesis proposes, and it is measurable): the fraction of
  training steps whose sampled group has zero reward variance. Secondary: sample
  efficiency (steps to first reach the base model's solve rate); mean reward;
  per-component reward trajectories.
- **Metric.** Same paired test as H1 for the primary. For the mediator, the
  per-step degenerate-group fraction, compared between arms over the run
  (Mann–Whitney U over steps).
- **Acceptance.** Reject H2₀ iff the paired difference in held-out solve rate is
  positive with a Holm-corrected 95% CI excluding 0. The mediator is
  **corroborating, not gating**: if solve rate improves while the degenerate-group
  fraction is *unchanged*, the effect is real but the proposed mechanism is
  wrong, and that must be reported as such.
- **A third arm, R2**, keeps the dense format/efficiency/semantic components but
  zeroes exactly the two *reasoning* components (hypothesis quality, localization).
  R0 vs. R2 asks whether paying for the reasoning specifically matters, or whether
  any dense shaping would do. This distinguishes the project's actual claim from
  the much weaker "dense rewards help RL", which is not news.

**Threats to validity.**
- *Reward hacking.* A dense reward can be climbed without fixing anything (write
  confident prose, name the right function, never pass a test). Mitigation: the
  primary DV is solve rate, which is immune to this by construction; additionally
  report the gap between mean total reward and solve rate — a widening gap *is*
  the reward-hacking signature.
- *Scale confound.* R0 and R1 have different reward magnitudes and variances.
  GRPO normalises advantages within a group, which largely absorbs a scale
  difference, but not a *variance* difference — and the variance difference is
  precisely the mechanism H2 proposes, so it must not be "corrected" away.
  Mitigation: state this explicitly; do not rescale R1; report raw reward
  statistics for both.
- *Component collinearity.* Hypothesis quality and localization are both text
  heuristics over the same response and may be highly correlated, so zeroing both
  in R2 may not isolate what it appears to. Mitigation: report the empirical
  correlation between the two components on the R0 run before interpreting R2.
- *Threshold artefacts.* `fix_quality` is a step function of pass rate
  (`1.0 → 0.35`, `≥0.75 → 0.20`, …). Fixes that sit near a boundary can move
  reward without moving solve rate. Mitigation: report the pass-rate distribution,
  not just the solve rate.

### H3 — Curriculum

> **Formal.** Let π_curr be a policy trained with the tiered schedule (tier 1 to
> step 150, tiers 1–2 to step 350, then all tiers) and π_flat an otherwise
> identical policy trained by sampling uniformly from all three tiers from step 0.
> Then π_curr exhibits (a) lower instability, and (b) a lower incidence of policy
> collapse, than π_flat — and does so without a lower final held-out solve rate.

Instability and collapse must be defined operationally, and *without an invented
threshold*. Both are therefore anchored to the **untrained base policy**, which
is a principled reference point rather than a chosen number:

- **Instability** = the standard deviation of mean training reward across steps,
  computed over a sliding window, after the first 50 steps **[A]** (excluding the
  initial transient); plus the standard deviation of the final held-out solve rate
  *across seeds*, which is what a practitioner actually feels as instability.
- **Collapse** = an absorbing degenerate state, declared when *either*: (i) the
  held-out solve rate, evaluated periodically, falls below the **untrained base
  model's** solve rate and does not recover within the next two evaluation points
  **[A]**; or (ii) output diversity — distinct-3-gram ratio across the sampled
  group — falls below that of the untrained base model and stays there. Both
  reference the base model, not a magic constant.

- **H3₀ (null).** Instability(π_curr) ≥ instability(π_flat), and collapse
  incidence(π_curr) ≥ collapse incidence(π_flat).
- **Independent variable.** Sampling schedule: staged vs. flat. Everything else,
  including the total number of gradient steps and the reward, held fixed.
- **Dependent variables.** Primary: across-seed standard deviation of final
  held-out solve rate; number of seeds (of *n*) that collapse. Secondary:
  within-run reward standard deviation; policy entropy; KL from the reference
  policy; gradient norm; the transient dip in reward at each curriculum boundary.
- **Metric.** Across-seed variance compared with Levene's test (robust to
  non-normality at small *n*); collapse incidence as a count out of *n* seeds,
  with a Fisher exact test.
- **Acceptance.** Reject H3₀ iff π_curr shows a lower across-seed variance in
  final solve rate (Levene, α = 0.05) **and** collapses in no more seeds than
  π_flat, **and** its mean final solve rate is not lower (i.e. stability is not
  bought with performance).

**Threats to validity.**
- *Underpowered by design.* Variance and collapse-incidence are being estimated
  from a handful of seeds. With 3 seeds per arm, only a near-total difference
  (e.g. 3/3 collapses vs. 0/3, Fisher p = 0.10 — *not significant at α = 0.05*)
  is detectable. **This is the weakest hypothesis in the plan, and the design
  cannot fix it cheaply**: a properly powered collapse-incidence test needs on the
  order of 10+ seeds per arm. The honest options are to (a) run H3 at the 3B scale
  where seeds are cheap enough to afford 8–10 of them, and report the 7B pair as
  an illustration rather than a test, or (b) demote H3 to a *descriptive* finding.
  The plan takes option (a); see §4.3.
- *Curriculum ≠ difficulty.* The tiers are hand-labelled. If the labels do not
  track true difficulty, the "curriculum" is just a different sampling order.
  Mitigation: report the base model's per-tier solve rate as an *empirical*
  difficulty check — tier 1 > tier 2 > tier 3 should hold before any curriculum
  claim is made.
- *Exposure confound.* π_curr sees tier-3 bugs for only 150 of 500 steps; π_flat
  sees them throughout. If π_curr wins on tier 3 anyway, the effect is strong; if
  it loses, that may be exposure, not instability. Mitigation: report per-tier
  final performance and total tier-3 exposure count for both arms.
- *Collapse may simply not occur.* If neither arm collapses in any seed, H3 is
  untestable at this scale and must be reported as such — not silently dropped.

---

## 2. Experiment table

Model column: `Qwen2.5-Coder-*-Instruct` throughout, so the base family is held
fixed and only scale varies. "Curriculum: staged" is the shipped schedule
(150/350); "flat" samples all tiers from step 0.

| ID | Model | Scale | Reward config | Curriculum | Type | Purpose |
| --- | --- | --- | --- | --- | --- | --- |
| **B0** | Qwen2.5-Coder-Instruct | 7B | — | — | Baseline (no training) | Zero-shot, structured prompt. The control every RL arm is measured against; also the anchor for the collapse definition in H3. |
| **B1** | Qwen2.5-Coder-Instruct | 7B | — | — | Baseline (no training) | Zero-shot, free-form prompt. Shows how much of any structured-vs-free gap exists *before* RL. |
| **B2** | GPT-4o-mini (API) | — | — | — | Baseline (no training) | External calibration: is the environment merely easy? Run in both prompt formats. See §5. |
| **B3** | OracleAgent | — | — | — | Ceiling (CPU) | The reference fix. Already in the repo; bounds the achievable score. |
| **E1** | Qwen2.5-Coder-Instruct | 7B | **R0** full | staged | RL (GRPO+LoRA) | **The full system.** Shared reference arm for H1 (joint), H2 and H3. |
| **E2** | Qwen2.5-Coder-Instruct | 7B | **R1** terminal | staged | RL | Free-form response. The unstructured baseline. With E3, isolates H1. |
| **E3** | Qwen2.5-Coder-Instruct | 7B | **R1** terminal | staged | RL | Structured response, monolithic reward. **The hinge of the design:** vs. E2 it isolates *structure*; vs. E1 it isolates *reward decomposition*. |
| **E4** | Qwen2.5-Coder-Instruct | 7B | **R2** no-reasoning | staged | RL | Dense reward minus the two reasoning components. Distinguishes "reasoning shaping helps" from "any dense shaping helps" (H2). |
| **E5** | Qwen2.5-Coder-Instruct | 7B | R0 full | **flat** | RL | No curriculum. Illustrates H3 at the target scale (underpowered — see E5′). |
| **E1′** | Qwen2.5-Coder-Instruct | 3B | R0 full | staged | RL (replication) | Does the H1/H2 direction hold at a second scale? Also the powered H3 arm. |
| **E2′** | Qwen2.5-Coder-Instruct | 3B | R1 terminal | staged | RL (replication) | Free-form, 3B. With E1′/E3′, replicates H1. |
| **E3′** | Qwen2.5-Coder-Instruct | 3B | R1 terminal | staged | RL (replication) | Structured + monolithic, 3B. Replicates the H1 and H2 contrasts. |
| **E5′** | Qwen2.5-Coder-Instruct | 3B | R0 full | **flat** | RL | No curriculum, 3B, **8–10 seeds**. The only arm with enough seeds to actually test H3's collapse-incidence claim. |

**Which comparison tests what** — this is the whole point of the matrix, and it
is deliberately built so that one reference arm (E1) and one hinge arm (E3) serve
all three hypotheses:

| Hypothesis | Primary contrast | Confirms | Joint/secondary contrast |
| --- | --- | --- | --- |
| H1 (structure) | **E3 vs E2** (reward held constant at R1) | E1′/E3′ vs E2′ at 3B | E1 vs E2 = whole-system effect |
| H2 (decomposition) | **E1 vs E3** (structure held constant) | E1′ vs E3′ at 3B | E1 vs E4 = *which* components matter |
| H3 (curriculum) | **E1′ vs E5′** (8–10 seeds, 3B) | — | E1 vs E5 at 7B, illustrative only |

Nine 7B runs are avoidable; five are not. Dropping E3 would collapse H1 and H2
into one confounded comparison, which is exactly the mistake this matrix exists
to avoid.

---

## 3. Reward configurations

All three are implemented as configurations of the **existing**
`TurnRewardCalculator` — component ceilings are already class attributes
([`rewards/turn.py`](../src/agentdebugger/rewards/turn.py)), so an ablation is a
set of ceilings, not a fork of the reward code. No new reward semantics are
introduced.

| Component | Ceiling | **R0** full (shipped) | **R1** terminal | **R2** no-reasoning |
| --- | ---: | :---: | :---: | :---: |
| `format_compliance` | 0.10 | ✅ | ❌ → 0.0 | ✅ |
| `hypothesis_quality` | 0.20 | ✅ | ❌ → 0.0 | ❌ → 0.0 |
| `localization` | 0.15 | ✅ | ❌ → 0.0 | ❌ → 0.0 |
| `fix_quality` | 0.35 | ✅ | ✅ **rescaled to 1.0** | ✅ |
| `semantic_similarity` | 0.10 | ✅ | ❌ → 0.0 | ✅ |
| `efficiency_potential` | 0.10 | ✅ | ❌ → 0.0 | ✅ |
| `penalties` | −0.55 | ✅ | ✅ (kept) | ✅ |

- **R0 — full (shipped, the treatment).** The reward exactly as it exists today.
  Sums to `1.0` on a perfect first-turn solve, floored at `−0.5`.

- **R1 — monolithic / terminal (the H2 control, and the shared reward for H1).**
  *Removed:* format, hypothesis, localization, semantic similarity, efficiency —
  every component that pays for anything other than the test outcome. *Changed:*
  `fix_quality` is rescaled from a `0.35` ceiling to `1.0`, so a solve is still
  worth `1.0` and the two configurations occupy the same range — otherwise a
  reward-magnitude difference would confound the comparison. *Kept:* penalties, so
  that breaking previously-passing tests still costs something (a control that
  cannot punish regressions is a straw man, not a baseline). The result is the
  reward an ordinary outcome-driven RL setup would use: graded pass-rate, plus
  penalties. **This is the arm expected to produce zero-variance GRPO groups on
  hard bugs — the mechanism H2 proposes.**

- **R2 — dense minus reasoning (the H2 discriminator).** *Removed:* exactly two
  components — `hypothesis_quality` (0.20) and `localization` (0.15), i.e. the
  only two that pay for *reasoning about* the bug rather than for the shape or
  the outcome of the answer. Everything else is untouched, including the format
  component, so R2 is still a dense, easy-to-climb reward. If E4 (R2) matches E1
  (R0), the project's claim narrows honestly from "rewarding structured reasoning
  works" to "dense shaping works, and the reasoning terms are decoration" — a
  result worth publishing, and one this matrix is built to be able to find.

Rescaling note: components are *zeroed*, not renormalised, in R2 — the maximum
achievable reward is therefore `0.65`, not `1.0`. GRPO's group-relative advantage
normalisation makes a uniform scale factor largely irrelevant, so R2 is left
un-rescaled to avoid perturbing the relative weight of `fix_quality` against the
components it is being compared with. **[A]** R1 *is* rescaled because there the
gap would otherwise be extreme (max `0.35` vs `1.0`).

---

## 4. Protocol

### 4.1 Scaling strategy — recommendation: **3B + 7B, not 7B + 14B**

The claims are about the *relative* effect of three design choices, not about
absolute capability. A single scale is sufficient to test all three; a second
scale exists only to answer the reviewer question "does this survive outside the
one model you tried?" The cheapest credible second point is **downward**, not
upward:

- **7B (`Qwen2.5-Coder-7B-Instruct`) is the primary scale.** It is the smallest
  model that is plausibly a serious coding agent, it fits comfortably on one
  80GB card with LoRA, and it is a scale reviewers recognise.
- **3B (`Qwen2.5-Coder-3B-Instruct`) is the replication scale.** The repository
  already trains it, the hardware profile for small cards already exists
  ([`HardwareProfile.for_vram`](../src/agentdebugger/training/grpo.py)), and seeds
  are cheap enough there to afford the 8–10 needed to give H3 any power at all.
  A second scale *below* the primary is what makes "the effect is not an artefact
  of one model" defensible.
- **14B is not recommended.** It roughly triples GPU-hours versus 7B (larger
  weights, slower generation — and generation, not the backward pass, dominates
  GRPO's cost), needs either an 80GB card at uncomfortable occupancy or multi-GPU
  sharding, and buys a *weaker* marginal claim than the 3B replication does: an
  upward point tells you the effect survives more capability, but the 3B point
  plus 7B already establishes a direction, and no serious scaling law is
  estimable from two or three points regardless. Spend the saved GPU-hours on
  **seeds**, which is where this design is actually starved.

If budget appears late, add 14B as a single confirmatory E1-vs-E3 pair — not a
full matrix.

### 4.2 API baseline — recommendation: **GPT-4o-mini**

One API baseline, run through the existing
[`ApiAgent`](../src/agentdebugger/agents/api.py) (OpenAI-compatible, so this needs
no new code), in both prompt formats.

**Why GPT-4o-mini and not Qwen-72B-Instruct:** the purpose of an API baseline here
is *calibration of the environment*, not a leaderboard. It answers one question —
"is this benchmark trivially solved by a competent hosted model, i.e. are we
training on something that does not need training?" — and for that, the right
choice is a cheap, strong, widely-recognised model that a reader can price
instantly. GPT-4o-mini costs on the order of cents for the entire held-out set,
needs no GPU, and is the model a practitioner would actually reach for.
Qwen-72B-Instruct is the *wrong* control for the opposite reason it looks
attractive: it is the same family as the trained model but ~10× the size, so a
gap against it confounds *training* with *scale* and invites exactly the
misreading the rest of this design works to prevent. The in-family, same-scale
control is B0 — the untrained 7B — and that is the comparison the claims rest on.
GPT-4o-mini is a sanity check bolted on beside it, and should be reported as one.

**[A]** Assumption: GPT-4o-mini remains available at its current price and
context. If it is retired, substitute any comparably cheap hosted model and say so
— nothing in the design depends on the specific model.

### 4.3 Seeds

- 7B RL arms (E1–E5): **3 seeds** each. Enough to report a mean and a spread on
  the primary DV; *not* enough to test H3's collapse incidence, which is why H3's
  primary test lives at 3B.
- 3B arms (E1′, E2′, E3′): **3 seeds** each.
- **E1′ and E5′ (the H3 pair): 8–10 seeds** each. Collapse incidence is a
  count-out-of-*n* statistic and is hopeless at n=3 (3/3 vs 0/3 gives Fisher
  p = 0.10, which does not clear α = 0.05). At n=8, 8/8 vs 0/8 gives p < 0.001,
  and even 6/8 vs 1/8 clears α. This is the cheapest place in the entire plan to
  buy statistical power, because a 3B LoRA run is a fraction of a 7B one.
- Seeds control model init/sampling **and** the curriculum's bug order. The
  held-out split is fixed once, globally, and is **never** reseeded — otherwise
  the pairing that all the primary tests depend on is destroyed.

### 4.4 Datasets and the held-out split

This is a precondition, not a step: **the experiments cannot be run on the
dataset as it currently stands**, because §0's leakage would make every number
uninterpretable.

1. **Fix the split.** Partition the 90 bugs, stratified by tier, into train and
   held-out. Train on the train side only; report on the held-out side only. This
   needs a `split` argument threaded through `load_bugs()` — a small, additive
   API change (default `"all"`, so nothing existing breaks).
2. **Grow the dataset first.** A 45/45 split leaves ~45 eval bugs, at which point
   the minimum detectable effect is ~30 points (§4.5) and the study is a waste of
   GPU. **Recommendation: grow the pool to ≈180 bugs (CPU-only authoring work, no
   GPU, already filed as [good first issue #1](good_first_issues.md)) and split
   90 train / 90 held-out, stratified by tier.** This keeps the *training* cost
   identical while roughly halving the detectable effect size. It is the highest
   value-per-dollar action in this entire plan.
3. **External transfer set (secondary).** [QuixBugs](https://github.com/jkoppel/QuixBugs)
   (40 single-function Python bugs) matches this environment's format closely
   enough to be adapted into `test_cases` records, and is not authored by us — so
   it tests transfer rather than memorisation of our own bug idioms. Report it as
   a secondary endpoint. **[A]** Assumes the adaptation preserves each bug's
   semantics; every adapted record must still pass `agentdebugger validate`.

Every bug used anywhere must pass `agentdebugger validate` — reference fix passes
all cases, buggy code fails at least one — or the split is measuring noise.

### 4.5 Metrics, success criteria, and what this design can actually detect

**Primary metric.** Held-out solve rate (all test cases pass).
**Secondary.** Mean total reward; per-component reward; regression rate
(`newly_broken > 0`); localization accuracy; format-validity rate; calibration
(high-confidence-and-wrong rate); per-tier breakdowns; and, for H3, across-seed
variance, collapse incidence, entropy, KL, and the degenerate-group fraction.

**Confidence intervals.** Wilson score intervals on every reported solve rate
(the normal approximation misbehaves at the rates near 0 and 1 where tiers 1 and 3
sit). Paired bootstrap (10,000 resamples over bugs) for every *difference* between
arms. Holm–Bonferroni across the three primary comparisons.

**Minimum detectable effect — the number that decides whether this study is worth
running.** At α = 0.05, 80% power, and a solve rate near 0.5 (the worst case for
variance):

| Held-out bugs | MDE, unpaired | MDE, paired (30% discordant) |
| ---: | ---: | ---: |
| 30 | 36 pts | 28 pts |
| **90** | **21 pts** | **16 pts** |
| 180 | 15 pts | 11 pts |

Read this honestly: **with 90 held-out bugs, this study can only detect large
effects** — a 10-point improvement in solve rate would be real, valuable, and
statistically invisible (detecting 10 points unpaired needs ~392 bugs per arm).
Two design choices follow directly, and both are already in the plan: pair every
comparison on the bug (right-hand column, ~25% cheaper in effect size), and grow
the dataset (§4.4). **[A]** The 30% discordance figure is an assumption used only
to size the study; the actual discordance is measured from the runs.

**Success criteria.** Stated per hypothesis in §1. In every case: predicted sign,
plus a Holm-corrected 95% CI excluding zero. A result that does not clear this bar
is reported as "not detected at this sample size" — **never** as a null effect,
because at these *n* the study is not powered to distinguish the two.

### 4.6 GPU estimate

**Every number here is an estimate from stated assumptions, and none of it has
been measured.** The first action of the experimental campaign is to falsify these
assumptions cheaply — see the calibration run below.

**[A] Assumptions.**
1. Generation, not the backward pass, dominates GRPO step time (it is the reason
   `num_generations` is what shrinks first in
   [`HardwareProfile`](../src/agentdebugger/training/grpo.py)).
2. Generation is served by vLLM (TRL's `use_vllm`), not HF `generate`. **This is
   not what the repository does today** and is the single biggest lever on cost;
   without it, multiply every figure below by roughly 3.
3. Batch geometry as the shipped 80GB profile: `batch 8 × 8 generations × 256
   completion tokens` = 64 completions/step.
4. 500 steps per run (the repo default).
5. Reward scoring is moved off the critical path or parallelised across CPU cores.
   **Today it is serial** — 64 sandboxed subprocesses per step, each with up to a
   10s ceiling — and if left serial it, not the GPU, becomes the bottleneck. This
   is an engineering precondition, not an optimisation.
6. 7B ≈ 20 s/step; 3B ≈ 8 s/step, on an A100-80GB under the above.

| | Runs | Seeds | Total runs | Est. h/run | GPU-hours |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7B RL (E1–E5) | 5 | 3 | 15 | 2.8 | **42** |
| 3B RL (E1′,E2′,E3′) | 3 | 3 | 9 | 1.1 | **10** |
| 3B H3 pair (E1′,E5′ extra seeds) | 2 | +7 | 14 | 1.1 | **15** |
| Evaluation (all arms × held-out) | — | — | — | — | **~5** |
| Baselines B0/B1 (inference only) | 2 | — | 2 | <0.5 | **~1** |
| | | | | **Subtotal** | **~73** |
| | | | | **+30% contingency** | **~95** |

- **Recommended hardware:** 1× A100 80GB (or 1× H100). An 80GB card is what the
  shipped batch geometry targets, and it removes the sharding complexity that a
  14B run would force.
- **Wall-clock:** ~95 GPU-hours ≈ **4 days** on a single A100 running
  back-to-back, or **~1 day** on 4 GPUs, since every arm is embarrassingly
  parallel across seeds. Add ~1 day of CPU-only work for the dataset growth and
  the split, which gates everything else.
- **B2 (GPT-4o-mini)** costs API pennies and no GPU.

**Calibration run (do this first, before committing to anything above).** Launch
E1, one seed, `--max-steps 20`, and measure: seconds/step, the share of step time
in generation vs. reward scoring, and peak VRAM. That single ~10-minute run
replaces assumptions 1, 2, 3, 5 and 6 with measurements, and the table above
should be rebuilt from it before any GPU budget is committed. **If reward scoring
turns out to dominate, no amount of GPU solves this problem** and the fix is a
process pool, not a bigger card.

---

## 5. Risks to experimental validity

Ordered by how badly each would damage the conclusions.

| # | Risk | Why it is serious | Mitigation |
| --- | --- | --- | --- |
| 1 | **Train/eval leakage** (§0) — the current code trains and evaluates on the same 90 bugs. | Every claim becomes a claim about memorisation. It invalidates *all three* hypotheses, and it exists in the code today. | Held-out split before any run (§4.4). Report the train-set number too, labelled as such, so the gap between them is visible. |
| 2 | **Underpowered** — 90 held-out bugs detects only ~16–21 point effects. | The most likely outcome of an underpowered study is a true effect reported as "no difference". | Grow to ~180 bugs (CPU-only). Pair every test on the bug. Pre-register MDE (§4.5) and report "not detected", never "no effect". |
| 3 | **Reward hacking** — a dense reward is climbable without fixing code. | Would produce a *confirmation* of H2 that is an artefact: reward rises, debugging does not. | Solve rate (not reward) is the primary DV. Report the reward-vs-solve-rate gap explicitly; a widening gap is the signature. |
| 4 | **The H1 confound** — structure and dense shaping are entangled by construction. | Naively comparing the full system to an unstructured baseline supports a claim about structure that the data cannot support. | E3 (structured + R1) holds reward constant so the only difference is format. Report E1-vs-E2 separately, as a joint effect. |
| 5 | **H3 is untestable at 3 seeds** — collapse incidence is a count statistic. | Would produce a curriculum claim resting on anecdote. | Move H3's primary test to 3B with 8–10 seeds. If neither arm collapses, report that H3 was untestable — do not quietly drop it. |
| 6 | **Sandbox/scoring throughput becomes the bottleneck** — 64 serial subprocesses per step. | Silently turns a 3-day campaign into a 3-week one, and tempts a mid-study reduction in `num_generations` that would break comparability. | Measure in the calibration run (§4.6). Parallelise scoring *before* starting. Never change batch geometry mid-campaign. |
| 7 | **Tier labels may not track difficulty** — the curriculum could be sampling order in disguise. | Undercuts H3's premise. | Verify empirically with the base model's per-tier solve rate (B0) *before* interpreting any curriculum result. |
| 8 | **Free-form parser asymmetry** — a weak fix extractor would fake H1. | Confirms H1 for a bad reason. | Report the free-form arm's extraction-failure rate; void the comparison if it exceeds the structured arm's format-failure rate. |
| 9 | **Single model family** — everything is Qwen2.5-Coder. | Limits external validity; the effect may be an idiosyncrasy of one instruction-tuning recipe. | Two scales (3B, 7B) is a partial answer. State the limitation plainly; do not generalise beyond the family tested. |
| 10 | **Non-determinism** — sandbox timing, GPU non-determinism, the `hard` grader's stress test. | Adds variance that is easy to mistake for an effect. | Fixed seeds; the held-out split fixed globally; report across-seed variance as a first-class number rather than averaging it away. |

---

## 6. What this plan deliberately does not claim

- It does not claim the environment transfers to real repositories. It operates on
  single functions (see the limitations in [report.md](report.md)).
- It does not claim GRPO beats PPO here. That comparison is on the roadmap and is
  not tested by this matrix.
- It does not claim any effect size in advance. Note on provenance: the historical prototype run in [`results/historical/`](../results/historical/) predates the train/held-out split and must not be cited as evidence for any hypothesis; all formal claims are evaluated on the 90 held-out bugs in [`results/primary/`](../results/primary/).
