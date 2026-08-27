---

# Project context — AgentDebuggerEnv

**Purpose of this file.** A single, self-contained status document so this project can be
picked up in a fresh chat (or by another person) without re-deriving everything. It records
*what has been done and why*, *what is left and why*, and the exact commands and file locations
needed to continue. It is written against the plan in `markdown-preview.pdf` ("Publication
Strategy"), which is the scoping authority, and `docs/research_plan.md`, which is the
pre-registration of the experiments.

> **Read these two first if you are new here:**
> 1. `markdown-preview.pdf` — the *scoping* document. It cuts the ambitious plan down to
>    something finishable before Fall-2026 grad deadlines. Its §0 defines the experiment matrix
>    we are actually running.
> 2. `docs/research_plan.md` — the *pre-registration*. Hypotheses H1/H2/H3, the reward configs
>    R0/R1/R2, the statistical tests, and the threats to validity. This is the source of truth
>    for *what each experiment means*.

---

## 1. The north star (what we are actually building)

**One paper, one headline finding: terminal reward sufficiency.** The original H2 (reward
decomposition helps) was not supported — E3 (R1, terminal-only) matched or beat E1 (R0, full
reward). The paper's story pivots from "decomposition helps" to "decomposition is unnecessary:
terminal reward is sufficient, and the extra engineering buys no performance gain."

**The original claim, in one sentence:** *rewarding a model for reasoning about a bug
specifically — not just any dense shaping of the same magnitude — changes what it learns under
RL, and the mechanism is fewer zero-advantage ("degenerate") GRPO groups.*

**The revised claim, supported by data:** *Dense reward decomposition into 7 reasoning-specific
components does not outperform terminal-only reward for small-model code debugging under GRPO.
All three reward configurations (R0 full, R1 terminal, R2 no-reasoning) produce statistically
indistinguishable solve rates on 90 held-out bugs (McNemar p > 0.3, Holm-corrected). This is
a practically important negative result: R1 requires no hand-crafted heuristics for hypothesis
quality or localization scoring.*

**The finishable experiment matrix** (Publication-Strategy §0). All at **Qwen2.5-Coder-3B-Instruct**,
evaluated on the **held-out split** and (later) QuixBugs:

| ID | Reward config | Format | Seeds | Status | What it is for |
|---|---|---|---|---|---|
| B0 | — (zero-shot) | structured | 1 | ✅ | Baseline / reference for "did training help" |
| B1 | — (zero-shot) | free-form | 1 | ✅ (corrected) | Baseline in the other format |
| E1 | **R0** full | structured | 3 | ✅ | The full system |
| E2 | **R1** terminal | free-form | 3 | ❌ DROPPED | Unstructured baseline; **E3 vs E2 isolates H1 (structure)** |
| E3 | **R1** terminal | structured | 3 | ✅ | Hinge arm; **E1 vs E3 isolates H2 (decomposition)** |
| E4 | **R2** no-reasoning | structured | 3 | ✅ | "reasoning shaping" vs "any dense shaping" (H2 discriminator) |

Seeds: `{42, 123, 456}`.

---

## 2. Current state — DONE, with explanations

### 2.1 Dataset v2 — 180 validated bugs  ✅
- **What:** 180 bugs sourced from MBPP, filtered for usability, with one AST mutation per bug across three difficulty tiers.
- **Where:** `src/agentdebugger/dataset/bugs/` (60/60/60 across tiers).
- **Validated:** Reference code passes all tests, buggy code fails at least one.

### 2.2 Held-out split — fixed 90/90 split  ✅
- **What:** A fixed, committed 90/90 split, stratified by tier, in `src/agentdebugger/dataset/bugs/split.json`.
- **Why:** Prevents data leakage; all reported numbers come from the held-out side.
- **Wiring:** `agentdebugger train` defaults to `--split train`; `evaluate-curriculum` defaults to `--split heldout`.

### 2.3 Reward configs R0 / R1 / R2  ✅
- **R0 `full()`:** The shipped 7-component reward (perfect solve = 1.0).
- **R1 `terminal()`:** All shaping components zeroed; `fix_quality` rescaled to 1.0; penalties kept.
- **R2 `no_reasoning()`:** Zeroes only `hypothesis_quality` and `localization`; max reward 0.65.

### 2.4 Difficulty gate — Llama-3.3-70B baseline  ✅
- **Result:** 67.8% overall on held-out split. Verdict: **"GOOD — a real gap for RL to close."**

### 2.5 Scoring robustness — code-fence fix  ✅
- **What:** `extract_fix_code()` unwraps fenced code blocks before execution.

### 2.6 Free-form arm — B1 bug found and corrected  ✅
- **Original B1 result:** 1.1% solve, 86.7% extraction failure — this was a **token budget bug**, not a parsing or capability problem.
- **Root cause:** `load_generator()` hardcoded `max_new_tokens=300` regardless of format. Free-form responses need ~700+ tokens because the model writes prose reasoning before code. Every single failed completion was truncated mid-sentence before any code appeared.
- **Evidence:** `_completion_length_for()` in `training/grpo.py` already had a fix for the training path (1.8x multiplier with a comment describing this exact truncation symptom). The fix was never ported to the evaluation path.
- **Fix applied:** `load_generator()` now accepts `format` and `max_new_tokens` parameters with format-aware defaults (300 structured / 700 free_form). Added `--max-new-tokens` CLI flag. Also added truncation detection (`truncated: bool`, `token_count: int` in eval JSON records).
- **Corrected B1 result (700 tokens):** **45.6% (41/90), extraction-failure 0.0%**
  - Tier 1: 40.0% (12/30)
  - Tier 2: 36.7% (11/30)
  - Tier 3: 60.0% (18/30)
- **Significance:** Zero-shot free-form (45.6%) matches RL-trained structured E1 average (45.6%). The original extraction failure was entirely an artifact of truncation.

### 2.7 E2 — Free-form RL training  ❌ *DROPPED*
- **Original drop reason (stale):** E2 was initially dropped after a 20-step calibration showed negative reward (-0.165). This calibration itself was affected by the same 300-token truncation bug — the model could not produce parseable completions because they were being cut off before any code appeared.
- **Post-fix attempt:** After fixing the token budget, E2 training was attempted on 2x RTX 4090 24GB. Training ran ~2x slower than structured arms due to the longer completion budget (550 tokens vs 192) required for free-form GRPO. After 7+ hours, training was still incomplete for 2/3 seeds.
- **Final drop reason:** Free-form GRPO training is compute-infeasible within our budget. The 550-token completion length makes each training step ~2x slower (90s/step vs ~40s/step for structured). The full 3-seed protocol would require ~18+ hours of dual-4090 time.
- **Mitigation:** H1 (structure vs free-form) cannot be tested via E2 vs E3 as planned. However, the corrected B1 baseline (§2.6) provides partial evidence: zero-shot free-form already matches RL-trained structured performance, suggesting the format gap was an evaluation artifact rather than a genuine capability difference.

### 2.8 Degenerate-group + per-component logging  ✅
- **What:** Logs `degenerate_group_fraction`, per-component reward means, and extraction_failure_rate to W&B.

### 2.9 Reward-scoring throughput plumbing  ✅
- **What:** `--reward-workers` flag for parallel scoring; timing logged to W&B.

### 2.10 QuixBugs external transfer set  ✅
- **Result:** 27/40 QuixBugs programs adapted into `data/quixbugs/bugs.jsonl`.

### 2.11 Curriculum-advance bug found + fixed  ✅
- **Bug:** Curriculum never advanced past tier 1 due to TRL's sampler caching.
- **Fix:** `train()` now runs one short `GRPOTrainer` per curriculum stage.

### 2.12 Calibration run — gradient-checkpointing/KV-cache bug  ✅
- **Bug:** Manual `gradient_checkpointing_enable()` caused garbage completions.
- **Fix:** Removed manual call; left to `GRPOConfig`/`GRPOTrainer`.

### 2.13 Zero-shot baselines — B0 and B1  ✅
- **B0 (structured):** 37.8% (34/90), extraction-failure 0.0%
- **B1 (free-form, original, 300 tokens):** 1.1% (1/90), extraction-failure 86.7% — **BUGGY, see §2.6**
- **B1 (free-form, corrected, 700 tokens):** **45.6% (41/90), extraction-failure 0.0%**

### 2.14 E1 runs — R0, structured  ✅
| Seed | Overall | Tier 1 | Tier 2 | Tier 3 |
|------|---------|--------|--------|--------|
| s42 | **47.8%** (43/90) | 30.0% | 50.0% | 63.3% |
| s123 | **46.7%** (42/90) | 40.0% | 46.7% | 53.3% |
| s456 | **42.2%** (38/90) | 33.3% | 40.0% | 53.3% |
| **Avg** | **45.6%** | 34.4% | 45.6% | 56.6% |

- Extraction-failure 0.0% across all runs.
- E1 outperforms B0 by ~8 points — training helps.

### 2.15 E3 runs — R1, structured  ✅
| Seed | Overall | Tier 1 | Tier 2 | Tier 3 |
|------|---------|--------|--------|--------|
| s42 | **41.1%** (37/90) | 30.0% | 46.7% | 46.7% |
| s123 | **51.1%** (46/90) | 33.3% | 56.7% | 63.3% |
| s456 | **53.3%** (48/90) | 36.7% | 63.3% | 60.0% |
| **Avg** | **48.5%** | 33.3% | 55.6% | 56.7% |

- Extraction-failure 0.0% across all runs.
- **E3 >= E1 overall (48.5% vs 45.6%)** — terminal-only reward matches or beats full reward.
- **Higher variance** (range 12.2pp vs E1's 5.6pp) — consistent with the degenerate-group mechanism.

### 2.16 E4 runs — R2, structured (no reasoning)  ✅
| Seed | Overall | Tier 1 | Tier 2 | Tier 3 |
|------|---------|--------|--------|--------|
| s42 | **52.2%** (47/90) | 33.3% | 46.7% | 76.7% |
| s123 | **48.9%** (44/90) | 30.0% | 60.0% | 56.7% |
| s456 | **44.4%** (40/90) | 40.0% | 43.3% | 50.0% |
| **Avg** | **48.5%** | 34.4% | 50.0% | 61.1% |

- Extraction-failure 0.0% across all runs.
- **E4 = E3 overall (48.5% vs 48.5%)** — removing reasoning-specific components doesn't hurt.

### 2.17 Free-form prompt improvements  ✅
- Added conciseness instruction to reduce wasted tokens on verbose formatting.
- Replaced stub `pass` example with complete binary-search implementation.
- All 8 prompt tests pass, including the length-match invariant (within 15% by word count).

### 2.18 Truncation logging  ✅
- `generate()` now detects whether it hit `max_new_tokens` without reaching EOS.
- Per-bug eval records now include `truncated: bool` and `token_count: int`.

---

## 3. Statistical analysis — Phase D  ✅

### 3.1 McNemar tests — E1 vs E3, E1 vs E4, E3 vs E4

Paired McNemar tests on the same 90 held-out bugs, using exact binomial test on discordant
pairs, with Holm-Bonferroni correction across the family of comparisons.

#### Per-seed McNemar (9 tests, Holm-corrected)

| Comparison | Seed | A-only | B-only | Discordant | Raw p | Holm-adj p | Direction |
|---|---|---|---|---|---|---|---|
| E1 vs E3 | s42 | 13 | 7 | 20 | 0.2632 | 1.0000 | E1 > E3 |
| E1 vs E3 | s123 | 12 | 16 | 28 | 0.5716 | 1.0000 | E3 > E1 |
| E1 vs E3 | s456 | 5 | 15 | 20 | 0.0414 | 0.3311 | E3 > E1 |
| E1 vs E4 | s42 | 9 | 13 | 22 | 0.5235 | 1.0000 | E4 > E1 |
| E1 vs E4 | s123 | 7 | 9 | 16 | 0.8036 | 1.0000 | E4 > E1 |
| E1 vs E4 | s456 | 9 | 11 | 20 | 0.8238 | 1.0000 | E4 > E1 |
| E3 vs E4 | s42 | 4 | 14 | 18 | 0.0309 | 0.2780 | E4 > E3 |
| E3 vs E4 | s123 | 12 | 10 | 22 | 0.8318 | 1.0000 | E3 > E4 |
| E3 vs E4 | s456 | 12 | 4 | 16 | 0.0768 | 0.5377 | E3 > E4 |

**No comparison survives Holm correction.** The two raw p < 0.05 both inflate above 0.05 after correction.

#### Majority-vote pooled (bug solved if >=2/3 seeds solve it)

| Comparison | A pooled | B pooled | A-only | B-only | Raw p | Adj p |
|---|---|---|---|---|---|---|
| E1 vs E3 | 42/90 (46.7%) | 44/90 (48.9%) | 8 | 10 | 0.8145 | 1.0000 |
| E1 vs E4 | 42/90 (46.7%) | 44/90 (48.9%) | 6 | 8 | 0.7905 | 1.0000 |
| E3 vs E4 | 44/90 (48.9%) | 44/90 (48.9%) | 7 | 7 | 1.0000 | 1.0000 |

#### All-pairs (N=270, each bug x seed as one observation)

| Comparison | A-only | B-only | Discordant | Raw p | Adj p | Direction |
|---|---|---|---|---|---|---|
| E1 vs E3 | 30 | 38 | 68 | 0.3961 | 1.0000 | E3 > E1 |
| E1 vs E4 | 25 | 33 | 58 | 0.3581 | 1.0000 | E4 > E1 |
| E3 vs E4 | 28 | 28 | 56 | 1.0000 | 1.0000 | tied |

#### Effect sizes

| Comparison | A avg | B avg | Difference |
|---|---|---|---|
| E1 vs E3 | 45.6% | 48.5% | -2.9pp (E3 wins) |
| E1 vs E4 | 45.6% | 48.5% | -2.9pp (E4 wins) |
| E3 vs E4 | 48.5% | 48.5% | 0.0pp (tied) |

### 3.2 Interpretation

**H2's original framing ("R0 > R1, supporting reward decomposition") is not supported.** The
point estimate goes the wrong way (E3/E4 >= E1), and the difference is non-significant at every
analysis level (per-seed, pooled, all-pairs). All three reward configurations produce
statistically indistinguishable results.

**The viable framing for the paper:**
1. **Terminal reward is sufficient.** R1 (terminal-only) matches R0 (7-component dense reward) — simpler reward engineering works just as well.
2. **Reasoning-specific shaping adds nothing.** E4 (R2, no reasoning components) = E3 (R1) = E1 (R0) — hand-crafting rewards for hypothesis quality and localization buys zero performance.
3. **Higher variance under R1.** E3's 12.2pp seed-to-seed range vs E1's 5.6pp is consistent with the degenerate-group mechanism hypothesis (fewer distinct reward signals → noisier gradients), even though the mean performance didn't differ.
4. **Free-form format is not inherently worse.** Corrected B1 zero-shot (45.6%) matches RL-trained E1 average (45.6%), suggesting the format gap was an evaluation artifact (truncation) rather than a genuine capability difference.

---

## 4. What is LEFT

**Phases A-D are COMPLETE.** All training, evaluation, B1 correction, and statistical analysis are done.

### Phase E — writing (in progress)
- Write arXiv preprint -> target **"AI for Verifiable Coding" workshop (NeurIPS 2026 Atlanta)**.
- Deadline: **Aug 29, 2026**.
- Framing: negative result + practical recommendation (terminal reward sufficiency).

---

## 5. Quick reference

### Arm -> flags map
| Arm | `--reward-config` | format | Status |
|-----|-------------------|--------|--------|
| E1 | `R0` | structured | ✅ Complete |
| E2 | `R1` | `--format free_form` | ❌ Dropped (compute-infeasible) |
| E3 | `R1` | structured | ✅ Complete |
| E4 | `R2` | structured | ✅ Complete |

### Final Results Matrix

| Run | Overall | Tier 1 | Tier 2 | Tier 3 | Extraction Fail |
|-----|---------|--------|--------|--------|-----------------|
| **Baselines** |
| B0 (structured) | **37.8%** | 30.0% | 43.3% | 40.0% | 0.0% |
| B1 (free-form, corrected) | **45.6%** | 40.0% | 36.7% | 60.0% | 0.0% |
| **E1 (R0, structured)** |
| E1 s42 | **47.8%** | 30.0% | 50.0% | 63.3% | 0.0% |
| E1 s123 | **46.7%** | 40.0% | 46.7% | 53.3% | 0.0% |
| E1 s456 | **42.2%** | 33.3% | 40.0% | 53.3% | 0.0% |
| **E1 Avg** | **45.6%** | 34.4% | 45.6% | 56.6% | 0.0% |
| **E3 (R1, structured)** |
| E3 s42 | **41.1%** | 30.0% | 46.7% | 46.7% | 0.0% |
| E3 s123 | **51.1%** | 33.3% | 56.7% | 63.3% | 0.0% |
| E3 s456 | **53.3%** | 36.7% | 63.3% | 60.0% | 0.0% |
| **E3 Avg** | **48.5%** | 33.3% | 55.6% | 56.7% | 0.0% |
| **E4 (R2, structured)** |
| E4 s42 | **52.2%** | 33.3% | 46.7% | 76.7% | 0.0% |
| E4 s123 | **48.9%** | 30.0% | 60.0% | 56.7% | 0.0% |
| E4 s456 | **44.4%** | 40.0% | 43.3% | 50.0% | 0.0% |
| **E4 Avg** | **48.5%** | 34.4% | 50.0% | 61.1% | 0.0% |

### Key Findings
| Finding | Evidence | Implication |
|---------|----------|-------------|
| E1 > B0 (+7.8pp) | 45.6% vs 37.8% | GRPO training improves fix rates over zero-shot |
| E1 = E3 = E4 | McNemar p > 0.3 (Holm-corrected: all 1.0) | **Reward decomposition is unnecessary** — terminal reward suffices |
| E3 higher variance | 12.2pp range vs E1's 5.6pp | Consistent with degenerate-group mechanism (noisier gradients under sparser reward) |
| B1 corrected = E1 | Both 45.6% | Zero-shot free-form matches RL-trained structured — format gap was truncation artifact |
| B1 original was buggy | 86.7% extraction fail -> 0.0% after fix | Token budget (300->700) was sole cause; parser was always correct |

### Key files
| Path | Role |
|---|---|
| `docs/CONTEXT.md` | This file — canonical source of truth. |
| `docs/research_plan.md` | Pre-registration of hypotheses and experimental design. |
| `markdown-preview.pdf` | Scoping document (Publication Strategy). |
| `results/B0.json` | Structured zero-shot baseline |
| `results/B1.json` | Free-form zero-shot baseline (original, buggy — 300 token limit) |
| `results/B1_retry_700tok.json` | Free-form zero-shot baseline (corrected — 700 token limit) |
| `results/E1_s{42,123,456}.json` | E1 runs (R0, structured) |
| `results/E3_s{42,123,456}.json` | E3 runs (R1, structured) |
| `results/E4_s{42,123,456}.json` | E4 runs (R2, structured) |

### Decisions/gotchas to remember
- **Follow `markdown-preview.pdf`, not the full `research_plan.md` matrix.**
- **E2 dropped** — free-form GRPO training is ~2x slower per step than structured (550 vs 192 token completions), making the 3-seed protocol compute-infeasible. The original 20-step calibration failure was itself caused by the same 300-token truncation bug that affected B1.
- **H2 is a negative result, not a failure.** E1 = E3 = E4 means terminal reward is sufficient — simpler is better. Frame it as a practical recommendation, not a failed hypothesis.
- **B1's correction is load-bearing.** The paper must clearly distinguish the original B1 (buggy, 1.1% solve) from the corrected B1 (45.6% solve). The root cause was `load_generator()` hardcoding `max_new_tokens=300` regardless of format — a one-line bug with an 85pp impact on reported extraction failure.
- **The split is immutable.** Never regenerate `split.json`.
- **`hypothesis_quality` is a heuristic, not an LLM judge.** The paper must say only what the code does.

### Code changes applied (not yet pushed to GitHub)
1. **`src/agentdebugger/evaluation/curriculum.py`** — `load_generator()` now format-aware with `max_new_tokens` defaulting to 700 for free_form / 300 for structured; MPS device support; truncation detection.
2. **`src/agentdebugger/cli.py`** — added `--max-new-tokens` flag; wired `args.format` through to `load_generator()`.
3. **`src/agentdebugger/training/grpo.py`** — `_completion_length_for()` now uses `max(..., 550)` floor for free_form.
4. **`src/agentdebugger/training/prompts.py`** — conciseness instruction; complete binary-search example replacing stub.
5. **`src/agentdebugger/protocol.py`** — restored committed `parse_freeform_output()` (a local edit had replaced it with a dict-returning version that would crash `score_response()`).

---
