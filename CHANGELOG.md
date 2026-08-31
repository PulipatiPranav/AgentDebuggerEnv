# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-08-29

Controlled empirical study of reward decomposition for GRPO program repair.

### Added

- **180-bug dataset with disjoint train/held-out split** (`src/agentdebugger/dataset/bugs/`): 90 training bugs and 90 held-out evaluation bugs stratified into Easy, Medium, and Hard tiers, resolving the single-split precondition of the pre-registration.
- **Nine primary RL training runs** across three reward configurations on `Qwen2.5-Coder-3B-Instruct`:
  - `E1` (R0 Full Dense): seeds 42, 123, 456
  - `E3` (R1 Graded Outcome): seeds 42, 123, 456
  - `E4` (R2 Dense No Reasoning): seeds 42, 123, 456
  Saved with full per-bug evaluation records in [`results/primary/`](results/primary/).
- **Paired statistical analysis suite** (`analysis/bootstrap.py`): 10,000 paired bootstrap iterations over bugs with fixed RNG seed (204), exact McNemar sensitivity tests, and Holm--Bonferroni corrections.
- **Adapted QuixBugs benchmark** (`data/quixbugs/`): 27 single-function Python programs adapted for out-of-distribution evaluation.
- **Scientific manuscript** (`docs/paper_preprint.pdf`) with official NeurIPS 2026 formatting and full reproducibility bounds.

### Changed

- Updated `README.md` to reflect empirical findings: dense reward shaping substantially reduces GRPO group degeneracy (16.2% vs 61.9%) but does not produce a detectable improvement in held-out solve rate (45.6% vs 48.5%).
- Relocated historical pre-split prototype run to `results/historical/`.

### Known limitations

- Evaluated on `Qwen2.5-Coder-3B-Instruct` with group size $G=2$ and single-turn single-function repair.
- With $N=90$ held-out bugs, the study is powered to detect large effects ($\ge 16$ pp paired); small effects of a few percentage points cannot be ruled out.
- QuixBugs external evaluation currently lacks a zero-shot base model run, making its performance figures descriptive.

## [0.1.0] - 2026-07-14

First public release. `0.1.0` is the initial released version: the package
carried a `2.0.0` version string during pre-release development, but no version
was ever tagged or published, so the public history starts here.

### Added

- **Execution sandbox** (`agentdebugger.sandbox`) — three layers, applied in
  order: static AST analysis of blocked imports, builtins and dunder attributes
  before any child is spawned; kernel `setrlimit` ceilings on address space, CPU
  time and file-size in the child; and a wall-clock deadline in the parent that
  kills the child's whole process group. The child runs in a scratch directory
  with a scrubbed environment and Python's isolated mode, and its output is made
  deterministic so rewards computed from it are reproducible.
- **Two environments** over one shared core. `TaskEnvironment` is a multi-step
  conversation with a broken program that refuses to execute a fix unless a
  hypothesis accompanies it. `CurriculumEnvironment` is the single-turn setting
  GRPO trains against, in the fixed OBSERVATION / HYPOTHESIS / CONFIDENCE /
  ACTION / DETAIL format.
- **Three hand-written tasks**, each isolating a distinct debugging failure mode:
  an off-by-one named by the stack trace (`easy`), a red herring whose error
  message points at the wrong function (`medium`), and a race condition that
  passes every sequential test (`hard`).
- **A dense, itemised turn reward** (`agentdebugger.rewards`) decomposed into
  format compliance, hypothesis quality, localization, fix quality, semantic
  similarity, efficiency and penalties, with a defended range of `[-0.5, 1.0]`.
  Fix quality is the largest component, so no amount of plausible prose can
  substitute for passing the tests.
- **A 90-bug tiered dataset** (40 / 30 / 20 across three difficulty tiers), each
  record carrying a buggy implementation, a reference fix and executable test
  cases, plus `agentdebugger validate` — which runs every reference fix and every
  buggy implementation through the sandbox and asserts the fix passes and the
  buggy code fails.
- **A curriculum schedule** (`agentdebugger.config`) that unlocks tier 2 at step
  150 and tier 3 at step 350, with the trainer swapping the bug pool at each
  boundary.
- **GRPO training** (`agentdebugger.training`) for `Qwen2.5-Coder-3B-Instruct`
  with LoRA, scoring completions with the same `score_response` the evaluator
  calls, and scaling batch geometry to the detected GPU (T4 through H100).
- **Agents**: an offline `OracleAgent` that submits the reference fix, and an
  `ApiAgent` for any OpenAI-compatible endpoint.
- **Evaluation** for both settings, emitting JSON reports; published results in
  [`results/`](results/).
- **A FastAPI server** (`agentdebugger serve`) exposing `/reset`, `/step` and
  `/state` over the multi-step environment.
- **The `agentdebugger` CLI**: `episode`, `evaluate`, `evaluate-curriculum`,
  `validate`, `tasks`, `serve` and `train`.
- **Documentation**: a [technical report](docs/report.md), an
  [architecture guide](docs/architecture.md), a
  [pre-registered research plan](docs/research_plan.md) stating the project's
  three hypotheses and the experiment matrix designed to test them, and
  [good first issues](docs/good_first_issues.md).
- **CI** on every push and pull request: Ruff, and pytest across Python
  3.10–3.13 on Linux plus 3.12 on macOS.

### Known limitations

- The environment operates on single, isolated functions, not multi-file
  repositories — the primary limit on real-world transfer.
- The sandbox is a workstation-grade defence, suitable for the code an LLM
  actually produces on a developer machine or CI box. It is not a substitute for
  a container or VM when running deliberately adversarial code.
- The `hard` task's grader detects the race empirically, by stress testing under
  a reduced GIL switch interval. It can demonstrate a lost update, never prove
  thread safety.
- The hypotheses in [docs/research_plan.md](docs/research_plan.md) were stated but
  not yet tested in v0.1.0 *(resolved in v0.2.0 for the reward decomposition study)*.

[Unreleased]: https://github.com/shasshaank/AgentDebuggerEnv/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/shasshaank/AgentDebuggerEnv/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/shasshaank/AgentDebuggerEnv/releases/tag/v0.1.0
