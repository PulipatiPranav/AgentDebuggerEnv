# Good first issues

Five self-contained pieces of work, each useful to the project and small enough
to be a first contribution. Each is written so it can be pasted into a GitHub
issue as-is.

New to the repository? Start with [CONTRIBUTING.md](../CONTRIBUTING.md), then run
`pytest` and `agentdebugger episode --task hard` to see the environment work.

---

## 1. Grow the tier-3 bug pool

**Difficulty:** easy · **Files:** `src/agentdebugger/dataset/bugs/bugs_tier3.jsonl`, `tests/test_dataset.py`

The dataset is deliberately top-light: 40 tier-1 bugs, 30 tier-2, and only **20
tier-3**. Tier 3 is the tier that discriminates between a model that reasons and
one that pattern-matches, and it is the tier where a per-tier solve rate is
computed over the fewest samples — so it is also the noisiest number the project
reports.

Add 10–20 new tier-3 bugs: subtle logic errors, mutable-default arguments,
incorrect boundary handling in recursion, silent integer/float coercion, state
leaking across calls. Each record needs a buggy implementation, a reference fix,
a plausible `initial_error`, a `bug_location`, and test cases that the reference
fix passes and the buggy code fails on at least one.

**Why it matters:** it directly tightens the confidence interval on every tier-3
result the project publishes, and it needs no GPU.

**Done when:** `agentdebugger validate --tiers 3` is green, and the new bugs are
not near-duplicates of existing ones (vary the *reasoning* required, not just the
identifiers).

---

## 2. Report a confidence interval alongside every solve rate

**Difficulty:** easy–medium · **Files:** `src/agentdebugger/evaluation/curriculum.py`, `src/agentdebugger/cli.py`, `tests/`

`evaluate-curriculum` reports a bare solve rate per tier — `0.45` on 20 tier-3
bugs. A solve rate over 20 binary trials has a Wilson 95% interval roughly
±0.2 wide, and nothing in the output says so, which invites over-reading small
differences between runs.

Add a Wilson score interval to `TierResult` and `CurriculumReport`, surface it in
the CLI output (`solve rate 45.0% [25.8, 65.8]`) and in the JSON. Wilson rather
than the normal approximation because it stays sane at rates near 0 and 1, which
is exactly where tier 1 and tier 3 sit. No dependency needed — it is a closed
form, a few lines of `math`.

**Why it matters:** it is the difference between "the trained model beats the
baseline" and "the trained model beats the baseline, and here is whether that is
distinguishable from noise."

**Done when:** the interval appears in CLI and JSON output, and a test pins it
against a known reference value (9 solved of 20, at 95%, is [0.2582, 0.6579]).

---

## 3. Make `agentdebugger validate` explain *why* a bug is unsound

**Difficulty:** easy · **Files:** `src/agentdebugger/dataset/validate.py`, `src/agentdebugger/cli.py`, `tests/test_dataset.py`

When a contributor adds a bug and `validate` rejects it, the failure output names
the bug and the problem — but not the actual sandbox output that led to the
verdict. So the first thing anyone does is re-run the code by hand to see the
traceback, which is exactly the work `validate` already did and threw away.

Attach the failing test case's input, the expected output, and the value actually
produced (or the exception) to the failure record, and print it under the failing
bug. Keep it truncated — the sandbox already caps captured output.

**Why it matters:** adding a bug is the single most common contribution this
project will receive (see issue 1), and this turns its main friction point into a
readable error.

**Done when:** a deliberately broken record produces output a contributor can act
on without re-running anything, and a test asserts the failure detail is present.

---

## 4. Log per-component reward breakdowns during training

**Difficulty:** medium · **Files:** `src/agentdebugger/training/grpo.py`, `src/agentdebugger/rewards/turn.py`

The reward function is decomposed into seven components precisely so that a
training curve can show *which behaviour* is improving — but
`make_reward_function` returns only `reward.total` to TRL, so the components are
computed and then discarded. `TurnRewardCalculator.mean_components` already
produces exactly the right dict (and `CurriculumEnvironment.episode_metrics`
already uses it, keyed for W&B), but nothing in `training/grpo.py` calls it.

Add a TRL callback (or a wrapper around the reward function) that accumulates the
per-component means over each step's completions and logs them to W&B under
`reward/format_compliance`, `reward/hypothesis_quality`, and so on, when W&B is
active — the trainer already detects this via `_wandb_active()`.

**Why it matters:** without this, "the model learned the format in the first 50
steps, then started actually localising bugs" is an anecdote. With it, it is a
plot — and it is the primary diagnostic for reward hacking, since a total that
rises while `fix_quality` stays flat is the signature.

**Done when:** a training run logs all seven components per step, and a test
covers the accumulation logic without requiring a GPU or W&B.

---

## 5. Sandbox-policy escapes as a regression corpus

**Difficulty:** medium · **Files:** `src/agentdebugger/sandbox/policy.py`, `tests/test_sandbox.py`

`tests/test_sandbox.py` covers a set of named escapes — `import os`,
`__subclasses__`, `eval`, and friends. Each is an individual test, which means
adding a newly discovered escape means writing another test function, and there
is no single place to see what the policy claims to block.

Refactor the escape tests into one parametrised corpus — a list of
`(name, source, should_be_blocked)` cases covering both halves of the policy's
contract: the escapes that must be refused, *and* the legitimate code
(`hashlib`, `threading`, `super()`, comprehensions, dataclasses) that must still
run. Then extend it: `breakpoint()`, `__import__` via `getattr` on a string,
`compile`, `memoryview`, indirect access through `vars()`, `.__mro__`.

**Why it matters:** the sandbox is the component whose failure is a real security
incident on a contributor's machine, and a corpus makes "does the policy still
block X?" a one-line addition instead of a design decision.

**Done when:** the corpus replaces the individual escape tests with no loss of
coverage, and at least four new escape attempts are added — each either blocked,
or filed as a follow-up issue if it turns out the policy lets it through.
