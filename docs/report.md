# Teaching language models to debug, not guess

AgentDebuggerEnv is a reinforcement-learning environment that rewards a language
model for reasoning about a bug before it acts on it. This report explains what
the environment measures, how the reward is shaped, and what the published
training run showed.

## The problem

A language model asked to fix a bug will almost always return a fix. On easy
bugs it is usually right. On subtle ones — a red herring in the error message, a
race that no sequential test exercises — the failure is consistent: the model
produces a confident, plausible-looking patch that does not address the cause,
because nothing in how it was trained rewards it for checking.

This is an incentive problem. A model trained to complete text has no gradient
that pays for slowing down, forming a hypothesis, and testing it. So it does not.
AgentDebuggerEnv supplies that gradient.

## What the environment asks of an agent

Two environments share one core.

The **multi-step environment** (`TaskEnvironment`) is a conversation with a
broken program. The agent submits a fix, asks for context, or gives up. A fix is
**not executed unless it is accompanied by a hypothesis** — the single hardest
constraint in the design. Every submission runs in the sandbox and the agent
sees the real test output. The episode is scored by a per-task grader.

The **single-turn environment** (`CurriculumEnvironment`) is what GRPO trains
against. The agent sees one buggy function and must reply in a fixed
OBSERVATION / HYPOTHESIS / CONFIDENCE / ACTION / DETAIL format. The response is
parsed, any proposed fix is executed against the bug's test cases, and a dense
reward prices the result.

## Three tasks, three failure modes

The hand-written tasks are chosen so that each isolates a distinct way LLM
debugging fails, and so that a strong general model cannot pass all three by
doing the same thing.

**Easy — off-by-one in binary search.** The error output names the symptom
precisely. This is a floor: an agent that reads a stack trace solves it in one
attempt. It exists to confirm the pipeline works, not to discriminate.

**Medium — a red herring in an authentication module.** Four of ten tests fail,
and every failure message names `authenticate_user`. `authenticate_user` is
correct. The bug is one call down the stack in `hash_password`, which wraps its
hex digest in `str(bytes(...))` and so returns `"b'5f4dcc…'"` instead of the
digest. The task measures whether the agent traces a symptom to its cause or
stops at the frame the error mentions. The grader scores this directly: blaming
only `authenticate_user` earns nothing, naming `hash_password` with a reason
earns full credit.

**Hard — a race condition invisible to every sequential test.** A connection
counter's `increment` reads the count, calls a helper to clamp it, then writes it
back. Under CPython's GIL a thread can be preempted at that call, so two threads
read the same value and write the same value: one update is lost. **All eight
tests pass on the buggy code.** An agent that trusts a green suite scores zero. To
solve it, the agent has to recognise that a passing test suite is not proof of
thread safety and guard the read-modify-write with a lock.

> A note on faithfulness: a bare `self.count += 1` does *not* race on CPython
> 3.12+, because the interpreter only checks for a thread switch at calls and
> backward jumps, never between the load and the store of an in-place add. The
> call to the clamp helper is what makes the race real — and it is exactly the
> kind of innocuous helper that hides a race in production. The grader confirms
> the bug by hammering the submitted counter from many threads with a reduced
> GIL switch interval, so a racy counter loses updates on every run and a locked
> one never does.

## The sandbox

Executing model-generated code is a remote-code-execution problem. The sandbox
answers it in three layers, in the order they take effect:

1. **Static analysis**, in the parent process, before any child is spawned. The
   candidate source is parsed and its AST walked for blocked imports (anything
   reaching the filesystem, network, another process, or the interpreter
   internals), blocked builtins (`eval`, `exec`, `open`, `getattr`, …), and
   blocked dunder attributes (`__class__`, `__subclasses__`, `__globals__`, …).
   A violation is refused without ever running the code.
2. **Kernel limits**, applied in the child before it executes anything:
   address space, CPU time, a zero file-write budget, and no core dumps. These
   hold even if the child stops responding to signals.
3. **A wall-clock deadline** in the parent, which kills the child's entire
   process group — so a fix that spawns threads or grandchildren still dies.

The child runs in a scratch directory with a scrubbed environment and Python's
isolated mode, so it cannot import from the caller's tree. Output is made
deterministic (the scratch path is replaced with a placeholder) because rewards
are computed from it.

The design is deliberately *permissive about legitimate code*. An early version
of this environment nullified `exec` in `builtins` to block code injection —
which also broke CPython's lazy import machinery, so any not-yet-loaded module
failed to import and the medium and hard tasks became unsolvable. The current
sandbox blocks imports by name at the source level, so a fix that legitimately
needs `hashlib`, `threading` or `super()` runs unimpeded. `tests/test_sandbox.py`
covers both halves: fifteen named escapes are refused, and the stdlib a real fix
needs still works.

## The reward

The dense per-turn reward is what GRPO optimises. A purely execution-based
reward (did the tests pass?) is almost always zero for a weak model and gives the
policy nothing to climb, so the total is decomposed into earned components and a
penalty term:

| Component | Max | Pays for |
| --- | ---: | --- |
| Format compliance | 0.10 | emitting the five required fields |
| Hypothesis quality | 0.20 | a specific, grounded, calibrated hypothesis |
| Localization | 0.15 | naming the function and line that broke |
| Fix quality | 0.35 | the fix passing the bug's test cases |
| Semantic similarity | 0.10 | resembling the canonical fix |
| Efficiency | 0.10 | solving while turns remain |
| Penalties | −0.55 | giving up, breaking passing tests, malformed output |

Three properties are load-bearing and are asserted by tests:

- **A perfect first-turn solve scores exactly `1.0`**, and the total is floored
  at `-0.5`. These two bounds define the advertised reward range.
- **Fix quality is the single largest lever.** No other component, and in
  particular not similarity to the reference fix, can substitute for actually
  passing the tests — that closes the obvious reward-hacking route of pasting
  canonical-looking code that does not run.
- **Confidence is calibrated.** A confident wrong hypothesis scores below a
  cautious wrong one.

At the episode level the per-turn rewards are discounted (0.9 per turn) so that
solving earlier is worth more than solving later, with a bonus if the bug was
ever solved.

## The curriculum

Training on the full bug distribution from step zero collapses the policy onto
the single easiest bug type. The curriculum introduces difficulty in stages: tier
1 only until step 150, tiers 1–2 until step 350, then all three tiers. The
`CurriculumEnvironment` samples only from the active tiers, and the trainer swaps
the pool at each boundary.

The dataset is 90 bugs — 40 tier 1, 30 tier 2, 20 tier 3 — each with a buggy
implementation, a reference fix, and input/expected-output test cases. Every
record is checked by execution: `agentdebugger validate` runs each reference fix
and each buggy implementation through the sandbox and asserts the fix passes all
its cases and the buggy code fails at least one. If that check is green, the
reported numbers measure something real.

## Training

The published run fine-tunes `Qwen2.5-Coder-3B-Instruct` with GRPO and LoRA
adapters. GRPO rather than PPO for a concrete reason: it scores a group of
sampled completions against each other instead of learning a value network, which
halves the memory per step — the difference between training and not training on
a 16GB T4. The trainer detects the GPU and scales batch size, gradient
accumulation, group size and LoRA rank accordingly; `num_generations` shrinks
first on small cards because it dominates GRPO's memory cost.

The reward a completion receives during training is exactly what the evaluator
computes for the same completion — both call `score_response` — so a training
reward curve and an evaluation number are directly comparable. This is asserted
by a test.

## Results

The reward curves from the published run are in [images/](images/):

- **Format compliance** reaches its ceiling within the first ~50 steps: the model
  learns the required response structure almost immediately.
- **Total reward** climbs steadily over the run, with a visible transient dip at
  the tier-2 curriculum boundary as the policy adapts to harder bugs — the
  signature of a curriculum that is doing its job rather than one that has
  plateaued.

Per-bug evaluation results for the trained adapter are published in
[../results/](../results/). Reproduce them with
`agentdebugger evaluate-curriculum --adapter <repo>`.

## Limitations

- **Single-file scope.** The environment operates on isolated functions, not
  multi-file repositories. Real bugs rarely present this cleanly; this is a
  training simplification, and the primary limit on real-world transfer.
- **The sandbox is a workstation-grade defence.** It safely runs the code an LLM
  actually produces on a developer machine or CI box. It is not a substitute for
  a container or VM when running deliberately adversarial code from an untrusted
  third party.
- **The hard-task grader is empirical.** It detects the race by stress testing
  rather than by proving thread safety, so it can only demonstrate the presence
  of a lost update, never its absence in general. The stress parameters are tuned
  so the buggy counter fails every run in practice.

## What would come next

- **Multi-file debugging**, with a language-server view of the codebase so the
  agent can trace across call stacks and imports.
- **Adversarial bug generation**, where a second model mutates bugs to exploit the
  current policy's weaknesses, turning the fixed curriculum into a self-sustaining
  one.
- **A PPO comparison** on this task, to put empirical numbers behind the
  memory-versus-stability trade-off that motivated the choice of GRPO.
