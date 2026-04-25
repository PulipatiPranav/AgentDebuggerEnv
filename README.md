---
title: AgentDebugger-Training 🧠
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: true
license: mit
---

# AgentDebuggerEnv 🐛

> **A live, iterative debugging environment for benchmarking genuine agentic reasoning in AI systems.**

[![HuggingFace Space](https://img.shields.io/badge/🤗%20Space-Live-yellow)](https://huggingface.co/spaces/shashaank0707/AgentDebugger-env)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-blue)](#openenv-api-compliance)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

*Submitted to the **Meta + PyTorch + HuggingFace OpenEnv Hackathon.***

---

## The Problem with Existing Code Benchmarks

Benchmarks like HumanEval, MBPP, and SWE-bench share a fundamental limitation: they are **one-shot**. A model reads a problem, generates code, and is scored on the final output. This measures code generation — not debugging ability.

Real software engineering is not one-shot. It is **iterative**. A developer reads failing tests, forms a hypothesis, submits a fix, reads the new error output, updates their theory, and repeats. No existing OpenEnv environment benchmarks this loop.

**AgentDebuggerEnv does.**

---

## How It's Different from SWE-bench

| Dimension | SWE-bench | AgentDebuggerEnv |
|---|---|---|
| Evaluation target | Final patch correctness | Full reasoning trajectory |
| Feedback to agent | None — single shot | Real `stdout/stderr` after every attempt |
| Reward signal | Binary end-of-episode | Dense — every step scored |
| What's measured | Code generation | Hypothesis formation + iterative reasoning |
| Hard task | Apply patch to existing issue | Must design a test to surface a hidden bug |
| Agent failure modes | Not tracked | 4 distinct measurable failure modes |

The iterative feedback loop is the core mechanic. Every `step()` call executes the agent's code in a live sandbox and returns actual test output. The agent must update its theory and try again — exactly like a real developer at a terminal.

---

## Baseline Performance

Evaluated using `gpt-4o` with zero-shot prompting. Each task run 5 times independently, scores averaged.

| Task | Difficulty | Mean Score | Std Dev | Solved % | Avg Attempts |
|---|---|---|---|---|---|
| Off-by-One Bug | 🟢 Easy | 0.85 | ±0.04 | 100% | 1.8 |
| Red Herring Auth Bug | 🟡 Medium | 0.50 | ±0.10 | 60% | 4.2 |
| Race Condition | 🔴 Hard | 0.18 | ±0.09 | 20% | 8.7 |
| **Overall Mean** | | **0.51** | | **60%** | |

The hard task is specifically designed so that frontier models fail most of the time. GPT-4o almost never spontaneously recognizes that a race condition can exist when all sequential tests pass — which is exactly the reasoning gap this environment is built to measure.

---

All four failure modes produce distinct, interpretable score components in the `breakdown` field of every `Reward` response:

*   **Red Herring Susceptibility**: Does the agent overtrust error messages (Medium Task symptom) or trace data flow to the root?
*   **Stagnation**: Does the agent repeat failed fixes? Prohibited by the `-0.05` stagnation penalty.
*   **Exploration/Exploitation**: Measures if agents query for context productively before attempting fixes.
*   **Test-Suite Overconfidence**: Detects if an agent fails to reason about concurrency when sequential tests pass (Hard Task).

---

## Task Suite

### 🟢 Task 1 — Easy: Off-by-One Bug

**Max attempts:** 5 | **Max steps:** 8 | **Tests:** 8

A binary search implementation with a single-character bug: the while loop uses `left < right` instead of `left <= right`. This causes the function to miss the target when it is the last element. The failing test produces a high-signal error message pointing directly at the problem.

**Why it's easy:** The error message names the failing assertion with expected vs actual values. Reading the while condition reveals the bug. 1–2 iterations expected.

**What the grader checks:** Did all 8 tests pass? Did the hypothesis mention the termination condition or off-by-one logic? Was it efficient?

---

### 🟡 Task 2 — Medium: Red Herring Authentication Bug

**Max attempts:** 7 | **Max steps:** 15 | **Tests:** 10 (6 pass, 4 fail on buggy code)

An authentication module with three interdependent functions: `hash_password`, `validate_password`, and `authenticate_user`. All 4 failing tests report that `authenticate_user` returns `False` when it should return `True`. But `authenticate_user` is completely correct. So is `validate_password`. The bug is in `hash_password`, which wraps the MD5 hex digest in `str(bytes(...))` — producing a `"b'...'"` prefix that makes the computed hash never match the stored hash.

**The red herring:** Every surface reading of the error points to `authenticate_user`. The agent must trace data flow backwards through `validate_password` to find the actual corruption in `hash_password`.

**Red herring detection in grader:** A hypothesis mentioning only `authenticate_user` scores 0.0 for hypothesis accuracy. Correctly identifying `hash_password` with supporting detail scores 1.0. GPT-4o follows the red herring ~40% of the time.

---

### 🔴 Task 3 — Hard: Concurrency Race Condition

**Max attempts:** 10 | **Max steps:** 25 | **Tests:** 8 (ALL 8 pass on the buggy code)

A `ConnectionCounter` class used in a web server to track active connections. It uses `threading.Lock` and appears correctly implemented. All 8 sequential unit tests pass. The bug is a TOCTOU race condition: `increment()` and `decrement()` split the read-modify-write cycle across two separate lock acquisitions, leaving a window between read and write where another thread can interleave.

```python
def increment(self):
    with self._lock:
        current = self.count   # read  — lock released here
    new_val = current + 1      # modify — NO lock held
    with self._lock:
        self.count = new_val   # write — race window
```

The agent must: recognize that 8/8 passing tests do not prove correctness for concurrent code, reason about thread interleaving, design a concurrent stress test that surfaces the race, fix the atomicity issue by collapsing read-modify-write into a single lock scope, and verify the fix survives a 1000-thread stress test.

**Hard task grader breakdown:**
- Sequential tests pass (agent submissions only): **0.40**
- 1000-thread concurrent stress test passes (run 5×, must pass >=4 for full credit): **0.30**
- Hypothesis accuracy (mentions "race condition", "atomic", "lock"): **0.20**
- Efficiency bonus (fixed within 5 attempts): **0.10**

---

## Reward Function Design

The reward function provides dense signal at every step so an RL agent can learn from every action — not just the final outcome.

### Step-Level Rewards

| Event | Reward | Reasoning |
|---|---|---|
| Fix increases tests passing | `+0.15 × (Δpassed / total)` | Scaled progress |
| Fix decreases tests passing | `-0.10 × (Δfailed / total)` | Regression penalty |
| Fix makes no change to passing count | `-0.05` | Stagnation penalty |
| All tests pass | `+0.50` | Major bonus on top of progress |
| Submitted code times out in sandbox | `-0.10` | Penalizes infinite loops |
| `submit_fix` without hypothesis field | `-0.10` | Hypothesis is required |
| First `query_context` use | `0.00` | Free |
| Subsequent `query_context` uses | `-0.05` each | Diminishing returns |
| Episode truncated at max_steps | `-0.20` | Penalizes indecision |

### Episode-Level Grader Score

```
grader_score = test_pass_ratio    × 0.60
             + efficiency_bonus   × 0.20
             + hypothesis_accuracy × 0.15
             + early_solve_bonus  × 0.05

test_pass_ratio    = agent_best_tests_passed / tests_total
                     (from agent submissions only — never the initial buggy code run)
efficiency_bonus   = max(0, (max_attempts - attempts_used) / max_attempts)
hypothesis_accuracy = fraction of hypotheses correctly identifying the bug
early_solve_bonus  = 0.05 if solved within ceil(max_attempts / 3) attempts
```

**Score floor design:** `test_pass_ratio` uses only the agent's submitted attempts — never the initial buggy code run. The medium buggy code passes 6/10 tests and the hard buggy code passes 8/8 tests sequentially. Without this design, a dummy agent that submits nothing would score 0.36 and 0.40 for free respectively. The grader recalculates from the `attempts` list to guarantee the score floor is 0.0.

---

## Security Sandbox

Every `submit_fix` action executes agent-generated Python code. All execution routes through `env/sandbox.py` — never via raw `exec()` anywhere in the codebase.

**Layer 1 — AST Import & Attribute Filtering:** Before execution, an AST walk detects blocked imports and prevents access to any attribute starting with an underscore (`_`). This blocks private member access and dunder escapes (like `__class__`).

**Layer 2 — Subprocess Isolation:** Code runs in a child subprocess with a stripped environment and no network access.

**Layer 3 — Hard Timeout:** Every execution killed after 10 seconds. Infinite loops in submitted code return `timed_out: True` and a `-0.10` step reward.

**Layer 4 — Memory Limit:** 256MB per execution.

**Threading exception:** The hard task requires `threading` to create and verify the race condition. The sandbox accepts `allow_threading=True` for that task only. All other tasks block threading entirely.

---

## Data Models

```python
class Observation(BaseModel):
    task_id: str                          # "easy" | "medium" | "hard"
    buggy_code: str                       # Original broken code
    test_suite: str                       # Full test file content
    current_code: str                     # Most recent submitted code
    current_error_output: str             # Sandbox stdout/stderr output
    tests_passed: int                
    attempts_remaining: int
    max_attempts: int
    done: bool
    score_estimate: float                 # Running grader estimate

class Action(BaseModel):
    action_type: str                      # "submit_fix" | "query_context" | "give_up"
    fixed_code: Optional[str]             # Complete corrected code
    hypothesis: Optional[str]             # Theory about the bug (required for submit)
    query_type: Optional[str]             # "function_signature" | "error_explanation" etc.

class Reward(BaseModel):
    step_reward: float                    # Dense signal: range -1.0 to +1.0
    cumulative_reward: float 
    grader_score: float                   # Official score (terminal step only)
    breakdown: Dict[str, float]           # Itemized components
```

---

## OpenEnv API Compliance

```yaml
name: agentdebugger-env
version: 1.0.0
domain: software_engineering
observation_type: structured
action_type: structured
reward_type: dense
episode_termination: action_or_step_limit
tasks:
  - {id: easy,   difficulty: easy,   max_steps: 8,  max_attempts: 5}
  - {id: medium, difficulty: medium, max_steps: 15, max_attempts: 7}
  - {id: hard,   difficulty: hard,   max_steps: 25, max_attempts: 10}
```

Application-level errors are returned in `info.error` inside the response body. Core evaluation endpoints are designed to avoid 4xx/5xx status codes for agent-level mistakes, ensuring the evaluation flow is never interrupted by network-level exceptions.

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API overview — lists all endpoints and tasks |
| `/health` | GET | Health check — always HTTP 200 |
| `/tasks` | GET | All tasks with metadata |
| `/reset` | POST | Start episode. Body: `{"task_id": "easy"}` |
| `/step` | POST | Submit one action |
| `/state` | GET | Full internal episode state |

---

## Installation & Usage

### Local Setup

```bash
git clone https://github.com/shasshaank/AgentDebuggerEnv
cd AgentDebuggerEnv
pip install -r requirements.txt

# Start the environment server
uvicorn env.server:app --reload --port 8000

# Verification: Run the pre-submission validator
python validator.py

# Verify it's running
curl http://localhost:8000/health
```

### Docker

```bash
docker build -t agentdebugger-env .
docker run -p 8000:8000 agentdebugger-env
```

### Running the Baseline Inference Script

```bash
git clone https://github.com/shasshaank/AgentDebuggerEnv
cd AgentDebuggerEnv
pip install -r requirements.txt

# Start the environment server
uvicorn env.server:app --reload --port 8000

# Verify it's running
curl http://localhost:8000/health
# {"status": "ok", "environment": "agentdebugger-env", "version": "1.0.0"}

# Run baseline inference
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o"
export HF_TOKEN="your_api_key"
export ENV_BASE_URL="http://localhost:8000"
python inference.py
```

Using Meta-Llama via HuggingFace (Recommended):

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-70B-Instruct"
export HF_TOKEN="your_huggingface_token"
export ENV_BASE_URL="http://localhost:8000"
python inference.py
```

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `API_BASE_URL` | LLM API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier | `meta-llama/Llama-3.1-70B-Instruct` |
| `HF_TOKEN` | Hugging Face Token (Read) | — |
| `ENV_BASE_URL` | Environment server address | `http://localhost:8000` |

---

## Project Structure

```
AgentDebuggerEnv/
├── inference.py                  # Baseline script (root — hackathon requirement)
├── env/
│   ├── environment.py            # Core OpenEnv: reset(), step(), state()
│   ├── models.py                 # Pydantic v2 Observation, Action, Reward
│   ├── sandbox.py                # AST-based sandboxed code execution
│   ├── server.py                 # FastAPI: /reset /step /state /health /tasks
│   ├── tasks/
│   │   ├── task_easy.py          # Off-by-one in binary search
│   │   ├── task_medium.py        # Red herring authentication bug
│   │   └── task_hard.py          # Concurrency race condition
│   └── graders/
│       ├── grader_easy.py        # Test pass + efficiency scoring
│       ├── grader_medium.py      # Red herring detection + score floor fix
│       └── grader_hard.py        # Sequential + concurrent stress test
├── openenv.yaml
├── Dockerfile
├── requirements.txt
└── uv.lock                       # Reproducible dependency resolution
```

---

## Design Decisions

**Why is hypothesis mandatory?** Requiring a hypothesis on every `submit_fix` prevents degenerate strategies of submitting random code until something passes. It also enables the grader to score `hypothesis_accuracy` independently from `test_pass_ratio` — measuring reasoning quality separately from outcome quality.

**Why recalculate `test_pass_ratio` from the attempts list?** The medium buggy code passes 6/10 tests and the hard buggy code passes 8/8 tests sequentially. If the grader used the environment's `best_tests_passed` (which includes the initial buggy code run at reset), a dummy agent that submits nothing would score 0.36 and 0.40 for free. Recalculating from the `attempts` list guarantees the score floor is 0.0.

**Why run the concurrent stress test 5 times?** Race conditions are non-deterministic. A partial fix that narrows the race window may pass once by luck. Requiring 4 of 5 runs to pass provides a robust statistical threshold that filters out lucky partial fixes while allowing for minor runner jitter. Passing 2 of 5 gives 0.15 — partial credit for progress.

**Why not use pytest directly?** Using pytest as the test runner makes output parsing dependent on pytest's version and output format. The environment uses a lightweight custom test runner embedded as a Python string, producing a consistent `"N passed, M failed"` format that `_parse_tests_passed()` can reliably parse across all platforms and environments.

**Why `query_context` costs reward after the first use?** Free unlimited context queries would allow agents to trivially read all available information before attempting any fix. The cost structure forces agents to make strategic decisions about when additional information is worth spending a step on — which is a core part of real debugging under time pressure.

---

## License & Attribution

**License:** MIT — see [LICENSE](LICENSE)

**Author:** Shashaank | GitHub: [@shasshaank](https://github.com/shasshaank) | HF: [@shashaank0707](https://huggingface.co/shashaank0707)

**Live Environment:** https://huggingface.co/spaces/shashaank0707/AgentDebugger-env

**Submitted to:** Meta + PyTorch + HuggingFace OpenEnv Hackathon

---

## Submission Integrity

- **Commit SHA:** `5c507c313ff2c209d7b770af6f08cf6ed6ab1568`
- **Last Verified Sync:** 2026-04-09
- **Platform Match:** GitHub and HF Space are in sync at this HEAD