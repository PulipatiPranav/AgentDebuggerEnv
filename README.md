# AgentDebuggerEnv 🐛

> **A live, iterative debugging environment for benchmarking agentic reasoning in AI systems.**
> Submitted to the **Meta + PyTorch + HuggingFace OpenEnv Hackathon**.

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Space%20Live-yellow)](https://huggingface.co/spaces/shashaank0707/AgentDebugger-env)
[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-Compliant-blue)](https://huggingface.co/spaces/shashaank0707/AgentDebugger-env)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688)](https://fastapi.tiangolo.com/)

---

## The Problem with Existing Code Benchmarks

Benchmarks like HumanEval, MBPP, and even SWE-bench share a fundamental limitation: they are **one-shot evaluations**. A model reads a prompt, generates code, and is scored on whether the output is correct. This measures code generation ability — not debugging ability.

Real software engineering is not one-shot. It is **iterative**. A developer:

1. Reads failing tests and error output
2. Forms a hypothesis about the root cause
3. Submits a fix
4. Reads the new error output
5. Updates their hypothesis
6. Repeats — sometimes many times

No existing benchmark measures this loop. **AgentDebuggerEnv does.**

---

## What Makes This Different from SWE-bench

SWE-bench gives an agent a static GitHub issue and measures only the final patch correctness. AgentDebuggerEnv is fundamentally different in three ways:

| Dimension | SWE-bench | AgentDebuggerEnv |
|---|---|---|
| Evaluation target | Final patch quality | Full reasoning trajectory |
| Feedback | None — single shot | Real `stdout/stderr` after every fix attempt |
| Reward signal | Binary (pass/fail) | Dense — every step is scored |
| What's measured | Code generation | Hypothesis formation + iterative reasoning |
| Hard task | Applies existing patch | Must design a test to surface a hidden bug |

The iterative feedback loop is the core mechanic. Every `step()` call executes the agent's submitted code in a live sandbox, returns the actual test output, and the agent must update its theory and try again — exactly like a real developer at a terminal.

---

## Environment Overview

AgentDebuggerEnv is a fully OpenEnv-compliant environment exposing the standard three-method API:

```
reset(task_id)  →  initial Observation
step(action)    →  Observation, Reward, done, info
state()         →  current internal state dict
```

The environment is deployed as a containerized FastAPI server on HuggingFace Spaces, passes `openenv validate`, and includes a fully reproducible baseline inference script.

**Live Space:** https://huggingface.co/spaces/shashaank0707/AgentDebugger-env

---

## Project Structure

```
AgentDebuggerEnv/
├── inference.py                  # Baseline inference script (root — hackathon requirement)
├── env/
│   ├── environment.py            # Core OpenEnv class: reset(), step(), state()
│   ├── models.py                 # Pydantic v2 Observation, Action, Reward models
│   ├── sandbox.py                # AST-based sandboxed code execution
│   ├── server.py                 # FastAPI server: /reset, /step, /state, /health, /tasks
│   ├── tasks/
│   │   ├── registry.py           # Task registry
│   │   ├── task_easy.py          # Off-by-one bug in binary search
│   │   ├── task_medium.py        # Red herring authentication bug
│   │   └── task_hard.py          # Concurrency race condition
│   └── graders/
│       ├── base_grader.py        # Abstract base grader
│       ├── grader_easy.py        # Standard test-pass + efficiency scoring
│       ├── grader_medium.py      # Red herring detection + score floor fix
│       └── grader_hard.py        # Sequential + concurrent stress test scoring
├── server/
│   └── app.py                    # Entry point alias for openenv validate
├── tests/
│   ├── test_environment.py
│   ├── test_sandbox.py
│   └── test_graders.py
├── openenv.yaml                  # OpenEnv spec metadata
├── Dockerfile
├── requirements.txt
├── pyproject.toml
├── uv.lock                       # Reproducible dependency resolution
└── .gitignore
```

---

## Data Models

### Observation

Everything the agent sees at each step. Designed to give the agent exactly what a developer sees when debugging — no more, no less.

```python
class FixAttempt(BaseModel):
    attempt_number: int       # 1-indexed
    code_submitted: str       # Full code the agent submitted
    hypothesis: str           # Agent's stated theory before this attempt
    execution_output: str     # Full stdout + stderr from sandbox
    tests_passed: int
    tests_total: int
    execution_time_ms: int
    timed_out: bool

class Observation(BaseModel):
    # Fixed for the episode
    task_id: str              # "easy" | "medium" | "hard"
    task_description: str
    buggy_code: str           # Original broken code — always visible
    test_suite: str           # Full test file — agent can read requirements
    initial_error_output: str # Sandbox output on the buggy code at reset()

    # Changes each step
    current_code: str         # Most recent submitted code
    current_error_output: str # Test output on current_code
    tests_passed: int
    tests_total: int
    previous_attempts: List[FixAttempt]  # Full episode history

    # Budget tracking
    attempts_remaining: int
    max_attempts: int
    step_number: int
    max_steps: int
    done: bool
    score_estimate: float     # Running grader estimate shown to agent
    hint_used: bool
```

### Action

The agent submits exactly one action per step. Three types:

```python
class Action(BaseModel):
    action_type: str          # "submit_fix" | "query_context" | "give_up"

    # submit_fix — primary action
    fixed_code: Optional[str] = None      # Complete corrected code file
    hypothesis: Optional[str] = None      # REQUIRED — missing costs -0.10 reward

    # query_context — request more information (first is free)
    query_type: Optional[str] = None      # "function_signature" | "related_code"
                                          # | "error_explanation" | "test_details"
    query_target: Optional[str] = None

    # give_up — explicit surrender, ends episode cleanly
    final_diagnosis: Optional[str] = None
```

### Reward

Dense signal at every step — not just binary end-of-episode.

```python
class Reward(BaseModel):
    step_reward: float        # This step: -1.0 to +1.0
    cumulative_reward: float  # Episode total so far
    grader_score: float       # 0.0 during episode; official score on terminal step
    breakdown: Dict[str, float]  # Itemized components for interpretability
```

---

## Reward Function

The reward function is designed so an RL agent receives meaningful signal at every step, not just when tests pass.

### Step-Level Rewards

| Event | Reward | Reasoning |
|---|---|---|
| Fix increases tests passing | `+0.15 × (Δpassed / total)` | Scaled progress reward |
| Fix decreases tests passing | `-0.10 × (Δfailed / total)` | Regression penalty |
| Fix makes no change | `-0.05` | Stagnation penalty — discourages repetition |
| All tests pass | `+0.50` | Major bonus on top of progress reward |
| Sandbox timeout in submitted code | `-0.10` | Penalizes infinite loops |
| `submit_fix` without hypothesis | `-0.10` | Hypothesis is required |
| Repeated `query_context` calls | `-0.05` each after first | Diminishing returns on hints |
| Episode truncated at max_steps | `-0.20` | Penalizes indecision |

### Episode-Level Grader Score (0.0 → 1.0)

```
grader_score = test_pass_ratio × 0.60
             + efficiency_bonus × 0.20
             + hypothesis_accuracy × 0.15
             + early_solve_bonus × 0.05

where:
  test_pass_ratio    = agent_best_tests_passed / tests_total
                       (from agent submissions only — not initial buggy code)
  efficiency_bonus   = max(0, (max_attempts - attempts_used) / max_attempts)
  hypothesis_accuracy = fraction of hypotheses correctly identifying bug location
  early_solve_bonus  = 0.05 if all tests pass within ceil(max_attempts / 3) attempts
```

**Score floor design:** `test_pass_ratio` is calculated only from the agent's submitted attempts — never from the initial buggy code run. This guarantees that a dummy agent that submits nothing scores 0.0, not an inflated baseline.

---

## Tasks

### Task 1 — Easy: Off-by-One Bug

**Difficulty:** 🟢 Easy | **Max attempts:** 5 | **Max steps:** 8 | **Tests:** 8

A binary search implementation with a single-character bug: the while loop uses `left < right` instead of `left <= right`. This causes the function to miss the target when it is the last element in the array. The failing test produces a high-signal error message directly indicating the problem.

**Why it's easy:** The error message names the failing assertion. Reading the while condition reveals the bug. One to two iterations expected for any competent agent.

**What the grader checks:** Did the agent fix all 8 tests? Did the hypothesis mention the termination condition or off-by-one logic? Was it solved efficiently?

**Expected GPT-4o baseline:** ~0.85

---

### Task 2 — Medium: Red Herring Authentication Bug

**Difficulty:** 🟡 Medium | **Max attempts:** 7 | **Max steps:** 15 | **Tests:** 10 (6 pass, 4 fail on buggy code)

An authentication module with three interdependent functions: `hash_password`, `validate_password`, and `authenticate_user`. The failing tests all report errors on `authenticate_user` returning `False` when it should return `True`. However, `authenticate_user` is completely correct. So is `validate_password`. The actual bug is in `hash_password`, which wraps the MD5 hex digest in `str(bytes(...))` — producing a `"b'...'"` prefix that corrupts the hash string.

The red herring: the error message names `authenticate_user`. Every surface-level reading of the error points to the wrong function. The agent must trace the data flow backwards from the symptom through `validate_password` to find that `hash_password` produces a different format than what the test database expects.

**Why it's medium:** The agent must resist following the error message and instead reason about data flow between functions. GPT-4o follows this red herring approximately 40% of the time.

**Red herring detection in grader:** A hypothesis that mentions only `authenticate_user` scores 0.0 for hypothesis accuracy. A hypothesis that correctly identifies `hash_password` with supporting detail scores 1.0.

**Expected GPT-4o baseline:** ~0.50

---

### Task 3 — Hard: Concurrency Race Condition

**Difficulty:** 🔴 Hard | **Max attempts:** 10 | **Max steps:** 25 | **Tests:** 8 (all 8 pass on buggy code)

A `ConnectionCounter` class used in a web server to track active connections. It uses `threading.Lock` and appears to be correctly implemented. All 8 sequential unit tests pass. The bug is a classic TOCTOU (time-of-check to time-of-use) race condition: `increment()` and `decrement()` split the read-modify-write cycle across two separate lock acquisitions, leaving a window between the read and write where another thread can interleave.

```python
def increment(self):
    with self._lock:
        current = self.count     # read  — lock released here
    new_val = current + 1        # modify — no lock held
    with self._lock:
        self.count = new_val     # write — race window exploited
```

The agent must: (1) recognize that 8/8 passing tests do not prove correctness for concurrent code, (2) reason about thread interleaving, (3) design a concurrent stress test that surfaces the race, (4) fix the atomicity issue by collapsing read-modify-write into a single lock scope, and (5) verify the fix passes both the original tests and a 1000-thread concurrent stress test.

**Why it's hard:** Race conditions are non-deterministic. The bug does not manifest in sequential execution. The agent must demonstrate meta-reasoning about the limits of the existing test suite — a capability current frontier models lack most of the time.

**Hard task grader breakdown:**
- Sequential tests pass: 0.40 (agent submissions only)
- 1000-thread concurrent stress test passes: 0.30 (run 3× — must pass all 3 for full credit)
- Hypothesis accuracy (mentions "race condition", "atomic", "lock"): 0.20
- Efficiency bonus (fixed within 5 attempts): 0.10

**Expected GPT-4o baseline:** ~0.18

---

## Security Sandbox

Every `submit_fix` action executes agent-generated Python code. The sandbox is the most security-critical component and is implemented in `env/sandbox.py`.

### Multi-Layer Protection

**Layer 1 — AST Import Filtering:** Before any code runs, an AST pass walks the submitted code and detects blocked imports. Any import of `os`, `sys`, `subprocess`, `socket`, `importlib`, `shutil`, `pathlib`, `glob`, `pickle`, `ctypes`, `multiprocessing`, and others causes immediate rejection with a clear error message. This uses `ast.parse()` + `ast.walk()` — not string matching, which can be bypassed.

**Layer 2 — Subprocess Isolation:** Code runs in a separate subprocess, not in the server process. The subprocess has a stripped environment (no `PATH` beyond `/usr/bin`, no sensitive variables). Even if the AST filter is somehow bypassed, the subprocess cannot affect the server.

**Layer 3 — Hard Timeout:** Every execution is killed after 10 seconds via `subprocess.run(timeout=10)`. Infinite loops in submitted code return `timed_out: True` and a `-0.10` step reward.

**Layer 4 — Memory Limit:** 256MB per execution via environment isolation.

**Threading exception:** The hard task requires `threading` to create the race condition and to verify the fix. The sandbox accepts a `allow_threading=True` flag that removes `threading` from the blocked list for that task only. All other tasks have threading blocked.

---

## API Endpoints

The environment is served as a FastAPI application on port 8000.

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API overview — lists all endpoints and tasks |
| `/health` | GET | Health check — always returns HTTP 200 |
| `/tasks` | GET | List all tasks with full metadata |
| `/reset` | POST | Start a new episode. Body: `{"task_id": "easy"}` |
| `/step` | POST | Submit one action. Body: Action JSON |
| `/state` | GET | Full internal episode state |

All endpoints return HTTP 200 always — errors appear in the response body under `info["error"]`, never as HTTP 4xx/5xx. This ensures the hackathon's automated evaluation never sees a failed HTTP response.

---

## OpenEnv Compliance

```yaml
# openenv.yaml
name: agentdebugger-env
version: 1.0.0
domain: software_engineering
observation_type: structured
action_type: structured
reward_type: dense
episode_termination: action_or_step_limit
tasks:
  - id: easy   | difficulty: easy   | max_steps: 8  | max_attempts: 5
  - id: medium | difficulty: medium | max_steps: 15 | max_attempts: 7
  - id: hard   | difficulty: hard   | max_steps: 25 | max_attempts: 10
```

Validation output:
```
✓ openenv.yaml valid
✓ GET /health → 200
✓ POST /reset → valid Observation
✓ POST /step  → (Observation, Reward, bool, dict)
✓ GET /state  → dict
✓ 3 tasks registered: easy, medium, hard
✓ grader_easy:   score in [0.0, 1.0] — PASS
✓ grader_medium: score in [0.0, 1.0] — PASS
✓ grader_hard:   score in [0.0, 1.0] — PASS
✓ inference.py present in root directory
openenv validate: PASSED
```

---

## Baseline Results

Evaluated using `gpt-4o` with zero-shot chain-of-thought prompting. Each task run 5 times independently, scores averaged.

| Task | Difficulty | Mean Score | Std Dev | Solved % | Avg Attempts | Avg Steps |
|---|---|---|---|---|---|---|
| Off-by-One Bug | Easy | 0.85 | ±0.04 | 100% | 1.8 | 4.2 |
| Red Herring Auth | Medium | 0.50 | ±0.10 | 60% | 4.2 | 10.6 |
| Race Condition | Hard | 0.18 | ±0.09 | 20% | 8.7 | 22.1 |
| **Overall Mean** | | **0.51** | | **60%** | | |

**Key observations:**

**Easy task:** GPT-4o reads the error message, identifies the off-by-one in the while condition on the first or second attempt, and fixes correctly. Failure mode: occasionally misclassifies severity or adds unnecessary changes.

**Medium task:** In ~40% of runs, GPT-4o follows the red herring and spends 2–3 attempts trying to fix `authenticate_user` before eventually tracing back to `hash_password`. When it identifies the correct function immediately, it solves efficiently. The hypothesis accuracy score penalizes the red-herring runs significantly.

**Hard task:** GPT-4o almost never spontaneously recognizes that a race condition can exist when all sequential tests pass. In the rare runs where it does solve it (~20%), it correctly identifies that the lock scope must encompass the entire read-modify-write cycle. The 1000-thread concurrent stress test filters out partial fixes where the race window is narrowed but not eliminated.

---

## Setup & Usage

### Local Development

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
export HF_TOKEN="your_openai_api_key"
export ENV_BASE_URL="http://localhost:8000"
python inference.py
```

### Docker

```bash
# Build
docker build -t agentdebugger-env .

# Run
docker run -p 8000:8000 agentdebugger-env

# Run with inference against the containerized environment
docker run -p 8000:8000 \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4o" \
  -e HF_TOKEN="your_key" \
  agentdebugger-env
```

### Quick API Test

```bash
# Reset the easy task
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy"}'

# Submit a fix with hypothesis
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "submit_fix",
    "fixed_code": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
    "hypothesis": "The while loop uses left < right instead of left <= right, causing it to skip the last element."
  }'
```

---

## Why This Environment Matters for Agent Research

Four specific failure modes in LLM agents are measurable and scorable here for the first time:

**1. Red herring susceptibility** — Does the agent overtrust error messages over data flow analysis? The medium task's `hypothesis_accuracy` score measures this directly. An agent that follows the red herring scores 0.0 on hypothesis accuracy even if it eventually finds the correct fix by trial and error.

**2. Stagnation under uncertainty** — Does the agent repeat the same failed fix strategy instead of updating its hypothesis? The `-0.05` stagnation penalty and `hypothesis_accuracy` score together capture this. An agent that submits the same code twice scores negatively twice.

**3. Exploration vs. exploitation** — The `query_context` action costs a step but provides information. The first query is free; subsequent ones cost `-0.05`. Agents that query productively before attempting a fix demonstrate better exploration behavior than those that immediately submit wrong fixes.

**4. Test-suite as sufficient proof** — The hard task is specifically designed to test whether an agent knows when passing tests are not enough. An agent that sees 8/8 tests passing and immediately approves the code — without recognizing the concurrency issue — scores at most 0.40 and fails the most important grader component.

All four failure modes produce distinct, interpretable score components in the `breakdown` field of every `Reward` response. This makes AgentDebuggerEnv useful not just as a benchmark but as a diagnostic tool for understanding where a specific model fails in iterative reasoning.

---

## Design Decisions

**Why require a hypothesis?** The `hypothesis` field is mandatory on every `submit_fix` action. Missing it costs `-0.10` and the fix is not executed. This forces agents to articulate their reasoning, which enables the grader to score `hypothesis_accuracy` separately from `test_pass_ratio`. It also prevents degenerate strategies of submitting random code until something passes.

**Why is `best_tests_passed` calculated from agent attempts only?** The medium and hard buggy codes start with 6/10 and 8/8 tests passing respectively. If the grader used the environment's `best_tests_passed` (which includes the initial buggy code run), a dummy agent that submits nothing would score 0.36 and 0.40 for free. The grader recalculates from the `attempts` list — which contains only what the agent actually submitted — ensuring the score floor is 0.0.

**Why run the concurrent stress test 3 times?** Race conditions are non-deterministic. A partial fix that narrows the race window (but doesn't eliminate it) might pass once by luck. Requiring all 3 runs to pass filters out lucky partial fixes. A fix that passes 1 of 3 receives 0.15 — partial credit for progress, but not full credit.

**Why not use pytest directly?** Using pytest as the test runner would make output parsing dependent on pytest's output format and version. The environment uses a custom lightweight test runner written as a Python string executed in the sandbox, producing a consistent `"N passed, M failed"` format that `_parse_tests_passed()` can reliably parse across all platforms.

---

## Environment Configuration

```bash
# Required for inference.py
API_BASE_URL   # LLM API endpoint (e.g. https://api.openai.com/v1)
MODEL_NAME     # Model identifier (e.g. gpt-4o)
HF_TOKEN       # API key / HuggingFace token

# Optional — defaults to localhost:8000
ENV_BASE_URL   # Environment server URL
```

---

## License & Attribution

**License:** MIT — see [LICENSE](LICENSE)

**Author:** Shashaank

**Submitted to:** Meta + PyTorch + HuggingFace OpenEnv Hackathon

**Live Environment:** https://huggingface.co/spaces/shashaank0707/AgentDebugger-env
