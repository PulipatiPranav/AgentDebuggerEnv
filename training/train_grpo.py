"""
AgentDebuggerEnv — GRPO Training Script
Model: Qwen2.5-Coder-7B-Instruct (4-bit quantized via Unsloth)
Algorithm: GRPO (Group Relative Policy Optimization) via HuggingFace TRL
GPU: HuggingFace ZeroGPU H200 (free) or paid HF Spaces A10G

Usage:
  # Local reward sanity-check (no GPU, no model loading):
  python training/train_grpo.py --test-local

  # Test run (Colab/GPU, 10 steps):
  python training/train_grpo.py --test

  # Full training run:
  python training/train_grpo.py

  # Resume from checkpoint:
  python training/train_grpo.py --resume ./checkpoints/checkpoint-400
"""

import os
import sys
import json
import argparse
import random
import subprocess
import tempfile
import shutil

# ── Parse args ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", help="Run 10 steps for testing (Colab/GPU)")
parser.add_argument("--test-local", action="store_true", dest="test_local",
                    help="Sanity-check reward function locally without any model or GPU")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
parser.add_argument("--max_steps", type=int, default=1000)
args = parser.parse_args()

# ── Install dependencies (for Colab/HF Spaces) ───────────────────────────────
# If running locally with venv, comment these out
if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("SPACE_ID"):
    os.system("pip install -q unsloth trl wandb datasets")

# ── GPU/training imports (skipped in --test-local mode) ───────────────────────
if not args.test_local:
    import torch
    import wandb
    from datasets import Dataset
    from unsloth import FastLanguageModel
    from trl import GRPOTrainer, GRPOConfig
    from transformers import TrainerCallback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server.reward_calculator import DebugRewardCalculator
from server.models import parse_agent_output

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
HF_REPO = "shashaank0707/AgentDebugger-trained"
MAX_STEPS = 10 if args.test else args.max_steps
CHECKPOINT_DIR = "./checkpoints"

# W&B — optional but strongly recommended for judging
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "") if not args.test_local else ""
if WANDB_API_KEY:
    wandb.init(
        project="AgentDebuggerEnv",
        name=f"grpo-qwen-7b-{'test' if args.test else 'full'}",
        config={
            "model": MODEL_NAME,
            "algorithm": "GRPO",
            "curriculum": "tier1->tier2->tier3",
            "max_steps": MAX_STEPS,
            "reward_components": ["format", "hypothesis", "localization", "fix", "semantic", "efficiency"],
            "paper_citations": ["Masud et al. 2026", "Ibrahim et al. 2024"],
        }
    )

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert Python debugger. You reason through bugs systematically.

You MUST respond in EXACTLY this format — no exceptions, no extra text:

OBSERVATION: [Specific observations about the code and error. Reference exact line numbers.]
HYPOTHESIS: [Your theory about the root cause. Must be at least 2 sentences. Reference specific variable names, operators, or logic.]
CONFIDENCE: [low | medium | high]
ACTION: [One of: inspect_lines | run_tests | propose_fix | request_context | give_up]
DETAIL: [For propose_fix: the complete corrected function code. For inspect_lines: line numbers. For others: specific details.]

Rules:
- Never omit any field
- HYPOTHESIS must explain WHY the bug causes the observed failure
- If proposing a fix, DETAIL must contain the complete function, not just the changed line
- Give up only if you have exhausted all reasonable hypotheses"""

# ── Load bugs ─────────────────────────────────────────────────────────────────
def load_bugs(tier: int) -> list[dict]:
    path = f"data/bugs_tier{tier}.jsonl"
    if not os.path.exists(path):
        print(f"WARNING: {path} not found. Run data/generate_bugs.py first.")
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def get_bugs_for_step(step: int) -> list[dict]:
    tier1 = load_bugs(1)
    if step < 300:
        return tier1
    elif step < 600:
        return tier1 + load_bugs(2)
    return tier1 + load_bugs(2) + load_bugs(3)

def bug_to_prompt(bug: dict) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Debug this Python function:\n\n```python\n{bug['buggy_code']}\n```\n\n"
        f"Initial failure: {bug.get('initial_error', 'Some tests are failing.')}\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def _run_fix(proposed_code: str, bug: dict) -> dict:
    """Safely run proposed fix with subprocess timeout."""
    test_cases = bug.get("test_cases", [])
    func_name = bug.get("function_name", "")
    if not proposed_code or not test_cases or not func_name:
        return {"passed": 0, "failed": 0, "total": len(test_cases), "newly_broken": 0}

    passed = 0
    for test in test_cases:
        inp = test["input"]
        args_str = ", ".join(repr(x) for x in inp) if isinstance(inp, (list, tuple)) else repr(inp)
        script = (
            f"{proposed_code}\n"
            f"try:\n"
            f"    r={func_name}({args_str})\n"
            f"    print('PASS' if r=={repr(test['expected_output'])} else 'FAIL')\n"
            f"except Exception as e:\n"
            f"    print(f'ERROR: {{e}}')\n"
        )
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(script)
                fname = f.name
            python = shutil.which("python3") or shutil.which("python") or sys.executable
            r = subprocess.run([python, fname], capture_output=True, text=True, timeout=5)
            os.unlink(fname)
            if "PASS" in r.stdout:
                passed += 1
        except Exception:
            pass

    return {"passed": passed, "failed": len(test_cases) - passed, "total": len(test_cases), "newly_broken": 0}

# ── Mock completions for --test-local ─────────────────────────────────────────
MOCK_GOOD = """
OBSERVATION: The loop condition on line 4 uses <= instead of 
HYPOTHESIS: This causes an off-by-one error because Python lists are 
0-indexed, so the last valid index is len(arr)-1 not len(arr)
CONFIDENCE: high
ACTION: propose_fix
DETAIL: def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left < right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
"""

MOCK_BAD = """
I think there might be a bug somewhere in the code.
Let me try fixing it.
"""

# ── --test-local: reward sanity-check without any model ───────────────────────
if args.test_local:
    print("=" * 60)
    print("LOCAL TEST MODE — no model loaded, testing reward function only")
    print("=" * 60)

    bugs = load_bugs(1)
    if not bugs:
        print("ERROR: No bugs found in data/bugs_tier1.jsonl. Run data/generate_bugs.py first.")
        sys.exit(1)

    bug = bugs[0]
    print(f"\nUsing bug: {bug.get('function_name', '?')} — {bug.get('bug_type', '?')}\n")

    calculator_local = DebugRewardCalculator()

    def _score(label: str, completion: str) -> float:
        try:
            agent_output = parse_agent_output(completion)
            test_results = {"passed": 0, "failed": 0, "total": 0, "newly_broken": 0}
            if agent_output.action == "propose_fix":
                test_results = _run_fix(agent_output.detail, bug)
            breakdown = calculator_local.compute_turn_reward(
                agent_output=agent_output,
                ground_truth={
                    "bug_function": bug.get("bug_location", {}).get("function", ""),
                    "bug_line": bug.get("bug_location", {}).get("line_start", -1),
                    "bug_type": bug.get("bug_type", ""),
                    "canonical_fix_code": bug.get("original_code", ""),
                },
                test_results=test_results,
                turn_number=0,
            )
            print(f"--- {label} reward breakdown ---")
            for field, value in breakdown.__dict__.items():
                print(f"  {field}: {value}")
            print(f"  TOTAL: {breakdown.total}\n")
            return breakdown.total
        except Exception as e:
            print(f"Reward error for {label}: {e}")
            return -0.3

    good_score = _score("MOCK_GOOD", MOCK_GOOD)
    bad_score = _score("MOCK_BAD", MOCK_BAD)

    print(f"MOCK_GOOD score: {good_score:.4f}")
    print(f"MOCK_BAD  score: {bad_score:.4f}")

    assert good_score > bad_score, (
        f"ASSERTION FAILED: MOCK_GOOD ({good_score:.4f}) should be > MOCK_BAD ({bad_score:.4f})"
    )
    print("\nLOCAL TEST PASSED")
    sys.exit(0)

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_NAME}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=4096,
    load_in_4bit=True,
    dtype=None,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
print(f"Trainable params: {model.num_parameters(only_trainable=True):,}")

# ── Reward function ───────────────────────────────────────────────────────────
calculator = DebugRewardCalculator()

def reward_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """
    GRPO reward function. Called on groups of completions for the same prompt.
    GRPO learns from RELATIVE differences within each group.
    """
    rewards = []
    bugs = kwargs.get("bug_metadata", [{}] * len(completions))

    for completion, bug in zip(completions, bugs):
        try:
            agent_output = parse_agent_output(completion)

            # Run fix if agent proposes one
            test_results = {"passed": 0, "failed": 0, "total": 0, "newly_broken": 0}
            if agent_output.action == "propose_fix" and bug:
                test_results = _run_fix(agent_output.detail, bug)

            breakdown = calculator.compute_turn_reward(
                agent_output=agent_output,
                ground_truth={
                    "bug_function": bug.get("bug_location", {}).get("function", ""),
                    "bug_line": bug.get("bug_location", {}).get("line_start", -1),
                    "bug_type": bug.get("bug_type", ""),
                    "canonical_fix_code": bug.get("original_code", ""),
                },
                test_results=test_results,
                turn_number=0,
            )

            if WANDB_API_KEY:
                wandb.log({k: v for k, v in breakdown.__dict__.items()})

            rewards.append(breakdown.total)

        except Exception as e:
            print(f"Reward error: {e}")
            rewards.append(-0.3)

    return rewards

# ── Baseline evaluation (run BEFORE training) ─────────────────────────────────
def run_baseline(n: int = 20) -> dict:
    print("\nRunning baseline evaluation on UNTRAINED model...")
    FastLanguageModel.for_inference(model)
    bugs = load_bugs(1)[:n]
    rewards = []
    solved = 0
    for bug in bugs:
        prompt = bug_to_prompt(bug)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=400, temperature=0.1, do_sample=False)
        completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        r = reward_fn([completion], [prompt], bug_metadata=[bug])
        rewards.append(r[0])
        if r[0] > 0.20:   # threshold: any positive structured response counts
            solved += 1

    result = {"solve_rate": solved / max(len(bugs), 1), "avg_reward": sum(rewards) / max(len(rewards), 1), "rewards": rewards}
    with open("baseline_results.json", "w") as f:
        json.dump(result, f)
    print(f"Baseline: solve_rate={result['solve_rate']:.1%}, avg_reward={result['avg_reward']:.3f}")
    if WANDB_API_KEY:
        wandb.log({"baseline/solve_rate": result["solve_rate"], "baseline/avg_reward": result["avg_reward"]})
    return result

baseline = run_baseline()
FastLanguageModel.for_training(model)

# ── Build initial dataset ─────────────────────────────────────────────────────
def make_dataset(step: int) -> Dataset:
    bugs = get_bugs_for_step(step)
    return Dataset.from_list([{"prompt": bug_to_prompt(b), "bug_metadata": b} for b in bugs])

# ── Training config ───────────────────────────────────────────────────────────
config = GRPOConfig(
    output_dir=CHECKPOINT_DIR,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=4,       # A100 80GB handles 4 (was 2)
    gradient_accumulation_steps=2,       # effective batch = 8 (same total, less accumulation lag)
    learning_rate=2e-5,                  # slightly higher lr for faster convergence
    lr_scheduler_type="cosine",
    warmup_steps=20 if args.test else 40,
    num_generations=8,                   # GRPO key: more rollouts = stronger learning signal (was 4)
    max_new_tokens=512,                  # longer responses = more complete fixes (was 400)
    temperature=0.9,                     # slightly higher temp = more diverse rollouts for GRPO
    logging_steps=5 if args.test else 5, # log every 5 steps for dense W&B curve
    save_steps=50 if args.test else 100,
    report_to="wandb" if WANDB_API_KEY else "none",
)

trainer = GRPOTrainer(
    model=model,
    args=config,
    train_dataset=make_dataset(0),
    reward_funcs=reward_fn,
    tokenizer=tokenizer,
)

# ── Curriculum callback ───────────────────────────────────────────────────────
class CurriculumCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step in [300, 600]:
            trainer.train_dataset = make_dataset(step)
            print(f"\nCurriculum advanced at step {step}!")
            if WANDB_API_KEY:
                wandb.log({"curriculum/step": step})

trainer.add_callback(CurriculumCallback())

# ── Train ─────────────────────────────────────────────────────────────────────
print(f"\nStarting GRPO training. Max steps: {MAX_STEPS}")
print(f"Baseline solve rate: {baseline['solve_rate']:.1%} — target: >60% after training")
trainer.train(resume_from_checkpoint=args.resume)

# ── Post-training evaluation ──────────────────────────────────────────────────
FastLanguageModel.for_inference(model)
bugs = load_bugs(1)[:20]
post_rewards = []
post_solved = 0
for bug in bugs:
    prompt = bug_to_prompt(bug)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=400, temperature=0.1, do_sample=False)
    completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    r = reward_fn([completion], [prompt], bug_metadata=[bug])
    post_rewards.append(r[0])
    if r[0] > 0.20:
        post_solved += 1

post_solve_rate = post_solved / max(len(bugs), 1)
print(f"\n{'='*60}")
print(f"RESULTS:")
print(f"Before training: {baseline['solve_rate']:.1%} solve rate")
print(f"After training:  {post_solve_rate:.1%} solve rate")
print(f"Improvement:     +{post_solve_rate - baseline['solve_rate']:.1%}")
print(f"{'='*60}")

if WANDB_API_KEY:
    wandb.log({"final/solve_rate": post_solve_rate, "final/improvement": post_solve_rate - baseline["solve_rate"]})
    wandb.finish()

# ── Save and push ─────────────────────────────────────────────────────────────
model.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN and not args.test:
    model.push_to_hub(HF_REPO, token=HF_TOKEN)
    tokenizer.push_to_hub(HF_REPO, token=HF_TOKEN)
    print(f"Pushed to https://huggingface.co/{HF_REPO}")
