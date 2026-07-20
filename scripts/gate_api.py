"""Go/no-go gate: how well does a hosted model already solve the bug set?

This answers the single question that decides whether the whole project is worth
running (research_plan.md §4.2, "is the environment merely easy?"): if a cheap,
strong, hosted model already solves the held-out bugs at a very high rate, then
the benchmark is trivial and training a small model on it teaches us nothing
interesting. If it sits somewhere in the middle, there is a real gap for RL to
close and the study is worth doing.

It scores the model with the *same* function GRPO training uses
(``score_response``), via the *same* evaluation harness (``evaluate_curriculum``),
so the number it prints means exactly what a training/eval solve rate means.

Reading the result:
    >90% solved  -> bugs are too easy; make them harder before training anything.
    50-75%       -> ideal; a real gap for training to close.
    <30%         -> possibly too hard / format-fighting; inspect a few completions.

Usage:
    # OpenAI (default):
    export OPENAI_API_KEY=sk-...
    python scripts/gate_api.py --model gpt-4o-mini

    # Any OpenAI-compatible endpoint (e.g. the Hugging Face router):
    export HF_TOKEN=hf_...
    python scripts/gate_api.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --base-url https://router.huggingface.co/v1

    # Verify the harness end-to-end with no API key and no network:
    python scripts/gate_api.py --self-test
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from agentdebugger.config import TIERS
from agentdebugger.dataset import Bug, load_bugs
from agentdebugger.envs.curriculum_env import score_response
from agentdebugger.training.prompts import bug_to_prompt

_CHATML_BLOCK = re.compile(
    r"<\|im_start\|>(system|user|assistant)\n(.*?)(?:<\|im_end\|>|\Z)",
    re.DOTALL,
)


def chatml_to_messages(prompt: str) -> list[dict[str, str]]:
    """Turn the ChatML string ``bug_to_prompt`` builds into chat messages.

    The trailing empty ``assistant`` block is the generation point, not content,
    so it is dropped. If nothing parses, the whole prompt is sent as one user
    message rather than failing.
    """
    messages = [
        {"role": role, "content": content.strip()}
        for role, content in _CHATML_BLOCK.findall(prompt)
        if content.strip()
    ]
    return messages or [{"role": "user", "content": prompt}]


def make_openai_generator(model: str, base_url: str | None, temperature: float, max_tokens: int):
    """Return a ``generate(prompt) -> completion`` backed by an OpenAI-compatible API.

    Works against OpenAI, the Hugging Face router, or Gemini's OpenAI-compatible
    endpoint. Retries rate limits and transient errors with exponential backoff,
    which free tiers (Gemini especially) will trigger.
    """
    import random
    import time

    from openai import (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        OpenAI,
        RateLimitError,
    )

    api_key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("HF_TOKEN")
    )
    if not api_key:
        sys.exit("Set OPENAI_API_KEY (or GEMINI_API_KEY / HF_TOKEN) first.")

    client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(prompt: str) -> str:
        max_attempts = 8
        for attempt in range(max_attempts):
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=chatml_to_messages(prompt),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=90.0,
                )
                return completion.choices[0].message.content or ""
            except (
                RateLimitError,
                APIConnectionError,
                APITimeoutError,
                InternalServerError,  # 503 "high demand" / 500, transient
            ) as exc:
                if attempt == max_attempts - 1:
                    raise
                delay = min(2**attempt + random.random(), 30.0)
                print(f"\n  [api] {type(exc).__name__}; retrying in {delay:.1f}s", flush=True)
                time.sleep(delay)
        return ""

    return generate


def load_dataset_bugs(
    bugs_dir: str | None, split: str, tiers: list[int], limit: int | None = None
) -> list[Bug]:
    """Load the bugs to score: the packaged v1 set, a v2 directory + split, or a
    flat single-file set such as ``data/quixbugs`` (Publication-Strategy §1.6).

    A flat set (one ``bugs.jsonl``, no ``bugs_tier{N}.jsonl``/``split.json``) is
    not tiered and is never split — it exists purely as a secondary,
    out-of-distribution eval, so ``--tiers``/``--split`` do not apply to it.
    ``limit`` caps the number of bugs *per tier* (or, for a flat set, in total),
    for a quick representative sample when an API's free tier will not allow the
    full set in one day.
    """
    if bugs_dir is None:
        bugs = list(load_bugs(tiers))
    elif (Path(bugs_dir) / "bugs.jsonl").exists():
        bugs = [
            Bug.from_dict(json.loads(line))
            for line in (Path(bugs_dir) / "bugs.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return bugs[:limit] if limit is not None else bugs
    else:
        directory = Path(bugs_dir)
        allowed: set[str] | None = None
        if split != "all":
            split_file = json.loads((directory / "split.json").read_text(encoding="utf-8"))
            allowed = set(split_file[split])

        bugs = []
        for tier in tiers:
            path = directory / f"bugs_tier{tier}.jsonl"
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if allowed is None or record["id"] in allowed:
                    bugs.append(Bug.from_dict(record))

    if limit is not None:
        seen: dict[int, int] = {}
        capped = []
        for bug in bugs:
            if seen.get(bug.tier, 0) < limit:
                capped.append(bug)
                seen[bug.tier] = seen.get(bug.tier, 0) + 1
        bugs = capped
    return bugs


def score_bugs(
    generate, name: str, bugs: list[Bug], on_bug=None, sleep: float = 0.0, format: str = "structured"
):
    """Score ``generate`` on ``bugs``, aggregating solve rate per tier.

    Mirrors ``evaluation.evaluate_curriculum`` but over an explicit bug list, so
    it can score a v2 held-out subset the packaged loader does not know about.
    ``sleep`` paces requests to stay under per-minute rate limits. ``format``
    must match the prompt format ``generate`` was actually shown (B0 uses
    ``"structured"``, B1 uses ``"free_form"``; research_plan.md's zero-shot
    baselines).
    """
    import time
    from collections import defaultdict

    per_tier: dict[int, dict[str, float]] = defaultdict(
        lambda: {"total": 0, "solved": 0, "reward": 0.0, "extraction_failures": 0}
    )
    records: list[dict[str, Any]] = []
    for index, bug in enumerate(bugs, start=1):
        completion = generate(bug_to_prompt(bug, format=format))
        outcome = score_response(bug, completion, format=format)
        stats = per_tier[bug.tier]
        stats["total"] += 1
        stats["solved"] += int(outcome.solved)
        stats["reward"] += outcome.reward.total
        stats["extraction_failures"] += int(not outcome.extraction_ok)
        records.append(
            {
                "id": bug.id,
                "tier": bug.tier,
                "solved": outcome.solved,
                "action": outcome.output.action,
                "extraction_ok": outcome.extraction_ok,
                "tests": outcome.tests.as_dict(),
                "reward": outcome.reward.as_dict(),
                "completion": completion,
            }
        )
        if on_bug is not None:
            on_bug(index, len(bugs), bug)
        if sleep and index < len(bugs):
            time.sleep(sleep)
    return per_tier, records


def make_oracle_generator(bugs: list[Bug], format: str = "structured"):
    """A generator that returns the reference fix in the required format.

    Used by ``--self-test`` to prove the harness scores a known-good answer as
    solved, with no API key and no network.
    """
    by_prompt = {bug_to_prompt(bug, format=format): bug for bug in bugs}

    def generate(prompt: str) -> str:
        bug = by_prompt[prompt]
        if format == "free_form":
            return (
                f"The function {bug.function_name} fails on its tests; the initial error was "
                f"{bug.initial_error}. The buggy line diverges from the intended logic, which is "
                "why the observed test cases fail; replacing the function body with the correct "
                "implementation restores the expected outputs for every case.\n\n"
                f"```python\n{bug.original_code}\n```"
            )
        return (
            f"OBSERVATION: The function {bug.function_name} fails on its tests; "
            f"the initial error was {bug.initial_error}.\n"
            "HYPOTHESIS: The buggy line diverges from the intended logic, which is "
            "why the observed test cases fail; replacing the function body with the "
            "correct implementation restores the expected outputs for every case.\n"
            "CONFIDENCE: high\n"
            "ACTION: propose_fix\n"
            f"DETAIL: {bug.original_code}"
        )

    return generate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-4o-mini", help="model name to evaluate")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    parser.add_argument("--tiers", nargs="+", type=int, choices=TIERS, default=list(TIERS))
    parser.add_argument(
        "--bugs-dir",
        default=None,
        help="score a v2 dataset directory (e.g. data/v2) instead of the packaged v1 set",
    )
    parser.add_argument(
        "--split",
        choices=("all", "train", "heldout"),
        default="all",
        help="with --bugs-dir, restrict to a split from that directory's split.json",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="0.0 = deterministic")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, help="cap bugs per tier (quick sample for tight free tiers)")
    parser.add_argument("--sleep", type=float, default=0.0, help="seconds to wait between requests")
    parser.add_argument(
        "--format",
        choices=("structured", "free_form"),
        default="structured",
        help="prompt/parser format (B0 baseline: structured; B1 baseline: free_form)",
    )
    parser.add_argument("--output", help="write the full JSON report here")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="score the reference fix locally (no API key/network) to check the harness",
    )
    args = parser.parse_args(argv)

    bugs = load_dataset_bugs(args.bugs_dir, args.split, args.tiers, args.limit)
    source = args.bugs_dir or "packaged v1"
    scope = f"{source} [{args.split}]" if args.bugs_dir else source

    if args.self_test:
        generate, name = make_oracle_generator(bugs, format=args.format), "oracle (self-test)"
    else:
        generate = make_openai_generator(
            args.model, args.base_url, args.temperature, args.max_tokens
        )
        name = args.model

    def progress(done: int, total: int, bug) -> None:
        print(f"\r  scoring {name} on {scope} [{args.format}]: {done}/{total} bugs", end="", flush=True)

    per_tier, records = score_bugs(
        generate, name, bugs, on_bug=progress, sleep=args.sleep, format=args.format
    )

    total = sum(s["total"] for s in per_tier.values())
    solved = sum(s["solved"] for s in per_tier.values())
    extraction_failures = sum(s["extraction_failures"] for s in per_tier.values())
    overall = solved / total if total else 0.0
    extraction_failure_rate = extraction_failures / total if total else 0.0

    print("\n")
    print(f"  Gate result for: {name}   ({scope}, {total} bugs, format={args.format})")
    print("  " + "-" * 48)
    for tier in sorted(per_tier):
        stats = per_tier[tier]
        rate = stats["solved"] / stats["total"] if stats["total"] else 0.0
        mean_reward = stats["reward"] / stats["total"] if stats["total"] else 0.0
        fail_rate = stats["extraction_failures"] / stats["total"] if stats["total"] else 0.0
        print(
            f"  tier {tier}   solve rate {rate:6.1%}  "
            f"({stats['solved']}/{stats['total']})   mean reward {mean_reward:+.3f}  "
            f"extraction-fail {fail_rate:.1%}"
        )
    print("  " + "-" * 48)
    print(f"  overall     {overall:6.1%}  ({solved}/{total})   extraction-fail {extraction_failure_rate:.1%}")
    print()

    if overall > 0.90:
        verdict = "TOO EASY - make the bugs harder before training (see the header)."
    elif overall < 0.30:
        verdict = "SURPRISINGLY HARD - inspect a few completions for format-fighting."
    else:
        verdict = "GOOD - a real gap for RL to close. Proceed."
    print(f"  Verdict: {verdict}\n")

    if args.output:
        report = {
            "model": name,
            "scope": scope,
            "format": args.format,
            "overall": {
                "total": total,
                "solved": solved,
                "solve_rate": round(overall, 4),
                "extraction_failure_rate": round(extraction_failure_rate, 4),
            },
            "tiers": {
                str(tier): {
                    "total": per_tier[tier]["total"],
                    "solved": per_tier[tier]["solved"],
                    "solve_rate": round(
                        per_tier[tier]["solved"] / per_tier[tier]["total"], 4
                    ),
                    "extraction_failure_rate": round(
                        per_tier[tier]["extraction_failures"] / per_tier[tier]["total"], 4
                    ),
                }
                for tier in sorted(per_tier)
            },
            "completions": records,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  Full report written to {args.output}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


