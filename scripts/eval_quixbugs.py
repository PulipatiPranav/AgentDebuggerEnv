#!/usr/bin/env python3
"""Evaluate a LoRA adapter on the QuixBugs dataset."""
import argparse
import json
from pathlib import Path
from agentdebugger.dataset.models import Bug
from agentdebugger.envs.curriculum_env import score_response
from agentdebugger.training.prompts import bug_to_messages
from agentdebugger.evaluation.curriculum import load_generator, TierResult, CurriculumReport

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--format", default="structured")
    args = parser.parse_args()

    quixbugs_file = Path("data/quixbugs/bugs.jsonl")
    if not quixbugs_file.exists():
        print(f"Error: {quixbugs_file} not found. Run scripts/build_quixbugs.py first.")
        return 1

    bugs = []
    for line in quixbugs_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            bugs.append(Bug.from_dict(json.loads(line)))

    print(f"Loaded {len(bugs)} QuixBugs.")

    generate, name = load_generator(args.base_model, args.adapter, format=args.format)
    
    records = []
    solved = 0
    total_reward = 0.0
    extraction_failures = 0

    for index, bug in enumerate(bugs, start=1):
        print(f"\r  quixbugs: {index}/{len(bugs)}", end="", flush=True)
        completion = generate(bug_to_messages(bug, format=args.format))
        outcome = score_response(bug, completion, format=args.format)

        solved += outcome.solved
        total_reward += outcome.reward.total
        extraction_failures += not outcome.extraction_ok
        records.append({
            "id": bug.id,
            "completion": completion,
            "solved": outcome.solved,
            "extraction_ok": outcome.extraction_ok,
            "reward": outcome.reward.as_dict()
        })

    print("\n")
    print(f"QuixBugs Solve Rate: {solved/len(bugs):.1%} ({solved}/{len(bugs)})")
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model": name,
        "format": args.format,
        "overall": {
            "total": len(bugs),
            "solved": solved,
            "solve_rate": solved / len(bugs),
            "extraction_failure_rate": extraction_failures / len(bugs)
        },
        "bugs": records
    }, indent=2))
    print(f"Results written to {args.output}")

if __name__ == "__main__":
    main()
