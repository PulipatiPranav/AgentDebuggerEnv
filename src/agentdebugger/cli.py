"""``agentdebugger`` — the command line entry point.

    agentdebugger episode --task hard        # watch one episode, no GPU or API key
    agentdebugger evaluate --model gpt-4o    # score a model on all three tasks
    agentdebugger validate                   # check every bug in the dataset
    agentdebugger serve                      # expose the environment over HTTP
    agentdebugger train --max-steps 500      # GRPO training run
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agentdebugger import __version__
from agentdebugger.agents import Agent, OracleAgent
from agentdebugger.config import TIERS
from agentdebugger.evaluation import evaluate_agent, run_episode
from agentdebugger.protocol import Action
from agentdebugger.render import bar, field, heading, signed, style, verdict
from agentdebugger.tasks import get_task, list_tasks


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 1
    return int(args.handler(args) or 0)


# ── commands ──────────────────────────────────────────────────────────────────


def _episode(args: argparse.Namespace) -> int:
    """Run one episode and narrate it."""
    agent = _make_agent(args)
    task = get_task(args.task)

    print()
    print(f"{heading('AgentDebuggerEnv')}  {style(f'task={task.task_id}  agent={agent.name}', 'dim')}")
    print()
    print(f"  {style(task.name, 'bold')}")
    print(field("", task.description, width=0, indent=2))
    print()

    result = run_episode(agent, args.task, on_step=_narrate_step)

    print()
    print(f"  {heading('episode')}")
    print(field("grader", f"{result.grader_score:.2f} / 1.00"))
    print(field("reward", f"{result.cumulative_reward:+.2f} cumulative"))
    print(field("attempts", f"{result.attempts_used} of {task.max_attempts}"))
    print(field("solved", verdict(result.solved)))
    print()
    return 0 if result.solved else 1


def _narrate_step(step: int, action: Action, result: Any) -> None:
    info = result.info
    print(f"  {style(f'step {step}', 'bold')}  {style(action.action_type, 'blue')}")

    if action.hypothesis:
        print(field("hypothesis", action.hypothesis))
    if action.action_type == "submit_fix" and "tests_passed" in info:
        detail = f"{bar(info['tests_passed'], info['tests_total'])}  ·  {info['execution_time_ms']}ms"
        if info.get("timed_out"):
            detail += style("  · timed out", "red")
        if info.get("blocked"):
            detail += style("  · blocked by sandbox policy", "red")
        print(field("sandbox", detail))
        print(field("solved", verdict(bool(info.get("solved")))))
    if action.action_type == "query_context":
        answer = info.get("query_result", "").splitlines()
        print(field("context", answer[0] if answer else ""))
    if info.get("error"):
        print(field("error", style(info["error"], "red")))
    if action.final_diagnosis:
        print(field("diagnosis", action.final_diagnosis))

    print(field("reward", signed(result.reward.step_reward)))
    print()


def _evaluate(args: argparse.Namespace) -> int:
    """Score an agent on every task."""
    agent = _make_agent(args)
    report = evaluate_agent(agent, args.tasks)

    print()
    print(f"  {heading(report.agent)}")
    print()
    for episode in report.episodes:
        print(
            f"  {episode.task_id:<8} score {episode.grader_score:.2f}  "
            f"{bar(episode.tests_passed, episode.tests_total)}  "
            f"attempts {episode.attempts_used}  {verdict(episode.solved, 'solved', 'unsolved')}"
        )
    print()
    print(f"  mean score  {style(f'{report.mean_score:.3f}', 'bold')}")
    print(f"  solve rate  {report.solve_rate:.0%}")
    print()

    _write_json(args.output, report.as_dict())
    return 0


def _evaluate_curriculum(args: argparse.Namespace) -> int:
    """Score a local model on the tiered bug dataset."""
    from agentdebugger.evaluation import evaluate_curriculum, load_generator

    generate, name = load_generator(args.base_model, args.adapter)

    def progress(done: int, total: int, bug: Any) -> None:
        print(f"\r  tier {bug.tier}: {done}/{total}", end="", flush=True)

    report = evaluate_curriculum(
        generate,
        name,
        tiers=args.tiers,
        limit=args.limit,
        on_bug=progress,
        split=args.split,
        format=args.format,
    )

    print("\n")
    print(f"  {heading(report.model)}   {style(f'format={report.format}', 'dim')}")
    print()
    for tier in report.tiers:
        print(
            f"  tier {tier.tier}   solve rate {tier.solve_rate:6.1%}  "
            f"({tier.solved}/{tier.total})   mean reward {tier.mean_reward:+.3f}  "
            f"extraction-fail {tier.extraction_failure_rate:.1%}"
        )
    print()
    print(f"  overall     {style(f'{report.solve_rate:.1%}', 'bold')} "
          f"({report.solved}/{report.total})   "
          f"extraction-fail {report.extraction_failure_rate:.1%}")
    print()

    _write_json(args.output, report.as_dict())
    return 0


def _validate(args: argparse.Namespace) -> int:
    """Check that every bug's reference fix passes and its buggy code fails."""
    from agentdebugger.dataset import validate_tiers

    print(f"\n  validating {len(args.tiers)} tier(s) in the sandbox...\n")
    report = validate_tiers(tuple(args.tiers))

    for failure in report.failures:
        print(f"  {style('FAIL', 'red')} tier{failure.tier} {failure.bug_id}")
        for problem in failure.problems:
            print(f"       {problem}")

    checked = report.total
    if report.ok:
        print(f"  {style('OK', 'green')}  all {checked} bugs are sound\n")
        return 0
    print(f"\n  {len(report.failures)} of {checked} bugs are unsound\n")
    return 1


def _tasks(args: argparse.Namespace) -> int:
    print()
    for task_id in list_tasks():
        task = get_task(task_id)
        budget = f"{task.tests_total} tests · {task.max_attempts} attempts · {task.max_steps} steps"
        print(f"  {style(task.task_id.ljust(8), 'bold')}{task.name}")
        print(f"          {style(budget, 'dim')}")
    print()
    return 0


def _serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("The server needs: pip install 'agentdebugger[serve]'", file=sys.stderr)
        return 1

    uvicorn.run("agentdebugger.serve.app:app", host=args.host, port=args.port)
    return 0


def _train(args: argparse.Namespace) -> int:
    from agentdebugger.training import TrainingConfig, train

    train(
        TrainingConfig(
            model=args.model,
            max_steps=args.max_steps,
            output_dir=args.output_dir,
            seed=args.seed,
            push_to_hub=args.push_to_hub,
            reward_config=args.reward_config,
            split=args.split,
            format=args.format,
            reward_workers=args.reward_workers,
        )
    )
    return 0


# ── plumbing ──────────────────────────────────────────────────────────────────


def _make_agent(args: argparse.Namespace) -> Agent:
    if args.agent == "oracle":
        return OracleAgent()

    from agentdebugger.agents.api import ApiAgent

    return ApiAgent(model=args.model, base_url=args.base_url)


def _write_json(path: str | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  written to {destination}\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentdebugger",
        description="A reinforcement learning environment for debugging Python.",
    )
    parser.add_argument("--version", action="version", version=f"agentdebugger {__version__}")
    parser.set_defaults(command=None)
    subparsers = parser.add_subparsers(dest="command")

    def add_agent_flags(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--agent",
            choices=("oracle", "api"),
            default="oracle",
            help="oracle submits the reference fix (no API key needed); "
            "api calls an OpenAI-compatible endpoint",
        )
        sub.add_argument("--model", help="model name, for --agent api (or set MODEL_NAME)")
        sub.add_argument("--base-url", help="OpenAI-compatible base URL (or set API_BASE_URL)")

    episode = subparsers.add_parser("episode", help="run and narrate a single episode")
    episode.add_argument("--task", choices=list_tasks(), default="easy")
    add_agent_flags(episode)
    episode.set_defaults(handler=_episode)

    evaluate = subparsers.add_parser("evaluate", help="score an agent on every task")
    evaluate.add_argument("--tasks", nargs="+", choices=list_tasks(), default=None)
    evaluate.add_argument("--output", help="write the report to this JSON file")
    add_agent_flags(evaluate)
    evaluate.set_defaults(handler=_evaluate)

    curriculum = subparsers.add_parser(
        "evaluate-curriculum", help="score a local model on the tiered bug dataset"
    )
    curriculum.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    curriculum.add_argument("--adapter", help="a LoRA adapter to load on top of the base model")
    curriculum.add_argument("--tiers", nargs="+", type=int, choices=TIERS, default=list(TIERS))
    curriculum.add_argument("--limit", type=int, help="only evaluate this many bugs per tier")
    curriculum.add_argument(
        "--split",
        choices=("all", "train", "heldout"),
        default="heldout",
        help="which dataset split to evaluate on (default: held-out, the only side to report)",
    )
    curriculum.add_argument(
        "--format",
        choices=("structured", "free_form"),
        default="structured",
        help="response format the prompt asks for and the parser expects (H1's independent variable)",
    )
    curriculum.add_argument("--output", help="write the report to this JSON file")
    curriculum.set_defaults(handler=_evaluate_curriculum)

    validate = subparsers.add_parser("validate", help="check the bug dataset is sound")
    validate.add_argument("--tiers", nargs="+", type=int, choices=TIERS, default=list(TIERS))
    validate.set_defaults(handler=_validate)

    tasks = subparsers.add_parser("tasks", help="list the available tasks")
    tasks.set_defaults(handler=_tasks)

    serve = subparsers.add_parser("serve", help="serve the environment over HTTP")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(handler=_serve)

    train = subparsers.add_parser("train", help="run GRPO training")
    train.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    train.add_argument("--max-steps", type=int, default=500)
    train.add_argument("--output-dir", default="./checkpoints")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument(
        "--reward-config",
        choices=("R0", "R1", "R2"),
        default="R0",
        help="R0 full (shipped), R1 terminal-only, R2 dense minus reasoning",
    )
    train.add_argument(
        "--split",
        choices=("all", "train", "heldout"),
        default="train",
        help="which dataset split to train on (default: train)",
    )
    train.add_argument(
        "--format",
        choices=("structured", "free_form"),
        default="structured",
        help="response format the prompt asks for and the parser expects (H1's independent variable)",
    )
    train.add_argument(
        "--reward-workers",
        type=int,
        default=1,
        help="score a group's completions in a process pool of this size "
        "(>1 only helps once the calibration run shows scoring dominates step time)",
    )
    train.add_argument("--push-to-hub", help="a HF repo to push the trained adapter to")
    train.set_defaults(handler=_train)

    return parser


if __name__ == "__main__":
    raise SystemExit(main())
