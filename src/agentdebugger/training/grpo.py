"""GRPO training against the curriculum environment.

GRPO rather than PPO for one concrete reason: it scores a *group* of sampled
completions against each other instead of learning a value network, which halves
the memory needed per step. On a 16GB T4 that is the difference between training
and not training.

The reward for a completion is exactly what :func:`score_response` returns — the
same function the evaluator calls — so a reward curve and an eval number are
directly comparable.

Optional dependency: ``pip install 'agentdebugger[train]'``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from agentdebugger.config import DEFAULT_CURRICULUM, CurriculumSchedule
from agentdebugger.dataset import Bug, load_bugs
from agentdebugger.envs.curriculum_env import score_response
from agentdebugger.rewards.turn import TurnRewardCalculator
from agentdebugger.training.prompts import PromptFormat, bug_to_prompt

#: Reward assigned when scoring a completion raises. Scoring runs untrusted model
#: output, so it can fail in ways the reward function cannot anticipate; a
#: crashed rollout must cost the policy something rather than halt training.
SCORING_FAILURE_REWARD = -0.3

#: Two rewards in the same GRPO group closer than this count as tied. Rewards
#: are rounded to 4 decimals (RewardBreakdown), so anything below that
#: resolution is noise, not signal.
DEGENERATE_GROUP_EPSILON = 1e-6


@dataclass(frozen=True)
class HardwareProfile:
    """Batch geometry for a GPU tier.

    GRPO's memory cost is dominated by ``num_generations`` — the group it
    compares completions within — so that is what shrinks first on small cards.
    """

    batch_size: int
    gradient_accumulation_steps: int
    num_generations: int
    max_completion_length: int
    lora_rank: int

    @classmethod
    def for_vram(cls, vram_gb: float) -> HardwareProfile:
        """Pick a profile that fits in ``vram_gb`` of device memory."""
        if vram_gb >= 70:  # A100 80GB, H100
            return cls(8, 1, 8, 256, 16)
        if vram_gb >= 40:  # A100 40GB
            return cls(4, 2, 4, 256, 16)
        if vram_gb >= 20:  # A10, 3090, 4090
            return cls(2, 4, 2, 192, 8)
        return cls(2, 4, 2, 160, 8)  # T4, and anything smaller


@dataclass(frozen=True)
class TrainingConfig:
    """Everything that defines a training run."""

    model: str = "Qwen/Qwen2.5-Coder-3B-Instruct"
    max_steps: int = 500
    learning_rate: float = 2e-5
    warmup_steps: int = 30
    temperature: float = 0.9
    output_dir: str = "./checkpoints"
    save_steps: int = 25
    logging_steps: int = 5
    seed: int = 0
    schedule: CurriculumSchedule = DEFAULT_CURRICULUM
    push_to_hub: str | None = None
    #: Reward configuration: R0 (full), R1 (terminal), or R2 (no-reasoning).
    reward_config: str = "R0"
    #: Which dataset split to train on. Training uses the train side only; the
    #: held-out side is reserved for evaluation, so a solve rate measures learning
    #: rather than memorisation.
    split: str = "train"
    #: Response format the prompt asks for and the parser expects (H1's
    #: independent variable). ``"structured"`` (default) or ``"free_form"``.
    format: PromptFormat = "structured"
    #: Reward scoring runs a sandboxed subprocess per completion, serially by
    #: default. If scoring dominates step time (research_plan.md §4.6, Risk #6
    #: — measure with the calibration run first), set this above 1 to score a
    #: group's completions in a process pool instead.
    reward_workers: int = 1


def _score_one(args: tuple[str, dict[str, Any], str, str]) -> dict[str, Any]:
    """Score exactly one completion. A module-level function so it can be sent
    to a :class:`~concurrent.futures.ProcessPoolExecutor` (closures cannot be
    pickled). Never raises: a crashed rollout must cost the policy something
    rather than take down a whole worker process mid-batch.
    """
    completion, bug_dict, reward_config, response_format = args
    try:
        calculator = TurnRewardCalculator.from_name(reward_config)
        bug = Bug.from_dict(bug_dict)
        outcome = score_response(bug, completion, calculator=calculator, format=response_format)
        return {
            "reward": outcome.reward.total,
            "breakdown": outcome.reward.as_dict(),
            "solved": outcome.solved,
            "extraction_ok": outcome.extraction_ok,
            "error": None,
        }
    except Exception as exc:  # scoring untrusted model output; must never crash a worker
        return {
            "reward": SCORING_FAILURE_REWARD,
            "breakdown": None,
            "solved": False,
            "extraction_ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _grouped(prompts: list[str], values: list[Any]) -> list[list[Any]]:
    """Split ``values`` into runs of contiguous identical ``prompts``.

    GRPO's reward function is called once per generation batch, in which TRL
    lays out each unique prompt's ``num_generations`` sampled completions
    contiguously — that contiguous run *is* the group the policy gradient is
    computed relative to. Grouping any other way would not match what GRPO
    actually normalises against.
    """
    groups: list[list[Any]] = []
    current_prompt: object = object()  # sentinel that cannot equal a real prompt
    for prompt, value in zip(prompts, values, strict=True):
        if prompt != current_prompt:
            groups.append([])
            current_prompt = prompt
        groups[-1].append(value)
    return groups


def make_reward_function(
    schedule: CurriculumSchedule = DEFAULT_CURRICULUM,
    reward_config: str = "R0",
    format: PromptFormat = "structured",
    reward_workers: int = 1,
):
    """Build the TRL reward function for a given reward configuration.

    TRL passes the dataset columns through as keyword arguments, so the bug each
    completion was generated from arrives in ``bug_metadata``. Beyond the reward
    list TRL requires, this also emits three diagnostics W&B needs
    (research_plan.md §2, §4.6):

    * the **degenerate-group fraction** — H2's proposed mechanism: the share of
      sampled groups whose completions all received the same reward, so the
      group-relative advantage is exactly zero and the step contributes no
      gradient;
    * **per-component reward means** and the **reward-vs-solve-rate gap** — a
      widening gap between mean reward and solve rate is the reward-hacking
      signature (Risk register, H2 threats);
    * **scoring throughput** (seconds spent scoring this batch, per completion)
      — Risk #6: if this dominates step time, ``reward_workers`` should go up
      before renting a bigger GPU.
    """
    executor = None
    if reward_workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        executor = ProcessPoolExecutor(max_workers=reward_workers)

    def reward_function(completions: list[str], prompts: list[str], **kwargs: Any) -> list[float]:
        raw_bugs = kwargs.get("bug_metadata") or [None] * len(completions)
        started = time.perf_counter()

        rewards: list[float] = [SCORING_FAILURE_REWARD] * len(completions)
        breakdowns: list[dict[str, float] | None] = [None] * len(completions)
        solved: list[bool] = [False] * len(completions)
        extraction_ok: list[bool] = [False] * len(completions)

        scorable = [i for i, raw in enumerate(raw_bugs) if raw is not None]
        tasks = [
            (
                completions[i],
                json.loads(raw_bugs[i]) if isinstance(raw_bugs[i], str) else raw_bugs[i],
                reward_config,
                format,
            )
            for i in scorable
        ]
        mapper = executor.map if executor is not None else map
        for i, result in zip(scorable, mapper(_score_one, tasks), strict=True):
            rewards[i] = result["reward"]
            breakdowns[i] = result["breakdown"]
            solved[i] = result["solved"]
            extraction_ok[i] = result["extraction_ok"]
            if result["error"]:
                print(f"[reward] scoring failed: {result['error']}", flush=True)

        _log_batch_diagnostics(
            prompts=prompts,
            rewards=rewards,
            breakdowns=breakdowns,
            solved=solved,
            extraction_ok=extraction_ok,
            elapsed_seconds=time.perf_counter() - started,
        )
        return rewards

    return reward_function


def _log_batch_diagnostics(
    prompts: list[str],
    rewards: list[float],
    breakdowns: list[dict[str, float] | None],
    solved: list[bool],
    extraction_ok: list[bool],
    elapsed_seconds: float,
) -> None:
    """Compute and emit the H2-mediator / reward-hacking / throughput diagnostics.

    A no-op unless W&B is active, so evaluation and tests (which call the
    reward function directly, without a trainer) never pay for or require it.
    """
    if not _wandb_active():
        return
    import wandb

    groups = _grouped(prompts, rewards)
    degenerate = sum(1 for group in groups if max(group) - min(group) < DEGENERATE_GROUP_EPSILON)

    scored_breakdowns = [b for b in breakdowns if b is not None]
    component_means = {}
    if scored_breakdowns:
        for component in scored_breakdowns[0]:
            if component == "total":
                continue
            component_means[f"reward/{component}"] = sum(
                b[component] for b in scored_breakdowns
            ) / len(scored_breakdowns)

    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    solve_rate = sum(solved) / len(solved) if solved else 0.0
    extraction_failure_rate = 1.0 - (sum(extraction_ok) / len(extraction_ok) if extraction_ok else 1.0)

    wandb.log(
        {
            "group/degenerate_fraction": degenerate / len(groups) if groups else 0.0,
            "group/count": len(groups),
            "reward/mean": mean_reward,
            "reward/solve_rate": solve_rate,
            "reward/solve_rate_gap": mean_reward - solve_rate,
            "reward/extraction_failure_rate": extraction_failure_rate,
            "timing/reward_seconds": elapsed_seconds,
            "timing/reward_seconds_per_completion": (
                elapsed_seconds / len(rewards) if rewards else 0.0
            ),
            **component_means,
        }
    )


def build_dataset(
    step: int,
    schedule: CurriculumSchedule = DEFAULT_CURRICULUM,
    split: str = "train",
    format: PromptFormat = "structured",
):
    """The bug pool for ``step`` and ``split``, as a HF dataset of prompts."""
    from datasets import Dataset

    bugs = load_bugs(schedule.tiers_at(step), split=split)
    return Dataset.from_list(
        [
            {"prompt": bug_to_prompt(bug, format=format), "bug_metadata": json.dumps(bug.as_dict())}
            for bug in bugs
        ]
    )


def train(config: TrainingConfig) -> None:
    """Run GRPO training end to end."""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    vram_gb = (
        torch.cuda.get_device_properties(0).total_memory / 1e9
        if torch.cuda.is_available()
        else 0.0
    )
    profile = HardwareProfile.for_vram(vram_gb)
    # bfloat16 needs Ampere (compute capability 8.0+); older cards must use fp16.
    ampere_or_newer = torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8
    dtype = torch.bfloat16 if ampere_or_newer else torch.float16

    print(f"VRAM {vram_gb:.0f}GB -> {profile}, dtype={dtype}", flush=True)
    print(
        f"reward config {config.reward_config}, format '{config.format}', "
        f"training on split '{config.split}', reward_workers={config.reward_workers}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(config.model, device_map="auto", dtype=dtype)
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=profile.lora_rank,
            lora_alpha=profile.lora_rank * 2,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        ),
    )
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    print(f"Trainable parameters: {model.num_parameters(only_trainable=True):,}", flush=True)

    trainer = GRPOTrainer(
        model=model,
        args=GRPOConfig(
            output_dir=config.output_dir,
            max_steps=config.max_steps,
            per_device_train_batch_size=profile.batch_size,
            gradient_accumulation_steps=profile.gradient_accumulation_steps,
            num_generations=profile.num_generations,
            max_completion_length=profile.max_completion_length,
            learning_rate=config.learning_rate,
            lr_scheduler_type="cosine",
            warmup_steps=config.warmup_steps,
            temperature=config.temperature,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            save_strategy="steps",
            seed=config.seed,
            report_to="wandb" if _wandb_active() else "none",
        ),
        train_dataset=build_dataset(0, config.schedule, config.split, config.format),
        reward_funcs=make_reward_function(
            config.schedule, config.reward_config, config.format, config.reward_workers
        ),
        processing_class=tokenizer,
    )

    class CurriculumCallback(TrainerCallback):
        """Swap the bug pool when the schedule says the next tier unlocks."""

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step in config.schedule.advances_at():
                tiers = config.schedule.tiers_at(state.global_step)
                trainer.train_dataset = build_dataset(
                    state.global_step, config.schedule, config.split, config.format
                )
                print(f"\n[curriculum] step {state.global_step}: tiers {tiers}", flush=True)

    trainer.add_callback(CurriculumCallback())
    trainer.train()

    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"Saved adapter to {config.output_dir}", flush=True)

    if config.push_to_hub:
        model.push_to_hub(config.push_to_hub)
        tokenizer.push_to_hub(config.push_to_hub)
        print(f"Pushed to https://huggingface.co/{config.push_to_hub}", flush=True)


def _wandb_active() -> bool:
    import os

    return bool(os.environ.get("WANDB_API_KEY"))
