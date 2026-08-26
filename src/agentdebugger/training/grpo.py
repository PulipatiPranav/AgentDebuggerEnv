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
from agentdebugger.training.prompts import PromptFormat, bug_to_messages

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
        return cls(2, 4, 2, 512, 8)  # T4, and anything smaller

def _completion_length_for(profile: HardwareProfile, format: PromptFormat) -> int:
    """Free-form responses run longer before ever emitting a fenced code block —
    they write full prose analysis (headers, numbered sections) ahead of the fix,
    where structured responses get to DETAIL's code block much faster. Diagnostic
    completions at the default 192-token budget consistently got cut off one
    sentence before the fix even started, with correct reasoning throughout —
    a token-budget problem, not a parsing or reasoning-quality problem. Give
    free-form ~1.8x the room so it has a real chance to finish.

    The floor (550) guards against small-VRAM profiles where the multiplier
    alone isn't enough: on a 4090 (192 * 1.8 = 345), B1 zero-shot completions
    ran 250-300+ tokens of prose before any code appeared, so 345 is still
    too tight. 550 is calibrated against the eval observation that successful
    free-form completions need ~400-600 tokens to complete reasoning + fix.
    """
    if format == "free_form":
        return max(int(profile.max_completion_length * 1.8), 550)
    return profile.max_completion_length

@dataclass(frozen=True)
class TrainingConfig:
    """Everything that defines a training run."""

    model: str = "Qwen/Qwen2.5-Coder-3B-Instruct"
    max_steps: int = 500
    learning_rate: float = 2e-5
    warmup_steps: int = 30
    temperature: float = 0.7
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


def _extract_completion_text(completion: Any) -> str:
    """Extract plain text string from completion, whether str or chat message dict."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and len(completion) > 0:
        last = completion[-1]
        if isinstance(last, dict) and "content" in last:
            return last["content"]
        if isinstance(last, str):
            return last
    if isinstance(completion, dict) and "content" in completion:
        return completion["content"]
    return str(completion)


def _score_one(args: tuple[Any, dict[str, Any], str, str]) -> dict[str, Any]:
    """Score exactly one completion. A module-level function so it can be sent
    to a :class:`~concurrent.futures.ProcessPoolExecutor` (closures cannot be
    pickled). Never raises: a crashed rollout must cost the policy something
    rather than take down a whole worker process mid-batch.
    """
    raw_completion, bug_dict, reward_config, response_format = args
    try:
        completion = _extract_completion_text(raw_completion)
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


def _grouped(prompts: list[Any], values: list[Any]) -> list[list[Any]]:
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

    def reward_function(completions: list[str], prompts: list[Any], **kwargs: Any) -> list[float]:
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
    prompts: list[Any],
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
            {"prompt": bug_to_messages(bug, format=format), "bug_metadata": json.dumps(bug.as_dict())}
            for bug in bugs
        ]
    )


def curriculum_stages(schedule: CurriculumSchedule, max_steps: int) -> list[tuple[int, int]]:
    """The ``[(start_step, end_step), ...]`` ranges a run of ``max_steps`` splits into.

    Pure and GPU-independent so it is unit-tested directly (see ``train()``'s
    docstring for *why* training must be split this way rather than swapping
    ``trainer.train_dataset`` mid-run).
    """
    boundaries = sorted(step for step in schedule.advances_at() if 0 < step < max_steps)
    starts = [0, *boundaries]
    ends = [*boundaries, max_steps]
    return list(zip(starts, ends, strict=True))


def _build_lora_model(model, profile: HardwareProfile, adapter_dir: str | None) -> Any:
    """Attach a fresh LoRA adapter, or reload one carried over from a prior curriculum stage."""
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    if adapter_dir is None:
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
    else:
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)
    model.enable_input_require_grads()
    # model.gradient_checkpointing_enable()
    return model


def train(config: TrainingConfig) -> None:
    """Run GRPO training end to end, one :class:`~trl.GRPOTrainer` per curriculum stage.

    **Why per-stage trainers, not one trainer with a dataset-swapping callback:**
    TRL/transformers builds the train dataloader's sampler exactly once per
    ``Trainer.train()`` call, and that sampler (``RepeatRandomSampler``) caches
    ``len(dataset)`` at *construction* time. A callback that reassigns
    ``trainer.train_dataset`` mid-run (the previous approach here) never reaches
    that sampler — the already-built dataloader keeps iterating over the original
    dataset object, so the bug pool silently never grows past tier 1 for the
    entire run. Starting a new, short-lived trainer at each curriculum boundary
    and carrying the LoRA adapter forward (`PeftModel.from_pretrained(...,
    is_trainable=True)`) is what actually makes the tier unlock real. Each stage
    already checkpoints to disk with a ``.stage_complete`` marker, so a run that
    dies partway through (e.g. a preemptible/spot GPU) can be restarted and will
    skip whatever stages already finished.
    """
    import shutil
    from pathlib import Path

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    # ── Compatibility shim ────────────────────────────────────────────────────
    # transformers 4.54+ changed Trainer._get_dataloader() to call the sampler
    # as `sampler_fn(dataset)`. GRPOTrainer._get_train_sampler in trl 0.16.x
    # only accepts `self`. This one-liner makes it accept (and ignore) the
    # extra positional arg so both old and new transformers work.
    _orig_sampler = GRPOTrainer._get_train_sampler
    GRPOTrainer._get_train_sampler = lambda self, dataset=None: _orig_sampler(self)

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
    # Force the EOS token to be the chat end token so generation terminates early.
    # Qwen's base eos_token is <|endoftext|>, but the instruct template ends turns with <|im_end|>.
    if "<|im_end|>" in tokenizer.vocab:
        tokenizer.eos_token = "<|im_end|>"
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"

    stages = curriculum_stages(config.schedule, config.max_steps)
    out_root = Path(config.output_dir)
    adapter_dir: str | None = None

    for stage_index, (stage_start, stage_end) in enumerate(stages):
        tiers = config.schedule.tiers_at(stage_start)
        stage_dir = out_root / f"_stage{stage_index}"
        done_marker = stage_dir / ".stage_complete"

        print(
            f"\n[curriculum] stage {stage_index}: steps {stage_start}-{stage_end} "
            f"({stage_end - stage_start} steps), tiers {tiers}",
            flush=True,
        )

        if done_marker.exists():
            print(f"[curriculum] stage {stage_index} already complete, reusing {stage_dir}", flush=True)
            adapter_dir = str(stage_dir)
            continue

        # transformers 4.57+ deprecated `torch_dtype=` in favour of `dtype=`;
        # older versions only accept `torch_dtype=`. Try the new name first.
        try:
            model = AutoModelForCausalLM.from_pretrained(
                config.model,
                dtype=dtype,
                attn_implementation="sdpa",
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                config.model,
                torch_dtype=dtype,
                attn_implementation="sdpa",
            )
        model = _build_lora_model(model, profile, adapter_dir)
        if stage_index == 0:
            print(f"Trainable parameters: {model.num_parameters(only_trainable=True):,}", flush=True)

        trainer = GRPOTrainer(
            model=model,
            args=GRPOConfig(
                output_dir=str(stage_dir),
                max_steps=stage_end - stage_start,
                per_device_train_batch_size=profile.batch_size,
                gradient_accumulation_steps=profile.gradient_accumulation_steps,
                num_generations=profile.num_generations,
                # max_completion_length=profile.max_completion_length,
                max_completion_length=_completion_length_for(profile, config.format),
                # Stop at the chat end token so the model can terminate early once it
                # learns the format. (Removed stop_strings as it's unsupported in trl<0.17;
                # Qwen naturally stops at EOS <|im_end|> anyway)
                learning_rate=config.learning_rate,
                lr_scheduler_type="cosine",
                # Only the first stage warms up; later stages continue from an
                # already-warm policy, so restarting the warmup would just spike
                # the LR right after a curriculum tier unlock.
                warmup_steps=config.warmup_steps if stage_index == 0 else 0,
                temperature=config.temperature,
                logging_steps=config.logging_steps,
                save_steps=config.save_steps,
                save_strategy="steps",
                seed=config.seed,
                report_to="wandb" if _wandb_active() else "none",
            ),
            train_dataset=build_dataset(stage_start, config.schedule, config.split, config.format),
            reward_funcs=make_reward_function(
                config.schedule, config.reward_config, config.format, config.reward_workers
            ),
            processing_class=tokenizer,
        )
        trainer.train()

        stage_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(stage_dir))
        done_marker.touch()
        adapter_dir = str(stage_dir)

        del trainer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # `adapter_dir` now holds the last stage's adapter; publish it as the run's
    # output so `--adapter <output_dir>` keeps working exactly as before. Skip
    # subdirectories (each stage's intermediate `checkpoint-*` saves live there
    # too) - only the adapter files saved by `trainer.save_model` belong at the
    # top level.
    assert adapter_dir is not None
    for item in Path(adapter_dir).iterdir():
        if item.is_file():
            shutil.copy2(item, out_root / item.name)
    tokenizer.save_pretrained(config.output_dir)
    print(f"Saved adapter to {config.output_dir}", flush=True)

    if config.push_to_hub:
        from peft import AutoPeftModelForCausalLM

        final_model = AutoPeftModelForCausalLM.from_pretrained(config.output_dir)
        final_model.push_to_hub(config.push_to_hub)
        tokenizer.push_to_hub(config.push_to_hub)
        print(f"Pushed to https://huggingface.co/{config.push_to_hub}", flush=True)


def _wandb_active() -> bool:
    import os

    return bool(os.environ.get("WANDB_API_KEY"))
