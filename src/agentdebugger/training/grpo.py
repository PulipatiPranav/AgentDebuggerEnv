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
from dataclasses import dataclass
from typing import Any

from agentdebugger.config import DEFAULT_CURRICULUM, CurriculumSchedule
from agentdebugger.dataset import Bug, load_bugs
from agentdebugger.envs.curriculum_env import score_response
from agentdebugger.training.prompts import bug_to_prompt

#: Reward assigned when scoring a completion raises. Scoring runs untrusted model
#: output, so it can fail in ways the reward function cannot anticipate; a
#: crashed rollout must cost the policy something rather than halt training.
SCORING_FAILURE_REWARD = -0.3


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


def make_reward_function(schedule: CurriculumSchedule = DEFAULT_CURRICULUM):
    """Build the TRL reward function.

    TRL passes the dataset columns through as keyword arguments, so the bug each
    completion was generated from arrives in ``bug_metadata``.
    """

    def reward_function(completions: list[str], prompts: list[str], **kwargs: Any) -> list[float]:
        raw_bugs = kwargs.get("bug_metadata") or [None] * len(completions)
        rewards = []

        for completion, raw in zip(completions, raw_bugs, strict=False):
            if raw is None:
                rewards.append(SCORING_FAILURE_REWARD)
                continue
            try:
                bug = Bug.from_dict(json.loads(raw) if isinstance(raw, str) else raw)
                rewards.append(score_response(bug, completion).reward.total)
            except Exception as exc:
                print(f"[reward] scoring failed: {type(exc).__name__}: {exc}", flush=True)
                rewards.append(SCORING_FAILURE_REWARD)

        return rewards

    return reward_function


def build_dataset(step: int, schedule: CurriculumSchedule = DEFAULT_CURRICULUM):
    """The bug pool for ``step``, as a HF dataset of prompts."""
    from datasets import Dataset

    bugs = load_bugs(schedule.tiers_at(step))
    return Dataset.from_list(
        [
            {"prompt": bug_to_prompt(bug), "bug_metadata": json.dumps(bug.as_dict())}
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
        train_dataset=build_dataset(0, config.schedule),
        reward_funcs=make_reward_function(config.schedule),
        processing_class=tokenizer,
    )

    class CurriculumCallback(TrainerCallback):
        """Swap the bug pool when the schedule says the next tier unlocks."""

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step in config.schedule.advances_at():
                tiers = config.schedule.tiers_at(state.global_step)
                trainer.train_dataset = build_dataset(state.global_step, config.schedule)
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
