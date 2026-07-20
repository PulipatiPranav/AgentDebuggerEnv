"""Evaluate a local causal LM on the curriculum bug set.

One structured response per bug, scored by the same path GRPO trains on, so the
numbers reported here mean the same thing as the numbers in the training curves.

``transformers`` and ``torch`` are optional dependencies: install with
``pip install 'agentdebugger[train]'``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from agentdebugger.config import TIERS
from agentdebugger.dataset import Bug, load_tier
from agentdebugger.envs.curriculum_env import score_response
from agentdebugger.training.prompts import PromptFormat, bug_to_prompt

#: Anything that turns a prompt into a completion.
Generate = Callable[[str], str]


class _Progress(Protocol):
    def __call__(self, done: int, total: int, bug: Bug) -> None: ...


@dataclass(frozen=True)
class TierResult:
    """How a model did on one difficulty tier."""

    tier: int
    total: int
    solved: int
    mean_reward: float
    #: Fraction of responses the scorer could not get a usable answer out of at
    #: all — format-failure rate for a structured arm, extraction-failure rate
    #: for a free-form one. Comparable across formats; see
    #: :attr:`agentdebugger.envs.curriculum_env.TurnOutcome.extraction_ok` and
    #: research_plan.md Threat #8.
    extraction_failure_rate: float = 0.0
    bugs: tuple[dict[str, Any], ...] = field(default=())

    @property
    def solve_rate(self) -> float:
        return self.solved / self.total if self.total else 0.0


@dataclass(frozen=True)
class CurriculumReport:
    """A model's results across the tiers it was evaluated on."""

    model: str
    tiers: tuple[TierResult, ...]
    #: The response format used to produce this report (``"structured"`` or
    #: ``"free_form"``), recorded so a report file is self-describing.
    format: str = "structured"

    @property
    def total(self) -> int:
        return sum(tier.total for tier in self.tiers)

    @property
    def solved(self) -> int:
        return sum(tier.solved for tier in self.tiers)

    @property
    def solve_rate(self) -> float:
        return self.solved / self.total if self.total else 0.0

    @property
    def extraction_failure_rate(self) -> float:
        if not self.total:
            return 0.0
        return sum(tier.extraction_failure_rate * tier.total for tier in self.tiers) / self.total

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "format": self.format,
            "overall": {
                "total": self.total,
                "solved": self.solved,
                "solve_rate": round(self.solve_rate, 4),
                "extraction_failure_rate": round(self.extraction_failure_rate, 4),
            },
            "tiers": {
                f"tier{tier.tier}": {
                    "total": tier.total,
                    "solved": tier.solved,
                    "solve_rate": round(tier.solve_rate, 4),
                    "mean_reward": round(tier.mean_reward, 4),
                    "extraction_failure_rate": round(tier.extraction_failure_rate, 4),
                }
                for tier in self.tiers
            },
            "bugs": [bug for tier in self.tiers for bug in tier.bugs],
        }


def evaluate_curriculum(
    generate: Generate,
    model_name: str,
    tiers: Iterable[int] = TIERS,
    limit: int | None = None,
    on_bug: _Progress | None = None,
    split: str = "heldout",
    format: PromptFormat = "structured",
) -> CurriculumReport:
    """Score ``generate`` on every bug in ``tiers`` within ``split``.

    ``generate`` maps a prompt to a completion; keeping it a plain callable means
    this function does not care whether the model is local, remote, or a stub in
    a test. ``split`` defaults to the held-out side — the only side any reported
    number should come from. ``format`` must match whatever format ``generate``
    was steered towards (i.e. the format used to build its prompts), or the
    reward/extraction diagnostics mean nothing.
    """
    results = []
    for tier in tiers:
        bugs = list(load_tier(tier, split))[:limit]
        records = []
        solved = 0
        total_reward = 0.0
        extraction_failures = 0

        for index, bug in enumerate(bugs, start=1):
            completion = generate(bug_to_prompt(bug, format=format))
            outcome = score_response(bug, completion, format=format)

            solved += outcome.solved
            total_reward += outcome.reward.total
            extraction_failures += not outcome.extraction_ok
            records.append(
                {
                    "id": bug.id,
                    "tier": bug.tier,
                    "bug_type": bug.bug_type,
                    "function_name": bug.function_name,
                    "completion": completion,
                    "action": outcome.output.action,
                    "tests": outcome.tests.as_dict(),
                    "reward": outcome.reward.as_dict(),
                    "solved": outcome.solved,
                    "extraction_ok": outcome.extraction_ok,
                }
            )
            if on_bug is not None:
                on_bug(index, len(bugs), bug)

        results.append(
            TierResult(
                tier=tier,
                total=len(bugs),
                solved=solved,
                mean_reward=total_reward / len(bugs) if bugs else 0.0,
                extraction_failure_rate=extraction_failures / len(bugs) if bugs else 0.0,
                bugs=tuple(records),
            )
        )

    return CurriculumReport(model=model_name, tiers=tuple(results), format=format)


def load_generator(
    base_model: str,
    adapter: str | None = None,
    max_new_tokens: int = 300,
) -> tuple[Generate, str]:
    """Load a causal LM (optionally with a LoRA adapter) and return a greedy generator.

    Greedy decoding, not sampling: an evaluation that changes its answer between
    runs cannot be compared against anything.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Model evaluation needs torch and transformers: "
            "pip install 'agentdebugger[train]'"
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model = model.to(device)
    model.eval()

    def generate(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated = output[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(generated, skip_special_tokens=True)

    return generate, adapter or base_model
