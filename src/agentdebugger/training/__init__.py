"""GRPO training against the curriculum environment.

Importing this package is cheap: ``torch``, ``trl`` and ``peft`` are imported
inside :func:`~agentdebugger.training.grpo.train`, not at module level, so the
evaluation and serving paths never pay for them.
"""

from agentdebugger.training.grpo import (
    HardwareProfile,
    TrainingConfig,
    build_dataset,
    make_reward_function,
    train,
)
from agentdebugger.training.prompts import FREE_FORM_SYSTEM_PROMPT, SYSTEM_PROMPT, bug_to_prompt

__all__ = [
    "FREE_FORM_SYSTEM_PROMPT",
    "SYSTEM_PROMPT",
    "HardwareProfile",
    "TrainingConfig",
    "bug_to_prompt",
    "build_dataset",
    "make_reward_function",
    "train",
]
