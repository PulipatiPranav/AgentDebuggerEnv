"""The two environments.

:class:`TaskEnvironment`
    Multi-step. Three hand-written tasks, tool-call style actions, an episode
    grader. This is what the HTTP server exposes.

:class:`CurriculumEnvironment`
    Single structured response per turn, scored densely against a tiered bug
    dataset. This is what GRPO trains against.

They share the sandbox, the protocol and the reward vocabulary, but they answer
different questions and are deliberately not forced into one class.
"""

from agentdebugger.envs.curriculum_env import (
    CurriculumEnvironment,
    TurnOutcome,
    score_response,
)
from agentdebugger.envs.task_env import EpisodeFinished, TaskEnvironment

__all__ = [
    "CurriculumEnvironment",
    "EpisodeFinished",
    "TaskEnvironment",
    "TurnOutcome",
    "score_response",
]
