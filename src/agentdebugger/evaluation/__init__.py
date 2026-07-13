"""Running agents and models against the environments, and reporting what they scored."""

from agentdebugger.evaluation.curriculum import (
    CurriculumReport,
    TierResult,
    evaluate_curriculum,
    load_generator,
)
from agentdebugger.evaluation.episode import (
    EpisodeResult,
    EvaluationReport,
    TurnRecord,
    evaluate_agent,
    run_episode,
)

__all__ = [
    "CurriculumReport",
    "EpisodeResult",
    "EvaluationReport",
    "TierResult",
    "TurnRecord",
    "evaluate_agent",
    "evaluate_curriculum",
    "load_generator",
    "run_episode",
]
