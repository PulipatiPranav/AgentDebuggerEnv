"""AgentDebuggerEnv — a reinforcement learning environment for debugging Python.

An agent is shown broken code and real test output, and has to state a
hypothesis before it is allowed to run a fix. Every submission executes in a
resource-limited sandbox, and a dense reward prices what the agent did: the
format it answered in, whether it localised the bug, whether the fix passed, and
whether it broke anything that used to work.

Two entry points, depending on what you are doing:

>>> from agentdebugger import TaskEnvironment, OracleAgent
>>> from agentdebugger.evaluation import run_episode
>>> result = run_episode(OracleAgent(), "hard")     # doctest: +SKIP
>>> result.solved                                    # doctest: +SKIP
True

>>> from agentdebugger import CurriculumEnvironment  # what GRPO trains against
"""

from agentdebugger.agents import Agent, OracleAgent
from agentdebugger.config import DEFAULT_CURRICULUM, CurriculumSchedule, SandboxLimits
from agentdebugger.envs import CurriculumEnvironment, TaskEnvironment, score_response
from agentdebugger.protocol import (
    Action,
    Observation,
    Reward,
    StructuredAgentOutput,
    parse_agent_output,
)
from agentdebugger.rewards import RewardBreakdown, TurnRewardCalculator, get_grader
from agentdebugger.sandbox import SandboxPolicy, execute, run_test_cases
from agentdebugger.tasks import get_task, list_tasks

__version__ = "2.0.0"

__all__ = [
    "DEFAULT_CURRICULUM",
    "Action",
    "Agent",
    "CurriculumEnvironment",
    "CurriculumSchedule",
    "Observation",
    "OracleAgent",
    "Reward",
    "RewardBreakdown",
    "SandboxLimits",
    "SandboxPolicy",
    "StructuredAgentOutput",
    "TaskEnvironment",
    "TurnRewardCalculator",
    "__version__",
    "execute",
    "get_grader",
    "get_task",
    "list_tasks",
    "parse_agent_output",
    "run_test_cases",
    "score_response",
]
