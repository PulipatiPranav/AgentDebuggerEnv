"""What the environment expects of an agent."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from agentdebugger.protocol import Action, Observation


@runtime_checkable
class Agent(Protocol):
    """Anything that can pick an action given an observation.

    ``name`` is what evaluation reports are keyed by, so it should identify the
    model or policy, not the class.
    """

    name: str

    def act(self, observation: Observation, info: dict[str, Any]) -> Action:
        """Choose the next action. ``info`` is the previous step's info dict, empty at reset."""
        ...
