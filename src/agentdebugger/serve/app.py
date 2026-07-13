"""HTTP interface to the multi-step task environment.

The endpoints follow the OpenEnv convention an external agent expects:

======  ==========  ===============================================
GET     /health     liveness check
GET     /tasks      task metadata
POST    /reset      start an episode, returns the first observation
POST    /step       apply one action, returns observation/reward/done
GET     /state      the current episode's internal state
======  ==========  ===============================================

The server holds **one** episode at a time, which is what the protocol assumes:
an agent resets, steps until done, and resets again. Run one process per agent
if you need concurrency — the sandbox is the expensive part, not the server.

Optional dependency: ``pip install 'agentdebugger[serve]'``.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agentdebugger import __version__
from agentdebugger.envs.task_env import EpisodeFinished, TaskEnvironment
from agentdebugger.protocol import Action
from agentdebugger.tasks import TASKS, list_tasks


class ResetRequest(BaseModel):
    task_id: str = Field(default="easy", description="easy | medium | hard")


class ActionRequest(BaseModel):
    """The wire form of :class:`agentdebugger.protocol.Action`."""

    action_type: str = Field(description="submit_fix | query_context | give_up")
    fixed_code: str | None = None
    hypothesis: str | None = None
    query_type: str | None = None
    query_target: str | None = None
    final_diagnosis: str | None = None

    def to_action(self) -> Action:
        return Action(**self.model_dump())


def create_app() -> FastAPI:
    """Build the FastAPI application around a fresh environment."""
    app = FastAPI(
        title="AgentDebuggerEnv",
        description="A sandboxed environment where agents debug real code and see real output.",
        version=__version__,
    )
    env = TaskEnvironment()

    @app.get("/")
    async def root() -> dict[str, Any]:
        return {
            "name": "AgentDebuggerEnv",
            "version": __version__,
            "tasks": list_tasks(),
            "action_types": ["submit_fix", "query_context", "give_up"],
            "reward_type": "dense",
        }

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/tasks")
    async def tasks() -> dict[str, Any]:
        return {
            "tasks": [
                {
                    "id": task.task_id,
                    "name": task.name,
                    "difficulty": task.difficulty,
                    "description": task.description,
                    "tests_total": task.tests_total,
                    "max_attempts": task.max_attempts,
                    "max_steps": task.max_steps,
                }
                for task in TASKS.values()
            ]
        }

    @app.post("/reset")
    async def reset(request: ResetRequest | None = None) -> JSONResponse:
        task_id = request.task_id if request else "easy"
        try:
            observation = env.reset(task_id)
        except ValueError as exc:
            return JSONResponse(
                {"error": str(exc), "available_tasks": list_tasks()}, status_code=400
            )
        return JSONResponse(observation.as_dict())

    @app.post("/step")
    async def step(request: ActionRequest) -> JSONResponse:
        try:
            result = env.step(request.to_action())
        except EpisodeFinished as exc:
            return JSONResponse({"error": str(exc)}, status_code=409)
        return JSONResponse(result.as_dict())

    @app.get("/state")
    async def state() -> dict[str, Any]:
        return env.state()

    return app


app = create_app()
