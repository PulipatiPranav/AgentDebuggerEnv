"""An agent backed by any OpenAI-compatible chat endpoint.

Works against the Hugging Face router, vLLM, Ollama, or the OpenAI API itself —
anything that speaks ``/v1/chat/completions``. This is the path used to produce
baseline numbers for models that are not being trained.

``openai`` is an optional dependency: install with ``pip install
'agentdebugger[api]'``.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from typing import Any

from agentdebugger.protocol import Action, Observation

SYSTEM_PROMPT = """You are an expert software debugger. You are given broken code and a \
failing test suite, and you fix it by forming a hypothesis, testing it, and revising.

Always reply with a single JSON object and nothing else. The available actions are:

Submit a fix (a hypothesis is mandatory — a fix submitted without one is not run):
{"action_type": "submit_fix",
 "fixed_code": "<the complete corrected source, not a diff or a snippet>",
 "hypothesis": "<what the root cause is and where it is>"}

Ask for context (the first question is free, later ones cost reward):
{"action_type": "query_context",
 "query_type": "error_explanation" | "function_signature" | "related_code" | "test_details" | "test_suggestion",
 "query_target": "<a function or test name, optional>"}

Give up:
{"action_type": "give_up", "final_diagnosis": "<your best explanation of the bug>"}

Rules:
- Submit the complete file, never a fragment.
- Take the previous execution output into account; it is real, not a simulation.
- A green test suite is evidence, not proof. Ask what the tests do not cover."""

_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class ApiAgent:
    """Drives an episode through an OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1500,
        max_retries: int = 5,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The API agent needs the openai package: pip install 'agentdebugger[api]'"
            ) from exc

        self.name = model or os.environ.get("MODEL_NAME", "")
        if not self.name:
            raise ValueError("Set --model or MODEL_NAME to name the model to evaluate.")

        base_url = base_url or os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
        api_key = api_key or os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Set HF_TOKEN or OPENAI_API_KEY to call the inference API.")

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._messages: list[dict[str, str]] = []

    def act(self, observation: Observation, info: dict[str, Any]) -> Action:
        """Send the current state to the model and parse its next action."""
        if not self._messages:
            self._messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _opening_prompt(observation)},
            ]
        else:
            self._messages.append({"role": "user", "content": _feedback_prompt(observation, info)})

        raw = self._complete()
        self._messages.append({"role": "assistant", "content": raw})
        return _parse_action(raw)

    def reset(self) -> None:
        """Forget the conversation so the agent can be reused for another episode."""
        self._messages = []

    def _complete(self) -> str:
        """Call the endpoint, retrying transient failures with exponential backoff."""
        from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError

        for attempt in range(self._max_retries):
            try:
                completion = self._client.chat.completions.create(
                    model=self.name,
                    messages=self._messages,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                    timeout=90.0,
                )
                return completion.choices[0].message.content or ""
            except (RateLimitError, APIConnectionError, APITimeoutError) as exc:
                self._backoff(attempt, exc)
            except APIError as exc:
                if getattr(exc, "status_code", None) not in _RETRYABLE_STATUS:
                    raise
                self._backoff(attempt, exc)
        return ""

    def _backoff(self, attempt: int, exc: Exception) -> None:
        if attempt == self._max_retries - 1:
            raise exc
        delay = 2**attempt + random.random()
        print(f"  [api] {type(exc).__name__}; retrying in {delay:.1f}s", flush=True)
        time.sleep(delay)


def _parse_action(raw: str) -> Action:
    """Turn a model response into an Action, tolerating code fences and stray prose."""
    stripped = _JSON_FENCE.sub("", raw.strip())
    payload: dict[str, Any] | None = None

    for candidate in (stripped, _first_object(stripped)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            payload = parsed
            break

    if payload is None:
        # An unparseable response is a real failure mode, and giving up is the
        # honest way to record it: it ends the episode and scores what was earned.
        return Action(
            action_type="give_up",
            final_diagnosis=f"Response was not valid JSON: {raw[:200]}",
        )

    return Action(
        action_type=str(payload.get("action_type", "")),
        fixed_code=payload.get("fixed_code"),
        hypothesis=payload.get("hypothesis"),
        query_type=payload.get("query_type"),
        query_target=payload.get("query_target"),
        final_diagnosis=payload.get("final_diagnosis"),
    )


def _first_object(text: str) -> str | None:
    match = _JSON_OBJECT.search(text)
    return match.group() if match else None


def _opening_prompt(observation: Observation) -> str:
    return (
        f"=== DEBUGGING TASK: {observation.task_id.upper()} ===\n\n"
        f"{observation.task_description}\n\n"
        f"CODE:\n```python\n{observation.buggy_code}```\n\n"
        f"TEST SUITE:\n```python\n{observation.test_suite}```\n\n"
        f"OUTPUT OF THE TEST SUITE:\n{observation.initial_error_output}\n\n"
        f"Tests passing: {observation.tests_passed}/{observation.tests_total}\n"
        f"Attempts remaining: {observation.attempts_remaining}\n"
        f"Steps remaining: {observation.max_steps - observation.step_number}\n\n"
        f"Diagnose the bug and submit your first fix."
    )


def _feedback_prompt(observation: Observation, info: dict[str, Any]) -> str:
    lines = [
        f"Step {observation.step_number}:",
        f"Tests passing: {observation.tests_passed}/{observation.tests_total}",
        f"Attempts remaining: {observation.attempts_remaining}",
    ]
    if info.get("error"):
        lines.append(f"ERROR: {info['error']}")
    if info.get("query_result"):
        lines.append(f"\nCONTEXT:\n{info['query_result']}")

    if observation.previous_attempts:
        output = observation.previous_attempts[-1].execution_output
        if len(output) > 1500:
            output = f"{output[:750]}\n...[truncated]...\n{output[-750:]}"
        lines.append(f"\nTEST OUTPUT:\n{output}")

    remaining = observation.tests_total - observation.tests_passed
    lines.append(
        "\nAll tests pass. If you are confident the root cause is fixed, you are done."
        if remaining == 0
        else f"\n{remaining} test(s) still failing. Revise your hypothesis and continue."
    )
    return "\n".join(lines)
