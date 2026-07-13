"""The multi-step environment: episode lifecycle, step rewards, and its guardrails."""

from __future__ import annotations

import pytest

from agentdebugger.agents import OracleAgent
from agentdebugger.envs.task_env import EpisodeFinished, TaskEnvironment
from agentdebugger.evaluation import run_episode
from agentdebugger.protocol import Action
from agentdebugger.tasks import get_task


@pytest.fixture
def env():
    return TaskEnvironment()


# ── lifecycle ─────────────────────────────────────────────────────────────────


def test_reset_runs_the_buggy_code_and_shows_the_real_failure(env):
    observation = env.reset("easy")
    assert observation.task_id == "easy"
    assert observation.done is False
    assert observation.tests_passed < observation.tests_total
    assert "passed" in observation.initial_error_output


def test_stepping_before_reset_is_an_error(env):
    with pytest.raises(EpisodeFinished):
        env.step(Action(action_type="give_up"))


def test_stepping_after_the_episode_ends_is_an_error(env):
    env.reset("easy")
    env.step(Action(action_type="give_up"))
    with pytest.raises(EpisodeFinished):
        env.step(Action(action_type="give_up"))


# ── submitting a fix ──────────────────────────────────────────────────────────


def test_a_correct_fix_solves_the_episode(env):
    env.reset("easy")
    result = env.step(
        Action(
            action_type="submit_fix",
            fixed_code=get_task("easy").ground_truth.fixed_code,
            hypothesis="off-by-one in the termination condition; use left <= right",
        )
    )
    assert result.done is True
    assert result.info["solved"] is True
    assert result.observation.tests_passed == result.observation.tests_total
    assert result.reward.grader_score >= 0.9


def test_a_fix_without_a_hypothesis_is_not_executed(env):
    """The central constraint: no hypothesis, no run. Guessing must not be free."""
    env.reset("easy")
    result = env.step(
        Action(action_type="submit_fix", fixed_code=get_task("easy").ground_truth.fixed_code)
    )
    assert result.reward.step_reward < 0
    assert result.info.get("solved") is None  # never ran, so never evaluated
    assert env.state()["attempts_used"] == 0
    assert not result.done


def test_partial_progress_is_rewarded_before_the_episode_ends(env):
    """A fix that turns some red tests green earns something at that step, not only at the end."""
    env.reset("medium")
    partial = "def hash_password(password):\n    return password\n"  # wrong, but changes behaviour
    result = env.step(
        Action(action_type="submit_fix", fixed_code=partial, hypothesis="maybe the hash is wrong")
    )
    assert not result.done
    assert result.reward.grader_score == 0.0  # no verdict mid-episode


def test_the_attempt_budget_is_enforced(env):
    task = get_task("easy")
    env.reset("easy")
    wrong = "def binary_search(arr, target):\n    return -1\n"
    last = None
    for _ in range(task.max_attempts):
        last = env.step(
            Action(action_type="submit_fix", fixed_code=wrong, hypothesis="a plausible guess here")
        )
    assert last.done is True
    assert last.observation.attempts_remaining == 0


# ── querying for context ──────────────────────────────────────────────────────


def test_the_first_query_is_free_and_later_ones_cost(env):
    env.reset("easy")
    first = env.step(Action(action_type="query_context", query_type="related_code"))
    second = env.step(Action(action_type="query_context", query_type="test_details"))
    assert first.reward.step_reward == 0.0
    assert second.reward.step_reward < 0.0
    assert first.observation.hint_used is True


def test_the_hard_task_hint_points_at_concurrency_without_naming_the_fix(env):
    env.reset("hard")
    result = env.step(Action(action_type="query_context", query_type="test_suggestion"))
    hint = result.info["query_result"].lower()
    assert "thread" in hint
    assert "lock" not in hint  # a hint, not the answer


def test_an_unknown_action_is_rejected_without_ending_the_episode(env):
    env.reset("easy")
    result = env.step(Action(action_type="teleport"))
    assert result.reward.step_reward < 0
    assert not result.done
    assert "Unknown action_type" in result.info["error"]


# ── termination ───────────────────────────────────────────────────────────────


def test_the_step_budget_truncates_a_stalling_agent(env):
    task = get_task("easy")
    env.reset("easy")
    result = None
    for _ in range(task.max_steps + 1):
        if result and result.done:
            break
        result = env.step(Action(action_type="query_context", query_type="related_code"))
    assert result.done is True


def test_scores_come_from_the_terminal_step_only(env):
    env.reset("easy")
    mid = env.step(Action(action_type="query_context", query_type="related_code"))
    assert mid.reward.grader_score == 0.0
    final = env.step(Action(action_type="give_up", final_diagnosis="I give up"))
    assert final.done is True


# ── the oracle solves everything (this is also a solvability check) ────────────


@pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
def test_the_oracle_solves_every_task_in_one_attempt(task_id):
    """If the oracle cannot solve a task, the task or the sandbox is broken."""
    result = run_episode(OracleAgent(), task_id)
    assert result.solved is True
    assert result.attempts_used == 1
    assert result.grader_score >= 0.9
