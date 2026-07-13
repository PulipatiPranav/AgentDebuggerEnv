"""Tests bound directly to the claims the README and report make.

Each test here corresponds to a sentence a reader could challenge. If one fails,
a documented claim is false, not merely a unit is broken.
"""

from __future__ import annotations

import pytest

from agentdebugger.config import DEFAULT_CURRICULUM, CurriculumSchedule
from agentdebugger.dataset import find_bug, load_bugs
from agentdebugger.envs import CurriculumEnvironment, score_response
from agentdebugger.evaluation import run_episode
from agentdebugger.rewards import TurnRewardCalculator
from agentdebugger.tasks import get_task
from agentdebugger.training.prompts import bug_to_prompt

# ── Claim: three tasks, each probing a distinct debugging failure mode ─────────


def test_the_hard_task_bug_is_invisible_to_every_sequential_test():
    """Claim: the race condition passes the whole suite; only concurrency reveals it."""
    task = get_task("hard")
    from agentdebugger.sandbox import execute

    result = execute(task.buggy_code, task.executable_tests, policy=task.policy)
    assert "8 passed, 0 failed" in result.output


def test_the_medium_error_points_at_the_wrong_function():
    """Claim: the error names authenticate_user; the bug is in hash_password."""
    task = get_task("medium")
    assert task.ground_truth.bug_location == "hash_password"
    assert task.ground_truth.red_herring_keyword == "authenticate_user"
    assert "authenticate_user" in task.description


# ── Claim: a hypothesis is mandatory before a fix is executed ──────────────────


def test_the_environment_refuses_to_run_a_fix_without_a_hypothesis():
    from agentdebugger.envs.task_env import TaskEnvironment
    from agentdebugger.protocol import Action

    env = TaskEnvironment()
    env.reset("easy")
    before = env.state()["attempts_used"]
    result = env.step(
        Action(action_type="submit_fix", fixed_code=get_task("easy").ground_truth.fixed_code)
    )
    assert env.state()["attempts_used"] == before  # the fix was never executed
    assert result.reward.step_reward < 0


# ── Claim: the reward range is [-0.5, 1.0] ─────────────────────────────────────


def test_the_dense_reward_stays_within_its_advertised_range():
    """Claim: reward_range = [-0.5, 1.0]. Checked across every bug and several responses."""
    calculator = TurnRewardCalculator()
    from agentdebugger.rewards import GroundTruth

    responses = [
        "garbage with no structure at all",
        "OBSERVATION: x\nHYPOTHESIS: y\nCONFIDENCE: high\nACTION: give_up\nDETAIL: z\n",
    ]
    for bug in load_bugs():
        truth = GroundTruth.from_bug(bug.as_dict())
        for raw in responses:
            from agentdebugger.protocol import parse_agent_output

            for turn in (0, 5):
                total = calculator.compute_turn_reward(
                    parse_agent_output(raw),
                    truth,
                    {"passed": 0, "total": len(bug.test_cases), "newly_broken": len(bug.test_cases)},
                    turn,
                ).total
                assert -0.5 <= total <= 1.0


# ── Claim: the curriculum introduces tiers progressively ───────────────────────


def test_the_default_curriculum_gates_tiers_by_step():
    """Claim: tier 1 first, tier 2 at step 150, tier 3 at step 350."""
    schedule = DEFAULT_CURRICULUM
    assert schedule.tiers_at(0) == (1,)
    assert schedule.tiers_at(149) == (1,)
    assert schedule.tiers_at(150) == (1, 2)
    assert schedule.tiers_at(349) == (1, 2)
    assert schedule.tiers_at(350) == (1, 2, 3)
    assert schedule.advances_at() == (150, 350)


def test_the_curriculum_environment_only_samples_active_tiers():
    env = CurriculumEnvironment(step=0, seed=1)
    assert env.active_tiers == (1,)
    assert all(bug.tier == 1 for bug in env.bugs)

    env.advance_to(150)
    assert env.active_tiers == (1, 2)
    assert {bug.tier for bug in env.bugs} == {1, 2}


def test_an_invalid_curriculum_is_rejected():
    with pytest.raises(ValueError):
        CurriculumSchedule(stages=())  # must have a stage at step 0


# ── Claim: training and evaluation use the same scoring path ───────────────────


def test_training_and_evaluation_score_a_response_identically():
    """Claim: score_response is the single scoring path shared by both.

    A reward reported during training must mean the same as one reported during
    evaluation, so both go through score_response for the same bug and response.
    """
    bug = find_bug("t1_001")
    fix = (
        "OBSERVATION: right starts at len(arr), one past the end\n"
        "HYPOTHESIS: the search window includes an out-of-range index on line 2, "
        "so binary_search raises IndexError; right should start at len(arr) - 1\n"
        "CONFIDENCE: high\n"
        "ACTION: propose_fix\n"
        f"DETAIL: {bug.original_code}\n"
    )

    from agentdebugger.training import make_reward_function

    training_reward = make_reward_function()([fix], [bug_to_prompt(bug)], bug_metadata=[bug.as_dict()])[0]
    eval_reward = score_response(bug, fix).reward.total
    assert training_reward == eval_reward


def test_the_reference_fix_earns_a_high_dense_reward():
    """Claim: the reward function actually rewards correct fixes, not just formatting."""
    for bug in load_bugs((1,))[:5]:
        response = (
            "OBSERVATION: the buggy line is identified with its line number here\n"
            f"HYPOTHESIS: {bug.function_name} has a {bug.bug_type} bug on line "
            f"{bug.location.line_start}; the fix restores the intended behaviour\n"
            "CONFIDENCE: high\n"
            "ACTION: propose_fix\n"
            f"DETAIL: {bug.original_code}\n"
        )
        outcome = score_response(bug, response)
        assert outcome.solved
        assert outcome.reward.total > 0.5


# ── Claim: a model with no API key or GPU can run a full episode ───────────────


def test_a_full_episode_runs_offline_with_no_model():
    """Claim: the oracle lets someone with no GPU and no API key watch an episode."""
    from agentdebugger import OracleAgent

    result = run_episode(OracleAgent(), "hard")
    assert result.solved
    assert result.grader_score >= 0.9
