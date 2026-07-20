"""End-to-end checks for the free-form arm (E2/B1), the Phase A H1 plumbing.

These tests exercise the whole path a free-form completion travels: prompt ->
sandbox-scored outcome -> training reward -> evaluation report, mirroring
test_claims.py's structured-arm claims so the two formats are held to the same
standard.
"""

from __future__ import annotations

from agentdebugger.dataset import load_bugs
from agentdebugger.envs import score_response
from agentdebugger.evaluation import evaluate_curriculum
from agentdebugger.rewards import TurnRewardCalculator
from agentdebugger.training import build_dataset, make_reward_function
from agentdebugger.training.grpo import _grouped
from agentdebugger.training.prompts import bug_to_prompt

BUG = load_bugs((1,))[0]


def _free_form_fix(bug):
    return (
        f"Looking at {bug.function_name}, the bug is on line {bug.location.line_start}: "
        "it does not match the intended behaviour, which is why the tests fail.\n\n"
        f"```python\n{bug.original_code}\n```"
    )


# ── score_response(format="free_form") ─────────────────────────────────────────


def test_a_correct_free_form_fix_is_solved_under_r1():
    calculator = TurnRewardCalculator.terminal()  # R1: E2's reward config
    outcome = score_response(BUG, _free_form_fix(BUG), calculator=calculator, format="free_form")
    assert outcome.solved
    assert outcome.extraction_ok
    assert outcome.reward.total == outcome.reward.fix_quality  # R1: only fix + penalties survive


def test_a_free_form_give_up_is_not_scored_as_an_extraction_failure():
    calculator = TurnRewardCalculator.terminal()
    outcome = score_response(
        BUG, "I have thought about this a lot and I cannot find the bug.", calculator=calculator, format="free_form"
    )
    assert not outcome.solved
    assert outcome.extraction_ok  # a deliberate give-up, not a parser failure
    assert outcome.output.action == "give_up"


def test_free_form_prose_with_no_code_is_an_extraction_failure_not_a_solve():
    calculator = TurnRewardCalculator.terminal()
    outcome = score_response(
        BUG, "hmm, tricky one, not sure honestly", calculator=calculator, format="free_form"
    )
    assert not outcome.solved
    assert not outcome.extraction_ok


def test_unknown_format_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        score_response(BUG, "whatever", format="verbose")


# ── training: build_dataset / make_reward_function thread the format through ──


def test_build_dataset_free_form_uses_the_free_form_prompt():
    from agentdebugger.training.prompts import FREE_FORM_SYSTEM_PROMPT

    dataset = build_dataset(step=0, split="train", format="free_form")
    assert FREE_FORM_SYSTEM_PROMPT in dataset[0]["prompt"]


def test_training_and_evaluation_score_free_form_identically():
    """The free-form arm gets the same cross-check test_claims.py runs for structured."""
    fix = _free_form_fix(BUG)
    reward_fn = make_reward_function(reward_config="R1", format="free_form")
    training_reward = reward_fn(
        [fix], [bug_to_prompt(BUG, format="free_form")], bug_metadata=[BUG.as_dict()]
    )[0]
    eval_reward = score_response(
        BUG, fix, calculator=TurnRewardCalculator.terminal(), format="free_form"
    ).reward.total
    assert training_reward == eval_reward


def test_scoring_failure_reward_when_bug_metadata_is_missing():
    from agentdebugger.training.grpo import SCORING_FAILURE_REWARD

    reward_fn = make_reward_function()
    rewards = reward_fn(["anything"], ["some prompt"], bug_metadata=[None])
    assert rewards == [SCORING_FAILURE_REWARD]


# ── the degenerate-group grouping helper (H2 mediator) ──────────────────────────


def test_grouped_splits_on_contiguous_prompt_runs():
    prompts = ["a", "a", "b", "b", "b", "a"]
    values = [1, 2, 3, 4, 5, 6]
    assert _grouped(prompts, values) == [[1, 2], [3, 4, 5], [6]]


def test_grouped_rejects_mismatched_lengths():
    import pytest

    with pytest.raises(ValueError):
        _grouped(["a", "b"], [1])


# ── evaluation harness reports extraction-failure rate per format ─────────────


def test_evaluate_curriculum_reports_extraction_failure_rate_for_free_form():
    def generate(prompt: str) -> str:
        return "no idea what to do here, sorry"  # neither code nor a give-up phrase

    report = evaluate_curriculum(
        generate, "stub", tiers=(1,), limit=2, split="train", format="free_form"
    )
    assert report.format == "free_form"
    assert report.extraction_failure_rate == 1.0
    assert report.solve_rate == 0.0


def test_evaluate_curriculum_structured_format_failure_rate_is_reported_too():
    def generate(prompt: str) -> str:
        return "no schema here, just prose"

    report = evaluate_curriculum(generate, "stub", tiers=(1,), limit=2, split="train")
    assert report.format == "structured"
    assert report.extraction_failure_rate == 1.0
