"""The dense turn reward is what GRPO optimises, so its shape is a claim to defend.

The properties tested here are the ones the README and report state outright:
a perfect first-turn solve scores exactly 1.0, the total never drops below -0.5,
a real fix outscores good prose about a fix, and confidence is calibrated.
"""

from __future__ import annotations

import pytest

from agentdebugger.protocol import parse_agent_output
from agentdebugger.rewards import REWARD_FLOOR, GroundTruth, TurnRewardCalculator

BINARY_SEARCH_TRUTH = GroundTruth(
    bug_function="binary_search",
    bug_line=4,
    bug_type="off_by_one",
    canonical_fix_code="def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        pass",
)


def response(observation="", hypothesis="", confidence="high", action="propose_fix", detail=""):
    return parse_agent_output(
        f"OBSERVATION: {observation}\n"
        f"HYPOTHESIS: {hypothesis}\n"
        f"CONFIDENCE: {confidence}\n"
        f"ACTION: {action}\n"
        f"DETAIL: {detail}\n"
    )


@pytest.fixture
def calculator():
    return TurnRewardCalculator()


# ── format compliance ─────────────────────────────────────────────────────────


def test_a_well_formed_response_earns_full_format_credit(calculator):
    output = response(
        observation="right is initialized past the end of the list",
        hypothesis="the loop uses < instead of <= so the last index is skipped every time",
        detail="def binary_search(arr, target): ...",
    )
    breakdown = calculator.compute_turn_reward(output, BINARY_SEARCH_TRUTH, {}, 0)
    assert breakdown.format_compliance == calculator.FORMAT_MAX


def test_prose_that_ignores_the_format_is_penalised(calculator):
    output = parse_agent_output("I think the bug is somewhere around the loop, let me try a fix.")
    breakdown = calculator.compute_turn_reward(output, BINARY_SEARCH_TRUTH, {}, 0)
    assert breakdown.format_compliance < 0
    assert breakdown.penalties < 0


def test_partial_formatting_scores_between_prose_and_a_full_response(calculator):
    """The format reward has to give a gradient, or the policy cannot climb toward it."""
    prose = calculator.compute_turn_reward(
        parse_agent_output("no structure at all"), BINARY_SEARCH_TRUTH, {}, 0
    ).format_compliance
    partial = calculator.compute_turn_reward(
        response(observation="I can see the loop boundary is wrong here", action="nonsense"),
        BINARY_SEARCH_TRUTH,
        {},
        0,
    ).format_compliance
    assert prose < partial < calculator.FORMAT_MAX


# ── fix quality ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "passed,total,expected",
    [(8, 8, 0.35), (6, 8, 0.20), (4, 8, 0.12), (1, 8, 0.05), (0, 8, 0.0)],
)
def test_fix_quality_tracks_the_pass_rate(calculator, passed, total, expected):
    output = response(hypothesis="a sufficiently detailed hypothesis about the bug goes here")
    breakdown = calculator.compute_turn_reward(
        output, BINARY_SEARCH_TRUTH, {"passed": passed, "total": total}, 0
    )
    assert breakdown.fix_quality == expected


def test_fix_quality_is_zero_without_a_proposed_fix(calculator):
    output = response(action="inspect_lines", detail="4, 5")
    breakdown = calculator.compute_turn_reward(
        output, BINARY_SEARCH_TRUTH, {"passed": 8, "total": 8}, 0
    )
    assert breakdown.fix_quality == 0.0


def test_fix_quality_is_the_single_largest_lever(calculator):
    """Passing the tests is worth more than any other single thing the agent can do."""
    assert calculator.FIX_MAX > calculator.HYPOTHESIS_MAX
    assert calculator.FIX_MAX > calculator.LOCALIZATION_MAX
    assert calculator.FIX_MAX > calculator.SEMANTIC_MAX
    assert calculator.FIX_MAX > calculator.FORMAT_MAX


def test_similarity_to_the_canonical_fix_cannot_substitute_for_passing_it(calculator):
    """The reward-hacking surface — pasting canonical-looking code — is capped small.

    An agent that submits code textually close to the reference but that does not
    pass the tests earns only the small semantic nudge, never fix-level reward.
    """
    looks_right_but_fails = response(
        hypothesis="this is my best guess at the off-by-one in the termination condition",
        detail=BINARY_SEARCH_TRUTH.canonical_fix_code,
    )
    breakdown = calculator.compute_turn_reward(
        looks_right_but_fails, BINARY_SEARCH_TRUTH, {"passed": 0, "total": 8}, 0
    )
    assert breakdown.fix_quality == 0.0
    assert breakdown.semantic_similarity <= calculator.SEMANTIC_MAX
    assert breakdown.semantic_similarity < calculator.FIX_MAX


# ── hypothesis quality and calibration ────────────────────────────────────────


def test_a_specific_grounded_hypothesis_beats_a_vague_one(calculator):
    vague = calculator.compute_turn_reward(
        response(observation="there is a bug", hypothesis="something is wrong with the code"),
        BINARY_SEARCH_TRUTH,
        {"passed": 8, "total": 8},
        0,
    )
    specific = calculator.compute_turn_reward(
        response(
            observation="the while loop on line 4 uses a strict less-than comparison",
            hypothesis="on line 4 the condition left < right exits before comparing the final "
            "index, so a target at the last position is reported as -1 instead of found",
        ),
        BINARY_SEARCH_TRUTH,
        {"passed": 8, "total": 8},
        0,
    )
    assert specific.hypothesis_quality > vague.hypothesis_quality


def test_confidence_is_calibrated(calculator):
    """Being confidently wrong is worse than being cautiously wrong."""
    hypothesis = "the loop boundary on line 4 is off by one and skips the last comparison entirely"
    failing = {"passed": 2, "total": 8}

    confident_wrong = calculator.compute_turn_reward(
        response(hypothesis=hypothesis, confidence="high"), BINARY_SEARCH_TRUTH, failing, 0
    )
    cautious_wrong = calculator.compute_turn_reward(
        response(hypothesis=hypothesis, confidence="low"), BINARY_SEARCH_TRUTH, failing, 0
    )
    assert confident_wrong.hypothesis_quality < cautious_wrong.hypothesis_quality


# ── localization ──────────────────────────────────────────────────────────────


def test_localization_rewards_naming_the_function_and_line(calculator):
    located = calculator.compute_turn_reward(
        response(hypothesis="binary_search has an off-by-one on line 4 in its termination test"),
        BINARY_SEARCH_TRUTH,
        {"passed": 8, "total": 8},
        0,
    )
    vague = calculator.compute_turn_reward(
        response(hypothesis="there is an off-by-one error somewhere in the comparison logic"),
        BINARY_SEARCH_TRUTH,
        {"passed": 8, "total": 8},
        0,
    )
    assert located.localization > vague.localization
    assert located.localization <= calculator.LOCALIZATION_MAX


# ── penalties ─────────────────────────────────────────────────────────────────


def test_breaking_a_passing_test_is_penalised(calculator):
    output = response(hypothesis="this change should be safe but it regresses another test")
    breakdown = calculator.compute_turn_reward(
        output, BINARY_SEARCH_TRUTH, {"passed": 5, "total": 8, "newly_broken": 2}, 0
    )
    assert breakdown.penalties <= -0.20


def test_giving_up_is_penalised(calculator):
    output = response(
        hypothesis="I cannot work out what is wrong with this function", action="give_up"
    )
    breakdown = calculator.compute_turn_reward(output, BINARY_SEARCH_TRUTH, {}, 0)
    assert breakdown.penalties <= -0.15


# ── bounds: these define the advertised reward range ──────────────────────────


def test_a_perfect_first_turn_solve_scores_exactly_one(calculator):
    output = response(
        observation="binary_search line 4 uses left < right and skips the final index",
        hypothesis="the termination condition on line 4 should be left <= right; as written the "
        "loop exits one step early and the last element is never compared, reading as missing",
        confidence="high",
        detail=BINARY_SEARCH_TRUTH.canonical_fix_code,
    )
    breakdown = calculator.compute_turn_reward(
        output, BINARY_SEARCH_TRUTH, {"passed": 8, "total": 8}, 0
    )
    assert breakdown.total == pytest.approx(1.0, abs=1e-9)


def test_the_total_never_falls_below_the_floor(calculator):
    worst = parse_agent_output("garbage")  # invalid format, no fields, unknown action
    breakdown = calculator.compute_turn_reward(
        worst, BINARY_SEARCH_TRUTH, {"passed": 0, "total": 8, "newly_broken": 8}, 999
    )
    assert breakdown.total >= REWARD_FLOOR


def test_efficiency_rewards_solving_while_turns_remain(calculator):
    output = response(hypothesis="a sufficiently detailed hypothesis about the off-by-one bug")
    early = calculator.compute_turn_reward(output, BINARY_SEARCH_TRUTH, {"passed": 8, "total": 8}, 0)
    late = calculator.compute_turn_reward(output, BINARY_SEARCH_TRUTH, {"passed": 8, "total": 8}, 4)
    assert early.efficiency_potential > late.efficiency_potential


# ── episode aggregation ───────────────────────────────────────────────────────


def test_solving_earlier_beats_solving_later(calculator):
    solve = response(
        hypothesis="binary_search line 4 off-by-one; the loop should use left <= right to compare "
        "the final index instead of exiting early and missing the last element",
        detail=BINARY_SEARCH_TRUTH.canonical_fix_code,
    )
    turn = lambda n: {  # noqa: E731 - terse on purpose
        "reward": calculator.compute_turn_reward(
            solve, BINARY_SEARCH_TRUTH, {"passed": 8, "total": 8}, n
        )
    }
    early = calculator.compute_episode_reward([turn(0)])
    late = calculator.compute_episode_reward([turn(3)])
    assert early > late
