"""Parsing the structured response format.

Format compliance is part of the reward, so the parser has to be forgiving about
whitespace and case but strict about which fields are actually present. Both
sides of that are load-bearing.
"""

from __future__ import annotations

from agentdebugger.protocol import parse_agent_output

WELL_FORMED = """OBSERVATION: the loop on line 4 uses a strict less-than comparison
HYPOTHESIS: the condition should be left <= right, otherwise the final index is skipped
CONFIDENCE: high
ACTION: propose_fix
DETAIL: def binary_search(arr, target): ...
"""


def test_a_well_formed_response_parses_and_is_valid():
    output = parse_agent_output(WELL_FORMED)
    assert output.valid
    assert output.confidence == "high"
    assert output.action == "propose_fix"
    assert output.observation.startswith("the loop")


def test_parsing_tolerates_case_and_whitespace():
    output = parse_agent_output(
        "observation:   spread out\n\n"
        "hypothesis:    the bug is a detailed and specific claim about the cause\n"
        "Confidence: HIGH\n"
        "Action:  Propose_Fix\n"
        "detail: some fix here\n"
    )
    assert output.valid
    assert output.confidence == "high"
    assert output.action == "propose_fix"


def test_a_missing_field_makes_the_response_invalid():
    output = parse_agent_output(
        "OBSERVATION: something\nHYPOTHESIS: a claim about the bug\nACTION: propose_fix\n"
    )  # no CONFIDENCE, no DETAIL
    assert not output.valid


def test_an_unknown_action_degrades_to_invalid_rather_than_raising():
    output = parse_agent_output(
        "OBSERVATION: something here\n"
        "HYPOTHESIS: a claim about the bug that is long enough\n"
        "CONFIDENCE: high\n"
        "ACTION: teleport\n"
        "DETAIL: whatever\n"
    )
    assert output.action == "invalid"
    assert not output.valid


def test_freeform_prose_is_invalid_but_does_not_crash():
    output = parse_agent_output("I think the bug is in the loop somewhere, let me try.")
    assert not output.valid
    assert output.action == "invalid"
    assert output.confidence == "low"


def test_a_multiline_detail_is_captured_whole():
    output = parse_agent_output(
        "OBSERVATION: x is wrong\n"
        "HYPOTHESIS: a detailed claim about the root cause of the bug\n"
        "CONFIDENCE: medium\n"
        "ACTION: propose_fix\n"
        "DETAIL: def f():\n    line_one()\n    line_two()\n"
    )
    assert "line_one()" in output.detail
    assert "line_two()" in output.detail
