"""The two prompt formats, and the length-match invariant H1 depends on.

research_plan.md flags prompt length as a confound: if the structured prompt is
meaningfully longer than the free-form one, a solve-rate gap could be explained
by "the model got more/better instruction" rather than by the format itself.
Both system prompts are asserted to be within 15% of each other by word count
(a cheap, tokenizer-independent proxy) so a future edit to either cannot drift
them apart silently.
"""

from __future__ import annotations

import pytest

from agentdebugger.dataset import load_bugs
from agentdebugger.training.prompts import (
    FREE_FORM_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    bug_to_prompt,
)

BUG = load_bugs((1,))[0]


def test_the_two_system_prompts_are_length_matched():
    structured_words = len(SYSTEM_PROMPT.split())
    free_form_words = len(FREE_FORM_SYSTEM_PROMPT.split())
    ratio = free_form_words / structured_words
    assert 0.85 <= ratio <= 1.15, (
        f"free-form ({free_form_words} words) and structured ({structured_words} words) "
        "prompts have drifted more than 15% apart — this is the prompt-length confound "
        "research_plan.md calls out for H1."
    )


def test_free_form_prompt_gives_a_worked_example_not_a_schema():
    """H1's independent variable is the schema, not general instruction quality."""
    for field in ("OBSERVATION:", "HYPOTHESIS:", "CONFIDENCE:", "ACTION:", "DETAIL:"):
        assert field not in FREE_FORM_SYSTEM_PROMPT
    assert "```" in FREE_FORM_SYSTEM_PROMPT  # the worked example still shows a fenced fix


def test_bug_to_prompt_defaults_to_structured():
    assert bug_to_prompt(BUG) == bug_to_prompt(BUG, format="structured")
    assert SYSTEM_PROMPT in bug_to_prompt(BUG)


def test_bug_to_prompt_free_form_uses_the_free_form_system_prompt():
    prompt = bug_to_prompt(BUG, format="free_form")
    assert FREE_FORM_SYSTEM_PROMPT in prompt
    assert SYSTEM_PROMPT not in prompt
    # The task-specific part (the actual buggy code) must still be present.
    assert BUG.buggy_code in prompt


def test_unknown_format_is_rejected():
    with pytest.raises(ValueError):
        bug_to_prompt(BUG, format="verbose")  # type: ignore[arg-type]
