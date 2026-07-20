"""The prompt the policy is trained and evaluated against.

Training and evaluation must use the *same* prompt, or an evaluation measures a
distribution shift rather than a policy. That is the only reason this is its own
module.

Two response formats are supported (research_plan.md §1, H1):

* ``"structured"`` — the OBSERVATION/HYPOTHESIS/CONFIDENCE/ACTION/DETAIL schema.
  This is what the project shipped with.
* ``"free_form"`` — no schema, just a worked example of reasoning-then-fix. It
  exists so H1 (does the *structure* help, independent of reward shaping) has a
  clean counterfactual: E3 (structured, R1) vs. E2 (free-form, R1).

The two system prompts are deliberately length-matched (within ~10% of each
other by word count) — see ``test_prompts.py`` — because a longer prompt that
also demonstrates a format is a confound the plan calls out explicitly
(research_plan.md, "Prompt-length confound").
"""

from __future__ import annotations

from typing import Literal

from agentdebugger.dataset import Bug

PromptFormat = Literal["structured", "free_form"]

SYSTEM_PROMPT = """You are an expert Python debugger. You reason through bugs systematically.

You MUST respond in EXACTLY this format — no exceptions, no extra text:

OBSERVATION: [What you see in the code and the error. Reference exact line numbers.]
HYPOTHESIS: [Why the bug causes this failure. At least two sentences. Name the variables, \
operators or logic involved.]
CONFIDENCE: [low | medium | high]
ACTION: [One of: inspect_lines | run_tests | propose_fix | request_context | give_up]
DETAIL: [For propose_fix: the complete corrected function. For inspect_lines: line numbers. \
Otherwise: the specifics.]

Rules:
- Never omit a field.
- HYPOTHESIS must explain WHY the bug produces the failure that was observed.
- For propose_fix, DETAIL must contain the whole function, not just the line you changed.
- Give up only once you have exhausted every reasonable hypothesis."""

#: The free-form counterpart to ``SYSTEM_PROMPT``: a worked example instead of a
#: schema. No field names, no required structure — the model reasons in prose
#: and hands back a fenced code block. Length-matched to ``SYSTEM_PROMPT`` on
#: purpose; see the module docstring.
FREE_FORM_SYSTEM_PROMPT = """You are an expert Python debugger. You will be shown a buggy \
function and the error it produces. Think it through like an experienced engineer: describe \
what you observe in the code and the error, reason step by step about why that failure \
happens, then decide what to do.

Here is a worked example of the expected style (not the bug you will see):

The loop's comparison on line 4 uses `<` instead of `<=`, so the final index is never reached \
and the last element is skipped every call — exactly the off-by-one in the error above. The \
fix is to change the comparison operator.

```python
def example(numbers, target):
    left, right = 0, len(numbers) - 1
    while left <= right:
        pass
```

Write your reasoning first, then give the complete corrected function in one fenced Python \
code block. If you cannot find the bug, say so plainly instead of guessing."""

_SYSTEM_PROMPTS: dict[str, str] = {
    "structured": SYSTEM_PROMPT,
    "free_form": FREE_FORM_SYSTEM_PROMPT,
}


def bug_to_prompt(bug: Bug, format: PromptFormat = "structured") -> str:
    """Render a bug as a ChatML prompt for the policy.

    ``format`` selects the system prompt (and therefore the response shape the
    model is steered towards): ``"structured"`` (default, backward compatible)
    or ``"free_form"``.
    """
    try:
        system_prompt = _SYSTEM_PROMPTS[format]
    except KeyError:
        raise ValueError(f"Unknown prompt format {format!r}. Choose from {tuple(_SYSTEM_PROMPTS)}.") from None

    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Debug this Python function:\n\n```python\n{bug.buggy_code}\n```\n\n"
        f"Initial failure: {bug.initial_error}\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
