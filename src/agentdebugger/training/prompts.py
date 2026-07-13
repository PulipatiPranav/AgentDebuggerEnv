"""The prompt the policy is trained and evaluated against.

Training and evaluation must use the *same* prompt, or an evaluation measures a
distribution shift rather than a policy. That is the only reason this is its own
module.
"""

from __future__ import annotations

from agentdebugger.dataset import Bug

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


def bug_to_prompt(bug: Bug) -> str:
    """Render a bug as a ChatML prompt for the policy."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Debug this Python function:\n\n```python\n{bug.buggy_code}\n```\n\n"
        f"Initial failure: {bug.initial_error}\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
