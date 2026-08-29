# Results

Published evaluation results, kept in version control as a record.

> **Note on evaluation splits:** The initial implementation had no split; this was identified as a precondition in the preregistration and subsequently corrected before the reported experiments. The reported runs use the committed 90/90 split.

| File | What it is |
| --- | --- |
| `qwen2.5-coder-3b-grpo.json` | The GRPO-trained `Qwen2.5-Coder-3B` adapter, scored on the 90 held-out curriculum bugs. |
| `oracle.json` | The oracle agent on the three hand-written tasks — the score ceiling every model is compared against. Regenerate with `agentdebugger evaluate --output results/oracle.json`. |

To reproduce the trained-model numbers:

```bash
pip install -e '.[train]'
agentdebugger evaluate-curriculum \
    --adapter shashaank0707/AgentDebugger-trained \
    --output results/qwen2.5-coder-3b-grpo.json
```
