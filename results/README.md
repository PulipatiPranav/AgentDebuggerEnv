# Results

Published evaluation results, kept in version control as a record.

| File | What it is |
| --- | --- |
| `qwen2.5-coder-3b-grpo.json` | The GRPO-trained `Qwen2.5-Coder-3B` adapter, scored on all 90 curriculum bugs. |
| `oracle.json` | The oracle agent on the three hand-written tasks — the score ceiling every model is compared against. Regenerate with `agentdebugger evaluate --output results/oracle.json`. |

To reproduce the trained-model numbers:

```bash
pip install -e '.[train]'
agentdebugger evaluate-curriculum \
    --adapter shashaank0707/AgentDebugger-trained \
    --output results/qwen2.5-coder-3b-grpo.json
```
