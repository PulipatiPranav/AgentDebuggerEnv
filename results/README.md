# Results

Published evaluation results, kept in version control as a record.

> **Note on evaluation splits:** The initial implementation had no split; this was identified as a precondition in the preregistration and subsequently corrected before the reported experiments. The reported runs use the committed 90/90 split.

The results are organized as follows:

- **`primary/`**: The primary results reported in the paper (B0, B1 corrected 700-token, and the 9 RL runs for E1, E3, and E4).
- **`diagnostics/`**: Difficulty gate calibration (Llama-3.3-70B) and the offline oracle agent.
- **`historical/`**: Historical evaluation artifacts (e.g., the original 300-token B1 truncation bug, older pre-split evaluation runs). These are kept for provenance but are not used for any primary claim.

To reproduce a trained-model evaluation (e.g., E1 seed 42):

```bash
pip install -e '.[train]'
agentdebugger evaluate-curriculum \
    --adapter shashaank0707/AgentDebugger-trained \
    --split heldout \
    --output results/primary/E1_s42.json
```
