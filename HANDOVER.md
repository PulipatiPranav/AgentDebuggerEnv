# AgentDebuggerEnv — Project Handover

## What This Project Is
A GRPO-trained LLM (Qwen2.5-Coder-7B-Instruct) that learns to debug Python code through
structured hypothesis-driven reasoning. Submitted to the Meta + PyTorch + HuggingFace OpenEnv Hackathon.

---

## Repo & Remotes

| Remote | URL |
|---|---|
| GitHub (source of truth) | https://github.com/shasshaank/meta_hackthon.git |
| HF Training Space | https://huggingface.co/spaces/shashaank0707/AgentDebugger-training-v2 |
| HF Trained Model | https://huggingface.co/shashaank0707/AgentDebugger-trained |

Push to GitHub first, then to HF Space if needed:
```bash
git push origin main
git push space main --force   # space remote = HF training space
```

The `space` remote URL includes your HF token:
```
https://shashaank0707:YOUR_HF_TOKEN@huggingface.co/spaces/shashaank0707/AgentDebugger-training-v2
```

---

## Project Structure

```
meta_hackathon/
├── app.py                      # Gradio training monitor — launched by HF Space SDK
├── training/
│   └── train_grpo.py           # Main training script (GRPO via TRL)
├── server/
│   ├── reward_calculator.py    # Multi-component reward (format, hypothesis, fix, semantic)
│   ├── models.py               # parse_agent_output() — parses structured LLM output
│   └── app.py                  # FastAPI server (for the inference/env Space, not training)
├── data/
│   ├── bugs_tier1.jsonl        # 9 easy bugs (used steps 0–150)
│   ├── bugs_tier2.jsonl        # 31 medium bugs (added at step 150)
│   ├── bugs_tier3.jsonl        # 21 hard bugs (added at step 350 → was 600)
│   └── generate_bugs.py        # Script that generated the bug datasets
├── requirements.txt            # HF Space deps (gradio[oauth,mcp]==6.13.0, cu121 torch)
├── requirements_kaggle.txt     # Kaggle/RunPod deps (no torch pin, bitsandbytes==0.45.3)
├── inference.py                # Inference wrapper for evaluation
├── Dockerfile                  # For the inference/env Space (not the training space)
└── README.md                   # HF Space config header (sdk: gradio, app_file: app.py)
```

---

## Dependency Versions (locked — do not change without testing)

| Package | Version | Why pinned |
|---|---|---|
| `trl` | `0.14.0` | First version with `GRPOTrainer` + `GRPOConfig` |
| `pydantic` | `2.12.5` | Only version satisfying both gradio base AND gradio[mcp] constraints |
| `gradio` | `6.13.0[oauth,mcp]` | HF Space builder requires extras in one install pass |
| `bitsandbytes` | `0.45.3` (Kaggle) / `0.43.3` (HF Space cu121) | 0.45.3 has CUDA 12.x binaries; 0.43.3 works with cu121 |
| `transformers` | `4.46.3` | Tested with TRL 0.14.0 |
| `torch` | `2.5.1+cu121` (HF Space) / pre-installed (Kaggle) | |

**GRPOConfig param name:** `max_completion_length` (NOT `max_new_tokens` — that's the old name, breaks on 0.14.0)

---

## Training Script — Key Design Decisions

### GPU Auto-Detection (train_grpo.py ~line 260)
The script detects GPU at runtime and sets all hyperparams automatically:

| GPU | dtype | batch | grad_accum | num_gen | max_comp | lora_r |
|---|---|---|---|---|---|---|
| A100 40GB+ | bfloat16 | 2 | 4 | 8 | 256 | 16 |
| V100 32GB | float16 | 1 | 8 | 6 | 220 | 12 |
| T4 / ≤16GB | float16 | 1 | 8 | 4 | 160 | 8 |

**Critical:** P100 is NOT supported — PyTorch 2.x dropped sm_60 support. Use T4 instead.

### Curriculum
- Steps 0–150: Tier 1 bugs only (9 bugs)
- Steps 150–350: Tier 1 + Tier 2 (40 bugs)
- Steps 350+: All tiers (61 bugs)

### Reward Components (server/reward_calculator.py)
| Component | Weight | What it measures |
|---|---|---|
| format_compliance | 0.10 | All 5 fields present (OBSERVATION/HYPOTHESIS/CONFIDENCE/ACTION/DETAIL) |
| hypothesis_quality | 0.20 | Length + references specific variable names |
| localization | 0.15 | Correct function/line identified |
| fix_quality | 0.35 | Tests pass on proposed fix |
| semantic_similarity | 0.10 | Similarity to canonical fix |
| efficiency_potential | 0.10 | Potential-based shaping (Ibrahim et al. 2024) |

### Required Output Format
```
OBSERVATION: [specific observations with line numbers]
HYPOTHESIS: [2+ sentences explaining root cause with variable names]
CONFIDENCE: [low | medium | high]
ACTION: [inspect_lines | run_tests | propose_fix | request_context | give_up]
DETAIL: [complete fixed function code if propose_fix, else details]
```

---

## Running Training

### On Kaggle (T4 — free):
```python
# Cell 1 — install
!pip install -q wandb==0.18.7 datasets==3.0.2 transformers==4.46.3 \
    accelerate==1.0.1 trl==0.14.0 bitsandbytes==0.45.3 peft==0.13.2

# Cell 2 — clone + secrets
from kaggle_secrets import UserSecretsClient
import os
secrets = UserSecretsClient()
os.environ["WANDB_API_KEY"] = secrets.get_secret("WANDB_API_KEY")
os.environ["HF_TOKEN"]      = secrets.get_secret("HF_TOKEN")
!git clone https://github.com/shasshaank/meta_hackthon.git /kaggle/working/repo
%cd /kaggle/working/repo

# Cell 3 — train (streams output live)
import subprocess, sys
proc = subprocess.Popen(
    [sys.executable, "training/train_grpo.py"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, bufsize=1, cwd="/kaggle/working/repo"
)
for line in proc.stdout:
    print(line, end="", flush=True)
proc.wait()

# Cell 4 — save outputs after training
import shutil
shutil.copytree("/kaggle/working/repo/checkpoints", "/kaggle/working/checkpoints", dirs_exist_ok=True)
```

**Kaggle secrets needed:** `WANDB_API_KEY`, `HF_TOKEN`
**Kaggle GPU:** T4 x1 (NOT P100 — incompatible with modern PyTorch)
**Expected time:** ~8–10 hours for 500 steps (default max_steps=500)

### On RunPod (A100 — ~$1.09/hr):
```bash
git clone https://github.com/shasshaank/meta_hackthon.git && cd meta_hackthon
pip install -q wandb==0.18.7 datasets==3.0.2 transformers==4.46.3 \
    accelerate==1.0.1 trl==0.14.0 bitsandbytes==0.45.3 peft==0.13.2
WANDB_API_KEY=xxx HF_TOKEN=xxx python training/train_grpo.py
```
**Expected time:** ~3–4 hours for 1000 steps on A100 40GB

### Resume from checkpoint:
```bash
python training/train_grpo.py --resume ./checkpoints/checkpoint-400
```

### Local sanity check (no GPU):
```bash
python training/train_grpo.py --test-local
```

---

## HF Space Setup (training monitor)

The training Space (`AgentDebugger-training-v2`) is a Gradio app that:
1. On startup, spawns `training/train_grpo.py` in a background thread
2. Shows a live training log in the UI, auto-refreshing every 30s

**Required Space secrets:**
- `WANDB_API_KEY`
- `HF_TOKEN`

**Push to Space:**
```bash
git remote set-url space https://shashaank0707:YOUR_HF_TOKEN@huggingface.co/spaces/shashaank0707/AgentDebugger-training-v2
git push space main --force
```

---

## Known Issues Fixed (do not revert)

| Issue | Fix |
|---|---|
| `ImportError: cannot import name 'GRPOTrainer'` | `trl==0.12.2` → `trl==0.14.0` |
| `TypeError: GRPOConfig got unexpected keyword 'max_new_tokens'` | renamed to `max_completion_length` |
| `pydantic` conflict with `gradio[mcp]` | `pydantic==2.10.6` → `2.12.5` |
| `P100 not supported by PyTorch 2.x` | Switch to T4 on Kaggle |
| `bitsandbytes CUDA binary not found` | `bitsandbytes==0.43.3` → `0.45.3` on Kaggle |
| `unsloth` CUDA driver crash on HF A100 | Replaced with `bitsandbytes + peft` |
| `gradio every=` deprecation | Replaced with `gr.Timer(value=30)` |

---

## W&B Dashboard
https://wandb.ai/shashaankjain07-keshav-memorial-college-of-law/AgentDebuggerEnv

Training runs appear here automatically when `WANDB_API_KEY` is set.

---

## What's Left To Do

- [ ] **Finish training** — 500–1000 steps, model pushes to HF Hub automatically on completion
- [ ] **Verify trained model** — run `inference.py` against the trained model checkpoint
- [ ] **Update HF Space README** — change curriculum description to match actual step boundaries (150/350)
- [ ] **Submission** — ensure the inference/env Space (`AgentDebugger-env`) is live and healthy for judging
