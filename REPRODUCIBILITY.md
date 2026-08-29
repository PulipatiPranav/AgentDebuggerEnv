# Reproducibility Guide

This document describes how to exactly reproduce the primary experiments in the paper. 

## Environment

To guarantee identical library behavior (especially around GRPO and model execution), we recommend pinning these dependencies on Linux (CUDA 12.1+):

```text
torch==2.4.0
transformers==4.46.3
trl==0.16.0
peft==0.13.2
datasets==3.0.1
accelerate==1.0.0
wandb==0.18.0
```

*See `requirements-lock.txt` for the exact versions.*

## Model and Dataset Revisions

For exact reproducibility, we used:
- **Base Model:** `Qwen/Qwen2.5-Coder-3B-Instruct`
- **Dataset:** The built-in 180-bug dataset at commit `ab17937` (split exactly into 90 training and 90 held-out bugs).

## 1. Evaluating the Baselines (Zero-Shot)

```bash
# B0: Zero-shot structured
agentdebugger evaluate-curriculum --base-model Qwen/Qwen2.5-Coder-3B-Instruct --split heldout --format structured --output results/primary/B0.json

# B1: Zero-shot free-form (700 tokens)
agentdebugger evaluate-curriculum --base-model Qwen/Qwen2.5-Coder-3B-Instruct --split heldout --format free_form --output results/primary/B1_700tok.json
```

## 2. Training the Models (E1, E3, E4)

We train 9 RL models (3 reward configs × 3 seeds). You can run the entire matrix with the included script:

```bash
# Hardware: Tested on 1x 24GB GPU (e.g. RTX 4090 or RTX 3090)
export WANDB_API_KEY="..."
./scripts/run_matrix.sh
```

To run a single arm manually (e.g., E1 with seed 42):

```bash
agentdebugger train --model Qwen/Qwen2.5-Coder-3B-Instruct \
    --max-steps 500 \
    --seed 42 \
    --reward-config R0 \
    --format structured \
    --output-dir ckpt/E1_s42
```
*Note: We train for 500 total optimizer steps, implemented as three trainer stages of 150/200/150 steps corresponding to the tiered curriculum; LoRA weights are carried across stages, while optimizer/scheduler state is reinitialized.*

## 3. Statistical Analysis

After training and generating the evaluation JSONs in `results/primary/`, you can regenerate all claims and confidence intervals in the paper:

```bash
python analysis/bootstrap.py
```
This script will output the paired differences and 95% confidence intervals, recreating the statistical bounds reported in the manuscript.
