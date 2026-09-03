#!/usr/bin/env bash
# One-time setup for a fresh Linux GPU box (RunPod / Vast.ai / Lambda / etc)
# or a Kaggle GPU notebook.
#
#   git clone <your-repo-url> AgentDebuggerEnv && cd AgentDebuggerEnv
#   bash scripts/setup_gpu_box.sh
set -euo pipefail

echo "== GPU check =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "WARNING: nvidia-smi not found -- no GPU visible on this box." >&2
fi

PYTHON="${PYTHON:-python3}"
PIP=""
PY=""

# Kaggle's Python often cannot create a venv (ensurepip fails). Detect that
# environment and install into the notebook interpreter instead.
on_kaggle=0
if [[ -n "${KAGGLE_KERNEL_RUN_TYPE:-}" || -d /kaggle/working ]]; then
  on_kaggle=1
fi

if [[ "$on_kaggle" -eq 1 ]]; then
  echo "== Kaggle detected: skipping venv, using notebook Python =="
  PY="$PYTHON"
  PIP="$PYTHON -m pip"
  $PIP install --upgrade pip
else
  echo "== Python venv =="
  if ! "$PYTHON" -m venv .venv; then
    echo "venv creation failed; falling back to the current Python (like Kaggle)." >&2
    PY="$PYTHON"
    PIP="$PYTHON -m pip"
    $PIP install --upgrade pip
  else
    PY="./.venv/bin/python"
    PIP="./.venv/bin/pip"
    $PIP install --upgrade pip
  fi
fi

echo "== Installing agentdebugger[train] (torch, transformers, trl, peft, datasets, accelerate, wandb) =="
# On Kaggle, torch+CUDA is often already present; still safe to re-resolve.
$PIP install -e '.[train,dev]'

# Kaggle pre-installs torchao==0.10.0. Newer PEFT (>=0.13) contains a version
# check in peft/tuners/lora/torchao.py that raises ImportError (not False) when
# torchao is present but below 0.16.0, crashing LoRA construction even though
# this project does not use torchao at all. Uninstalling it lets PEFT fall back
# to its standard dispatcher with no other effect.
echo "== Removing pre-installed torchao (incompatible with PEFT >= 0.13) =="
$PIP uninstall torchao -y 2>/dev/null || true

echo "== Sanity check: CUDA visible to torch =="
$PY -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

echo "== Fast test suite (sandbox + reward + dataset checks; no GPU needed) =="
$PY -m pytest -q -m "not slow"

if [[ "$on_kaggle" -eq 1 ]]; then
  RUN_PY="$PYTHON -m agentdebugger.cli"
  MATRIX_NOTE="Use: python -m agentdebugger.cli ...  (no .venv on Kaggle)"
else
  RUN_PY="./.venv/bin/python -m agentdebugger.cli"
  MATRIX_NOTE="Use: ./.venv/bin/python -m agentdebugger.cli ...  or ./scripts/run_matrix.sh"
fi

cat <<EOF

Setup done. Next steps:

  1. Authenticate W&B so training curves get logged (optional but strongly
     recommended -- see REPRODUCIBILITY.md for details):
       export WANDB_API_KEY=...
       export WANDB_PROJECT=agentdebugger

  2. (Optional) HF token, only needed if push_to_hub or a gated model is used:
       export HF_TOKEN=...

  3. Run the calibration run FIRST:
       $RUN_PY train \\
           --reward-config R0 --split train --max-steps 20 --seed 42 \\
           --output-dir ./ckpt/calibration

  4. Then the real matrix (Vast/RunPod): ./scripts/run_matrix.sh
     On Kaggle: $MATRIX_NOTE
EOF
