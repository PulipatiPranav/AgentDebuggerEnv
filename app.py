"""
AgentDebuggerEnv — Training Monitor
Gradio UI that boots GRPO training in a background process and streams live status.
"""

import subprocess
import threading
import gradio as gr
import os
import json
import sys
import time

# ── Start training in background ───────────────────────────────────────────────
training_log: list[str] = []
training_proc: subprocess.Popen | None = None
training_started_at: float = time.time()


def _stream_training():
    global training_proc
    script = os.path.join(os.path.dirname(__file__), "training", "train_grpo.py")
    training_proc = subprocess.Popen(
        [sys.executable, script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in training_proc.stdout:
        line = line.rstrip()
        training_log.append(line)
        if len(training_log) > 300:
            training_log.pop(0)
    training_proc.wait()


training_thread = threading.Thread(target=_stream_training, daemon=True)
training_thread.start()


# ── Status checker ─────────────────────────────────────────────────────────────
def check_status() -> str:
    lines: list[str] = []
    elapsed = int(time.time() - training_started_at)
    lines.append(f"Elapsed: {elapsed // 60}m {elapsed % 60}s")

    if training_proc is None:
        lines.append("Status: starting up (give it ~2 minutes)...")
    elif training_proc.poll() is None:
        lines.append("Status: TRAINING RUNNING ✓")
    else:
        code = training_proc.poll()
        lines.append(f"Status: {'COMPLETED ✓' if code == 0 else f'EXITED (code {code})'}")

    if os.path.exists("baseline_results.json"):
        try:
            with open("baseline_results.json") as f:
                baseline = json.load(f)
            lines.append(f"\nBaseline solve rate : {baseline['solve_rate']:.1%}")
            lines.append(f"Baseline avg reward : {baseline['avg_reward']:.3f}")
        except Exception:
            pass

    if os.path.exists("checkpoints"):
        ckpts = sorted(
            [d for d in os.listdir("checkpoints") if os.path.isdir(f"checkpoints/{d}")]
        )
        if ckpts:
            lines.append(f"\nLatest checkpoint   : {ckpts[-1]}")
            lines.append(f"Total checkpoints   : {len(ckpts)}")

    if os.path.exists("final_model"):
        lines.append("\nFinal model saved ✓ — training complete!")

    lines.append("\n" + "─" * 50)
    lines.append("Recent log (last 40 lines):")
    lines.extend(training_log[-40:] if training_log else ["(no output yet)"])

    return "\n".join(lines)


# ── Gradio UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(title="AgentDebuggerEnv Training Monitor") as demo:
    gr.Markdown(
        """
# AgentDebuggerEnv — GRPO Training Monitor
Training **Qwen2.5-Coder-7B-Instruct** on structured hypothesis-driven debugging.
- Algorithm: GRPO (same as DeepSeek-R1)
- Dataset: 90 hand-validated bugs across 3 difficulty tiers
- Curriculum: Tier 1 (steps 0–300) → Tier 1+2 (300–600) → All tiers (600+)
        """
    )
    status_box = gr.Textbox(
        label="Training Status",
        lines=50,
        max_lines=50,
        interactive=False,
    )
    refresh_btn = gr.Button("Refresh Status")
    refresh_btn.click(fn=check_status, outputs=status_box)

    # Auto-refresh every 30s
    demo.load(fn=check_status, outputs=status_box, every=30)

demo.launch(server_name="0.0.0.0", server_port=7860)
