#!/usr/bin/env python3
"""Render the CLI demo GIF that heads the README.

The GIF is generated, never hand-drawn: this script captures the real output of
``agentdebugger episode --task hard``, replays it through a terminal emulator one
chunk at a time, rasterises each frame, and encodes the frames with ffmpeg. Re-run
it and the GIF regenerates from whatever the CLI currently prints, so the demo can
never drift away from the tool.

    python scripts/render_demo.py --out docs/images/demo.gif

Requirements (installed as part of the `dev` extra plus Pillow/pyte):
    pip install pyte pillow
    ffmpeg on PATH

The renderer needs no GPU, no API key and no network; the episode it records is
driven by the oracle agent.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pyte
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent

# A calm, dark palette close to a default terminal.
BACKGROUND = (13, 17, 23)
FOREGROUND = (201, 209, 217)
PALETTE = {
    "black": (48, 54, 61),
    "red": (255, 123, 114),
    "green": (63, 185, 80),
    "yellow": (210, 168, 255),
    "blue": (88, 166, 255),
    "magenta": (188, 140, 255),
    "cyan": (57, 197, 207),
    "white": (201, 209, 217),
    "brightblack": (110, 118, 129),
}

COLUMNS, ROWS = 86, 34
CELL_W, CELL_H = 9, 19
PAD = 18
PROMPT = "$ agentdebugger episode --task hard"


def _load_font() -> ImageFont.FreeTypeFont:
    for name in (
        "/usr/share/fonts/adwaita-mono-fonts/AdwaitaMono-Regular.ttf",
        "/usr/share/fonts/google-noto-vf/NotoSansMono[wght].ttf",
        "/usr/share/fonts/liberation-mono-fonts/LiberationMono-Regular.ttf",
    ):
        if os.path.exists(name):
            return ImageFont.truetype(name, 15)
    return ImageFont.load_default()


FONT = _load_font()


def capture_episode() -> str:
    """Run the demo command and return its output with ANSI colour preserved."""
    env = {**os.environ, "TERM": "xterm-256color", "PYTHONPATH": str(ROOT / "src")}
    script = (
        "from agentdebugger import render; render.colour_enabled = lambda: True; "
        "render._terminal_width = lambda: 82; "
        "from agentdebugger.cli import main; main(['episode', '--task', 'hard'])"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True
    )
    return result.stdout


def _colour(cell: pyte.screens.Char) -> tuple[int, int, int]:
    if cell.fg == "default":
        return FOREGROUND
    if isinstance(cell.fg, str) and cell.fg in PALETTE:
        return PALETTE[cell.fg]
    try:
        return tuple(int(cell.fg[i : i + 2], 16) for i in (0, 2, 4))
    except (ValueError, TypeError):
        return FOREGROUND


def _render_screen(screen: pyte.Screen) -> Image.Image:
    width = COLUMNS * CELL_W + 2 * PAD
    height = ROWS * CELL_H + 2 * PAD
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)

    for row in range(ROWS):
        line = screen.buffer[row]
        for col in range(COLUMNS):
            cell = line[col]
            if not cell.data or cell.data == " ":
                continue
            x, y = PAD + col * CELL_W, PAD + row * CELL_H
            colour = _colour(cell)
            if cell.bold and colour == FOREGROUND:
                colour = (255, 255, 255)
            draw.text((x, y), cell.data, font=FONT, fill=colour)
    return image


def build_frames(output: str) -> list[Image.Image]:
    """Replay the captured output through pyte, one frame per chunk of bytes."""
    screen = pyte.Screen(COLUMNS, ROWS)
    stream = pyte.Stream(screen)
    frames: list[Image.Image] = []

    def snapshot(repeat: int = 1) -> None:
        frame = _render_screen(screen)
        frames.extend(frame for _ in range(repeat))

    # Type the command a few characters at a time, then "press enter".
    stream.feed(f"\033[92m{PROMPT[:2]}\033[0m")
    for index in range(2, len(PROMPT) + 1):
        screen.reset()
        stream.feed(f"\033[92m$\033[0m{PROMPT[1:index]}")
        snapshot()
    snapshot(6)

    # Stream the episode output, breaking on blank lines so each step lands as a beat.
    screen.reset()
    pending = ""
    for line in output.splitlines(keepends=True):
        pending += line
        if line.strip() == "":
            stream.feed(pending.replace("\n", "\r\n"))
            pending = ""
            snapshot(4)
    if pending:
        stream.feed(pending.replace("\n", "\r\n"))
    snapshot(30)  # hold on the final summary
    return frames


def encode_gif(frames: list[Image.Image], out: Path, fps: int = 10) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for index, frame in enumerate(frames):
            frame.save(Path(tmp) / f"frame_{index:04d}.png")
        palette = Path(tmp) / "palette.png"
        filters = "scale=iw/2:-1:flags=lanczos"
        subprocess.run(
            ["ffmpeg", "-y", "-i", f"{tmp}/frame_%04d.png",
             "-vf", f"{filters},palettegen=stats_mode=diff", str(palette)],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(fps), "-i", f"{tmp}/frame_%04d.png",
             "-i", str(palette),
             "-lavfi", f"{filters}[x];[x][1:v]paletteuse=dither=bayer",
             "-loop", "0", str(out)],
            check=True, capture_output=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "images" / "demo.gif")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        print("ffmpeg is required to encode the GIF.", file=sys.stderr)
        return 1

    print("capturing episode...", flush=True)
    output = capture_episode()
    print(f"rendering {len(output.splitlines())} lines to frames...", flush=True)
    frames = build_frames(output)
    print(f"encoding {len(frames)} frames -> {args.out}", flush=True)
    encode_gif(frames, args.out, fps=args.fps)
    size_kb = args.out.stat().st_size // 1024
    print(f"done: {args.out} ({size_kb} KB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
