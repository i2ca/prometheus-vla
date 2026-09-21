"""HUD compartilhado: desenha o passo atual (lerobot-ext/step.json) no frame.

Usado pelo teleop VR (teleop/xr_g1_arm.py) e pelo espelho OBS (mirror_cam.py),
garantindo a mesma renderização UTF-8 nos dois lugares.
"""

import json
import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

STEP_FILE = Path(os.environ.get(
    "STEP_FILE",
    os.path.expanduser("~/I2CA/prometheus-vla/lerobot-ext/step.json"),
))

_BOLD_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_REG_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


@lru_cache(maxsize=64)
def _font(path, size):
    return ImageFont.truetype(path, size)


def _width(draw, text, font):
    left, _, right, _ = draw.textbbox((0, 0), text, font=font)
    return max(right - left, draw.textlength(text, font=font))


def _ellipsize(draw, text, font, width):
    if _width(draw, text, font) <= width:
        return text
    if _width(draw, "…", font) > width:
        return ""
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if _width(draw, text[:mid].rstrip() + "…", font) <= width:
            low = mid
        else:
            high = mid - 1
    return text[:low].rstrip() + "…"


def _wrap(draw, text, font, width):
    lines, current = [], ""
    words = text.split()
    for index, word in enumerate(words):
        candidate = f"{current} {word}".strip()
        if current and _width(draw, candidate, font) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
        # Split long tokens too, rather than allowing them to cross the margin.
        while _width(draw, current, font) > width and len(current) > 1:
            low, high = 1, len(current) - 1
            while low < high:
                mid = (low + high + 1) // 2
                if _width(draw, current[:mid], font) <= width:
                    low = mid
                else:
                    high = mid - 1
            end = low
            lines.append(current[:end])
            current = current[end:]
            if len(lines) >= 2:
                break
        if len(lines) >= 2:
            # The layout only displays two lines. Keep the remainder for ellipsis.
            return lines + [" ".join([current, *words[index + 1:]])]
    if current:
        lines.append(current)
    return lines


@lru_cache(maxsize=32)
def _layout(w, h, header, sub):
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    pad = max(4, round(min(w / 640, h / 480) * 12))
    width = w - 2 * pad
    max_height = int(h * 0.28)
    if width < 8 or max_height < 20:
        return None
    base = max(12, round(min(w / 640, h / 480) * 30))
    minimum = max(10, round(base * 0.7))
    for size in range(base, minimum - 1, -1):
        font = _font(_BOLD_FONT, size)
        font_sub = _font(_REG_FONT, max(9, round(size * 0.65)))
        lines = _wrap(draw, header, font, width)
        truncated = len(lines) > 2
        if truncated:
            lines = [lines[0], _ellipsize(draw, " ".join(lines[1:]), font, width)]
        subtitle = _ellipsize(draw, sub, font_sub, width) if sub else ""
        gap = max(2, round(size * 0.2))
        y, items = pad, []
        rows = [(line, font, (255, 255, 255)) for line in lines]
        if subtitle:
            rows.append((subtitle, font_sub, (140, 220, 255)))
        for text, face, color in rows:
            left, top, right, bottom = draw.textbbox((0, 0), text, font=face)
            items.append((text, face, color, (pad - left, y - top)))
            y += bottom - top + gap
        bar_h = y - gap + pad + 1
        if bar_h <= max_height and (not truncated or size == minimum):
            return bar_h, items
    return None


class StepHud:
    """Lê lerobot-ext/step.json (com cache por mtime) e desenha o passo no frame."""

    def __init__(self, path=STEP_FILE):
        self.path = Path(path)
        self._mtime: float | None = None
        self._step: dict | None = None

    def step(self) -> dict | None:
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            self._mtime = None
            self._step = None
            return None
        if mtime == self._mtime and self._step is not None:
            return self._step
        self._mtime = mtime
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self._step = json.load(f)
        except Exception:
            self._step = None
        return self._step

    def draw(self, frame: np.ndarray) -> np.ndarray:
        step = self.step()
        if step is None:
            return frame
        desc = str(step.get("description") or "—")
        idx = step.get("step_index", "?")
        agent = step.get("agent") or step.get("tool") or ""

        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return frame

        header = f"PASSO {idx} — {desc}"
        sub = f"({agent})" if agent else ""

        layout = _layout(w, h, header, sub)
        if layout is None:
            return frame
        bar_h, items = layout
        # Only convert/blend the banner, leaving the camera pixels below intact.
        banner = frame[:bar_h]
        overlay = np.full_like(banner, (0, 18, 34))
        cv2.addWeighted(overlay, 0.82, banner, 0.18, 0, banner)
        img = Image.fromarray(cv2.cvtColor(banner, cv2.COLOR_BGR2RGB))
        dr = ImageDraw.Draw(img)
        for text, font, color, position in items:
            dr.text(position, text, font=font, fill=color)
        banner[:] = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        cv2.line(frame, (0, bar_h - 1), (w - 1, bar_h - 1), (0, 200, 255), 1)
        return frame


_step_hud = StepHud()
