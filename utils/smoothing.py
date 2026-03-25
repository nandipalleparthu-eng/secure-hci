"""Utilities for smoothing cursor motion and debouncing discrete gestures."""

from __future__ import annotations

import time


class CursorSmoother:
    """Exponential moving average smoother for cursor coordinates."""

    def __init__(self, alpha: float = 0.22) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in the range (0, 1].")
        self.alpha = alpha
        self._prev_x: float | None = None
        self._prev_y: float | None = None

    def smooth(self, x: float, y: float) -> tuple[float, float]:
        if self._prev_x is None or self._prev_y is None:
            self._prev_x = x
            self._prev_y = y
            return x, y

        smooth_x = self._prev_x + (x - self._prev_x) * self.alpha
        smooth_y = self._prev_y + (y - self._prev_y) * self.alpha
        self._prev_x = smooth_x
        self._prev_y = smooth_y
        return smooth_x, smooth_y

    def reset(self) -> None:
        self._prev_x = None
        self._prev_y = None


class Debouncer:
    """Cooldown gate for click and scroll actions."""

    def __init__(self, cooldown_seconds: float) -> None:
        self.cooldown_seconds = max(0.0, cooldown_seconds)
        self._last_trigger = 0.0

    def is_ready(self) -> bool:
        return (time.time() - self._last_trigger) >= self.cooldown_seconds

    def trigger(self) -> None:
        self._last_trigger = time.time()
