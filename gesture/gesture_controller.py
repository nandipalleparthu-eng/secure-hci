"""
MediaPipe-based gesture recognition — robust & conflict-free.
Each gesture has a unique, non-overlapping finger combination.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)


class GestureState(Enum):
    IDLE            = auto()
    MOVE            = auto()
    CLICK           = auto()
    RIGHT_CLICK     = auto()
    DOUBLE_CLICK    = auto()
    SCROLL          = auto()
    PAUSE           = auto()
    SWITCH_TAB      = auto()
    SCREENSHOT      = auto()
    ZOOM_IN         = auto()
    WORKSPACE_LEFT  = auto()
    WORKSPACE_RIGHT = auto()
    DRAG            = auto()


@dataclass(slots=True)
class GestureData:
    state: GestureState  = GestureState.IDLE
    pointer_x: float     = 0.5
    pointer_y: float     = 0.5
    scroll_delta: float  = 0.0
    hand_present: bool   = False
    distance_cm: float   = 0.0   # estimated distance from camera


# ── Landmark indices ──────────────────────────────────────────
WRIST      = 0
THUMB_TIP  = 4
THUMB_IP   = 3
THUMB_MCP  = 2
INDEX_MCP  = 5
INDEX_PIP  = 6
INDEX_TIP  = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_TIP = 12
RING_MCP   = 13
RING_PIP   = 14
RING_TIP   = 16
PINKY_MCP  = 17
PINKY_PIP  = 18
PINKY_TIP  = 20

# Real-world average palm width ~8cm, used for distance estimation
PALM_WIDTH_CM = 8.0
# Focal length calibration constant (tuned for typical webcam)
FOCAL_LENGTH  = 600.0


def _dist(a, b) -> float:
    return float(np.hypot(a.x - b.x, a.y - b.y))


def _dist_px(a, b, w, h) -> float:
    return float(np.hypot((a.x - b.x) * w, (a.y - b.y) * h))


def _up(tip, pip, mcp) -> bool:
    """Finger is extended: tip above pip above mcp (y decreases upward)."""
    return tip.y < pip.y and pip.y < mcp.y


def _curled(tip, mcp) -> bool:
    """Finger is deeply curled: tip clearly below mcp."""
    return tip.y >= mcp.y + 0.02


def _thumb_out(thumb_tip, thumb_ip, index_mcp) -> bool:
    """Thumb extended sideways."""
    return abs(thumb_tip.x - index_mcp.x) > abs(thumb_ip.x - index_mcp.x) * 1.1


def _estimate_distance(lm, frame_w: int) -> float:
    """Estimate distance in cm using palm width in pixels."""
    index_mcp = lm[INDEX_MCP]
    pinky_mcp = lm[PINKY_MCP]
    palm_px = abs(index_mcp.x - pinky_mcp.x) * frame_w
    if palm_px < 1:
        return 0.0
    return round((PALM_WIDTH_CM * FOCAL_LENGTH) / palm_px, 1)


def _classify(lm, swipe_buf: deque, frame_w: int) -> tuple[GestureState, float, float]:
    """Returns (GestureState, scroll_delta, distance_cm)."""

    tt  = lm[THUMB_TIP];  ti  = lm[THUMB_IP];   tm  = lm[THUMB_MCP]
    it  = lm[INDEX_TIP];  ip  = lm[INDEX_PIP];   im  = lm[INDEX_MCP]
    mdt = lm[MIDDLE_TIP]; mdp = lm[MIDDLE_PIP];  mdm = lm[MIDDLE_MCP]
    rt  = lm[RING_TIP];   rp  = lm[RING_PIP];    rm  = lm[RING_MCP]
    pt  = lm[PINKY_TIP];  pp  = lm[PINKY_PIP];   pm  = lm[PINKY_MCP]
    wr  = lm[WRIST]

    i_up  = _up(it,  ip,  im)
    m_up  = _up(mdt, mdp, mdm)
    r_up  = _up(rt,  rp,  rm)
    p_up  = _up(pt,  pp,  pm)
    th_up = _thumb_out(tt, ti, im)

    i_curl  = _curled(it,  im)
    m_curl  = _curled(mdt, mdm)
    r_curl  = _curled(rt,  rm)
    p_curl  = _curled(pt,  pm)

    palm_h = max(abs(wr.y - mdt.y), 1e-3)

    # Pinch ratios
    pinch_i = _dist(tt, it)  / palm_h
    pinch_m = _dist(tt, mdt) / palm_h
    pinch_r = _dist(tt, rt)  / palm_h

    dist_cm = _estimate_distance(lm, frame_w)

    # ── 1. PAUSE — full fist ──────────────────────────────────
    if i_curl and m_curl and r_curl and p_curl:
        swipe_buf.clear()
        return GestureState.PAUSE, 0.0, dist_cm

    # ── 2. SCROLL — all 5 open (open palm) ───────────────────
    if i_up and m_up and r_up and p_up and th_up:
        delta = float(np.clip((wr.y - mdt.y) * 2.5, -1.0, 1.0))
        swipe_buf.clear()
        return GestureState.SCROLL, delta, dist_cm

    # ── 3. SCREENSHOT — 4 fingers up, thumb tucked ───────────
    if i_up and m_up and r_up and p_up and not th_up:
        if _dist(tt, im) / palm_h < 0.55:
            swipe_buf.clear()
            return GestureState.SCREENSHOT, 0.0, dist_cm

    # ── 4. DOUBLE CLICK — all 3 pinch fingers close ──────────
    # Must check BEFORE single CLICK to avoid being swallowed
    if pinch_i < 0.35 and pinch_m < 0.35 and pinch_r < 0.35:
        swipe_buf.clear()
        return GestureState.DOUBLE_CLICK, 0.0, dist_cm

    # ── 5. CLICK — thumb+index pinch only, middle+ring not up ──
    if pinch_i < 0.35 and not m_up and not r_up:
        swipe_buf.clear()
        return GestureState.CLICK, 0.0, dist_cm

    # ── 6. DRAG — index+middle up AND tips crossed ────────────
    if i_up and m_up and not r_up and not p_up:
        tip_d = _dist(it, mdt)
        pip_d = _dist(ip, mdp)
        if tip_d < pip_d * 0.55:
            swipe_buf.clear()
            return GestureState.DRAG, 0.0, dist_cm
        # ── 7. RIGHT CLICK — index+middle spread (not crossed) ─
        if not th_up:
            swipe_buf.clear()
            return GestureState.RIGHT_CLICK, 0.0, dist_cm

    # ── 8. SWITCH TAB — pinky+thumb, others not extended ──────
    if p_up and th_up and not i_up and not m_up and not r_up:
        swipe_buf.clear()
        return GestureState.SWITCH_TAB, 0.0, dist_cm

    # ── 9. ZOOM IN — index+pinky up, middle+ring not extended ──
    if i_up and p_up and not m_up and not r_up and not th_up:
        swipe_buf.clear()
        return GestureState.ZOOM_IN, 0.0, dist_cm

    # ── 10. MOVE — index only + swipe ────────────────────────
    if i_up and not m_up and not r_up and not p_up:
        swipe_buf.append(it.x)
        if len(swipe_buf) == swipe_buf.maxlen:
            dx = swipe_buf[-1] - swipe_buf[0]
            if dx > 0.20:
                swipe_buf.clear()
                return GestureState.WORKSPACE_RIGHT, 0.0, dist_cm
            if dx < -0.20:
                swipe_buf.clear()
                return GestureState.WORKSPACE_LEFT, 0.0, dist_cm
        return GestureState.MOVE, 0.0, dist_cm

    return GestureState.IDLE, 0.0, dist_cm


class GestureController:
    def __init__(
        self,
        result_queue: queue.Queue[GestureData],
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.75,
        min_tracking_confidence: float = 0.65,
        process_every_n: int = 1,
    ) -> None:
        self._result_queue = result_queue
        self._max_num_hands = max_num_hands
        self._min_det = min_detection_confidence
        self._min_trk = min_tracking_confidence
        self._process_every_n = max(1, process_every_n)
        self._frame_count = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._hands = mp.solutions.hands
        self._swipe_buf: deque = deque(maxlen=14)

    def start(self, frame_queue: queue.Queue[np.ndarray]) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, args=(frame_queue,),
            daemon=True, name="GestureThread",
        )
        self._thread.start()
        logger.info("Gesture worker started.")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _run(self, frame_queue: queue.Queue[np.ndarray]) -> None:
        with self._hands.Hands(
            static_image_mode=False,
            max_num_hands=self._max_num_hands,
            min_detection_confidence=self._min_det,
            min_tracking_confidence=self._min_trk,
        ) as hands:
            while not self._stop_event.is_set():
                try:
                    frame = frame_queue.get(timeout=0.05)
                except queue.Empty:
                    continue

                self._frame_count += 1
                if self._frame_count % self._process_every_n != 0:
                    continue

                gesture = self._process_frame(frame, hands)
                if self._result_queue.full():
                    try:
                        self._result_queue.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    self._result_queue.put_nowait(gesture)
                except queue.Full:
                    pass

    def _process_frame(self, frame: np.ndarray, hands) -> GestureData:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if not result.multi_hand_landmarks:
            self._swipe_buf.clear()
            return GestureData(state=GestureState.IDLE, hand_present=False)

        lm = result.multi_hand_landmarks[0].landmark
        state, scroll_delta, dist_cm = _classify(lm, self._swipe_buf, w)
        pointer = lm[INDEX_TIP]

        return GestureData(
            state=state,
            pointer_x=float(pointer.x),
            pointer_y=float(pointer.y),
            scroll_delta=scroll_delta,
            hand_present=True,
            distance_cm=dist_cm,
        )
