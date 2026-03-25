"""
MediaPipe-based gesture recognition worker.

Supported gestures:
- Index finger only: move cursor
- Thumb and index pinch: left click
- Open palm: scroll
- Fist: pause/disable control
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from enum import Enum, auto

import cv2
import mediapipe as mp
import numpy as np


logger = logging.getLogger(__name__)


class GestureState(Enum):
    IDLE = auto()
    MOVE = auto()
    CLICK = auto()
    SCROLL = auto()
    PAUSE = auto()


@dataclass(slots=True)
class GestureData:
    state: GestureState = GestureState.IDLE
    pointer_x: float = 0.5
    pointer_y: float = 0.5
    scroll_delta: float = 0.0
    hand_present: bool = False


WRIST = 0
THUMB_TIP = 4
THUMB_IP = 3
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_TIP = 20


def _distance(a, b) -> float:
    return float(np.hypot(a.x - b.x, a.y - b.y))


def _is_finger_up(tip, pip, mcp) -> bool:
    """Finger is extended if tip is above both pip and mcp joints."""
    return tip.y < pip.y < mcp.y


def _is_thumb_extended(thumb_tip, thumb_ip, index_mcp) -> bool:
    """Loose thumb extension check that works on mirrored webcam frames."""
    return abs(thumb_tip.x - index_mcp.x) > abs(thumb_ip.x - index_mcp.x)


def _classify(hand_landmarks) -> tuple[GestureState, float]:
    thumb_tip = hand_landmarks[THUMB_TIP]
    thumb_ip = hand_landmarks[THUMB_IP]
    index_tip = hand_landmarks[INDEX_TIP]
    index_pip = hand_landmarks[INDEX_PIP]
    index_mcp = hand_landmarks[INDEX_MCP]
    middle_tip = hand_landmarks[MIDDLE_TIP]
    middle_pip = hand_landmarks[MIDDLE_PIP]
    middle_mcp = hand_landmarks[MIDDLE_MCP]
    ring_tip = hand_landmarks[RING_TIP]
    ring_pip = hand_landmarks[RING_PIP]
    ring_mcp = hand_landmarks[RING_MCP]
    pinky_tip = hand_landmarks[PINKY_TIP]
    pinky_pip = hand_landmarks[PINKY_PIP]
    pinky_mcp = hand_landmarks[PINKY_MCP]
    wrist = hand_landmarks[WRIST]

    index_up = _is_finger_up(index_tip, index_pip, index_mcp)
    middle_up = _is_finger_up(middle_tip, middle_pip, middle_mcp)
    ring_up = _is_finger_up(ring_tip, ring_pip, ring_mcp)
    pinky_up = _is_finger_up(pinky_tip, pinky_pip, pinky_mcp)
    thumb_up = _is_thumb_extended(thumb_tip, thumb_ip, index_mcp)

    pinch_distance = _distance(thumb_tip, index_tip)
    palm_height = max(abs(wrist.y - middle_tip.y), 1e-3)
    pinch_ratio = pinch_distance / palm_height
    if pinch_ratio < 0.32 and index_up:
        return GestureState.CLICK, 0.0

    if not any((thumb_up, index_up, middle_up, ring_up, pinky_up)):
        return GestureState.PAUSE, 0.0

    if all((index_up, middle_up, ring_up, pinky_up)) and thumb_up:
        scroll_delta = float(np.clip((wrist.y - middle_tip.y) * 2.5, -1.0, 1.0))
        return GestureState.SCROLL, scroll_delta

    if index_up and not middle_up and not ring_up and not pinky_up:
        return GestureState.MOVE, 0.0

    return GestureState.IDLE, 0.0


class GestureController:
    def __init__(
        self,
        result_queue: queue.Queue[GestureData],
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.70,
        min_tracking_confidence: float = 0.60,
        process_every_n: int = 1,
    ) -> None:
        self._result_queue = result_queue
        self._max_num_hands = max_num_hands
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._process_every_n = max(1, process_every_n)
        self._frame_count = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._hands_module = mp.solutions.hands

    def start(self, frame_queue: queue.Queue[np.ndarray]) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(frame_queue,),
            daemon=True,
            name="GestureThread",
        )
        self._thread.start()
        logger.info("Gesture worker started.")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _run(self, frame_queue: queue.Queue[np.ndarray]) -> None:
        with self._hands_module.Hands(
            static_image_mode=False,
            max_num_hands=self._max_num_hands,
            min_detection_confidence=self._min_detection_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
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
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if not result.multi_hand_landmarks:
            return GestureData(state=GestureState.IDLE, hand_present=False)

        hand_landmarks = result.multi_hand_landmarks[0].landmark
        state, scroll_delta = _classify(hand_landmarks)
        pointer = hand_landmarks[INDEX_TIP]

        return GestureData(
            state=state,
            pointer_x=float(pointer.x),
            pointer_y=float(pointer.y),
            scroll_delta=scroll_delta,
            hand_present=True,
        )
