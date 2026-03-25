"""
Real-Time Secure Human-Computer Interaction System
Using Hand Gesture Recognition and Face Authentication.

This module starts three concurrent workers:
1. Camera capture thread
2. Gesture processing thread
3. Face recognition thread

The main thread combines their outputs, renders the UI overlay, and executes
OS mouse actions only when the active face is recognized.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import pyautogui

from face.face_auth import FaceAuthenticator, FaceAuthState
from gesture.gesture_controller import GestureController, GestureData, GestureState
from utils.smoothing import CursorSmoother, Debouncer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


CONFIG = {
    "camera_index": 0,
    "frame_width": 960,
    "frame_height": 540,
    "camera_queue_size": 2,
    "gesture_queue_size": 2,
    "camera_backend": cv2.CAP_DSHOW,
    "gesture_process_every_n": 1,
    "gesture_detection_confidence": 0.70,
    "gesture_tracking_confidence": 0.60,
    "known_faces_dir": "data/known_faces",
    "face_recognition_interval": 6,
    "face_tolerance": 0.50,
    "face_resize_scale": 0.25,
    "face_state_timeout": 1.5,
    "cursor_alpha": 0.22,
    "click_cooldown": 0.80,
    "scroll_cooldown": 0.10,
    "scroll_strength": 120,
    "window_name": "Secure HCI System",
}


pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0
SCREEN_W, SCREEN_H = pyautogui.size()


cam_to_gesture: queue.Queue[np.ndarray] = queue.Queue(maxsize=CONFIG["camera_queue_size"])
cam_to_face: queue.Queue[np.ndarray] = queue.Queue(maxsize=CONFIG["camera_queue_size"])
gesture_results: queue.Queue[GestureData] = queue.Queue(maxsize=CONFIG["gesture_queue_size"])

display_lock = threading.Lock()
display_frame: np.ndarray | None = None
stop_event = threading.Event()


@dataclass(slots=True)
class RuntimeState:
    gesture: GestureData = field(default_factory=GestureData)
    face: FaceAuthState = field(default_factory=FaceAuthState)
    manual_freeze: bool = False
    fps: float = 0.0


def push_latest(frame_queue: queue.Queue[np.ndarray], frame: np.ndarray) -> None:
    """Keep only the freshest frame so workers do not accumulate latency."""
    if frame_queue.full():
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass
    try:
        frame_queue.put_nowait(frame)
    except queue.Full:
        pass


def camera_thread() -> None:
    """Continuously capture frames and broadcast them to the worker queues."""
    global display_frame

    cap = cv2.VideoCapture(CONFIG["camera_index"], CONFIG["camera_backend"])
    if not cap.isOpened():
        logger.warning(
            "Camera backend %s failed. Retrying with OpenCV default backend.",
            CONFIG["camera_backend"],
        )
        cap.release()
        cap = cv2.VideoCapture(CONFIG["camera_index"])

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["frame_height"])
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        logger.error("Unable to open webcam at index %s.", CONFIG["camera_index"])
        stop_event.set()
        return

    logger.info(
        "Camera started at %sx%s.",
        CONFIG["frame_width"],
        CONFIG["frame_height"],
    )

    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            logger.warning("Webcam frame grab failed.")
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (CONFIG["frame_width"], CONFIG["frame_height"]))

        with display_lock:
            display_frame = frame.copy()

        push_latest(cam_to_gesture, frame.copy())
        push_latest(cam_to_face, frame.copy())

    cap.release()
    logger.info("Camera thread stopped.")


def draw_status_panel(frame: np.ndarray, state: RuntimeState) -> np.ndarray:
    overlay = frame.copy()
    height, width = frame.shape[:2]
    cv2.rectangle(overlay, (0, 0), (width, 112), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)

    auth_color = (60, 200, 60) if state.face.authorized else (30, 80, 220)
    control_enabled = state.face.authorized and not state.manual_freeze
    control_color = (40, 220, 80) if control_enabled else (0, 0, 220)
    gesture_color = {
        GestureState.IDLE: (200, 200, 200),
        GestureState.MOVE: (100, 230, 100),
        GestureState.CLICK: (100, 160, 255),
        GestureState.SCROLL: (0, 210, 255),
        GestureState.PAUSE: (0, 90, 240),
    }[state.gesture.state]

    cv2.putText(
        frame,
        f"FPS: {state.fps:4.1f}",
        (14, 28),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7,
        (225, 255, 225),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"User: {state.face.name}",
        (14, 58),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7,
        auth_color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Authorized: {'YES' if state.face.authorized else 'NO'}",
        (14, 88),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7,
        auth_color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Gesture: {state.gesture.state.name}",
        (330, 28),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7,
        gesture_color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Faces: {state.face.face_count}",
        (330, 58),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7,
        (255, 220, 140),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Control: {'ENABLED' if control_enabled else 'DISABLED'}",
        (330, 88),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7,
        control_color,
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "ESC Exit  |  F Freeze/Unfreeze",
        (width - 285, height - 14),
        cv2.FONT_HERSHEY_DUPLEX,
        0.5,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    return frame


def apply_controls(
    runtime: RuntimeState,
    smoother: CursorSmoother,
    click_debouncer: Debouncer,
    scroll_debouncer: Debouncer,
) -> None:
    gesture = runtime.gesture

    if not runtime.face.authorized or runtime.manual_freeze:
        smoother.reset()
        return

    if not gesture.hand_present or gesture.state == GestureState.PAUSE:
        smoother.reset()
        return

    if gesture.state == GestureState.MOVE:
        raw_x = int(np.clip(gesture.pointer_x, 0.0, 1.0) * SCREEN_W)
        raw_y = int(np.clip(gesture.pointer_y, 0.0, 1.0) * SCREEN_H)
        sx, sy = smoother.smooth(raw_x, raw_y)
        pyautogui.moveTo(int(sx), int(sy))

    elif gesture.state == GestureState.CLICK and click_debouncer.is_ready():
        pyautogui.click()
        click_debouncer.trigger()

    elif gesture.state == GestureState.RIGHT_CLICK and click_debouncer.is_ready():
        pyautogui.rightClick()
        click_debouncer.trigger()

    elif gesture.state == GestureState.DOUBLE_CLICK and click_debouncer.is_ready():
        pyautogui.doubleClick()
        click_debouncer.trigger()

    elif gesture.state == GestureState.DRAG:
        raw_x = int(np.clip(gesture.pointer_x, 0.0, 1.0) * SCREEN_W)
        raw_y = int(np.clip(gesture.pointer_y, 0.0, 1.0) * SCREEN_H)
        sx, sy = smoother.smooth(raw_x, raw_y)
        pyautogui.dragTo(int(sx), int(sy), button='left')

    elif gesture.state == GestureState.SCROLL and scroll_debouncer.is_ready():
        amt = int(np.clip(gesture.scroll_delta * CONFIG["scroll_strength"], -500, 500))
        if amt != 0:
            pyautogui.scroll(amt)
            scroll_debouncer.trigger()

    elif gesture.state == GestureState.SWITCH_TAB and click_debouncer.is_ready():
        pyautogui.hotkey('ctrl', 'tab')
        click_debouncer.trigger()

    elif gesture.state == GestureState.SCREENSHOT and click_debouncer.is_ready():
        pyautogui.hotkey('win', 'shift', 's')
        click_debouncer.trigger()

    elif gesture.state == GestureState.ZOOM_IN and scroll_debouncer.is_ready():
        pyautogui.hotkey('ctrl', '+')
        scroll_debouncer.trigger()

    elif gesture.state == GestureState.WORKSPACE_LEFT and click_debouncer.is_ready():
        pyautogui.hotkey('ctrl', 'win', 'left')
        click_debouncer.trigger()

    elif gesture.state == GestureState.WORKSPACE_RIGHT and click_debouncer.is_ready():
        pyautogui.hotkey('ctrl', 'win', 'right')
        click_debouncer.trigger()


def main() -> None:
    runtime = RuntimeState()

    capture_worker = threading.Thread(target=camera_thread, daemon=True, name="CameraThread")
    gesture_controller = GestureController(
        result_queue=gesture_results,
        min_detection_confidence=CONFIG["gesture_detection_confidence"],
        min_tracking_confidence=CONFIG["gesture_tracking_confidence"],
        process_every_n=CONFIG["gesture_process_every_n"],
    )
    face_authenticator = FaceAuthenticator(
        known_faces_dir=CONFIG["known_faces_dir"],
        recognition_interval=CONFIG["face_recognition_interval"],
        tolerance=CONFIG["face_tolerance"],
        resize_scale=CONFIG["face_resize_scale"],
        state_timeout=CONFIG["face_state_timeout"],
    )

    capture_worker.start()
    gesture_controller.start(cam_to_gesture)
    face_authenticator.start(cam_to_face)

    smoother = CursorSmoother(alpha=CONFIG["cursor_alpha"])
    click_debouncer = Debouncer(CONFIG["click_cooldown"])
    scroll_debouncer = Debouncer(CONFIG["scroll_cooldown"])

    cv2.namedWindow(CONFIG["window_name"], cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONFIG["window_name"], CONFIG["frame_width"], CONFIG["frame_height"])

    fps_frames = 0
    fps_started = time.time()

    try:
        while not stop_event.is_set():
            try:
                runtime.gesture = gesture_results.get_nowait()
            except queue.Empty:
                pass

            runtime.face = face_authenticator.get_state()
            apply_controls(runtime, smoother, click_debouncer, scroll_debouncer)

            with display_lock:
                frame = None if display_frame is None else display_frame.copy()

            if frame is None:
                time.sleep(0.01)
                continue

            fps_frames += 1
            elapsed = time.time() - fps_started
            if elapsed >= 1.0:
                runtime.fps = fps_frames / elapsed
                fps_frames = 0
                fps_started = time.time()

            frame = draw_status_panel(frame, runtime)
            cv2.imshow(CONFIG["window_name"], frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in (ord("f"), ord("F")):
                runtime.manual_freeze = not runtime.manual_freeze
                logger.info("Manual freeze set to %s.", runtime.manual_freeze)
    except KeyboardInterrupt:
        logger.info("Interrupted by keyboard.")
    finally:
        stop_event.set()
        gesture_controller.stop()
        face_authenticator.stop()
        cv2.destroyAllWindows()
        logger.info("System shut down.")


if __name__ == "__main__":
    main()
