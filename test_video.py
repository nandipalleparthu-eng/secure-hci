"""
Test runner for the Secure HCI System using a video file instead of a webcam.

Usage:
    python test_video.py --video path/to/your/video.mp4

Controls:
    ESC  - Exit
    F    - Freeze / Unfreeze gesture control
    SPACE - Pause / Resume video playback
"""

from __future__ import annotations

import argparse
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
    "frame_width": 960,
    "frame_height": 540,
    "camera_queue_size": 2,
    "gesture_queue_size": 2,
    "gesture_process_every_n": 1,
    "gesture_detection_confidence": 0.70,
    "gesture_tracking_confidence": 0.60,
    "known_faces_dir": "data/known_faces",
    "face_recognition_interval": 6,
    "face_tolerance": 0.50,
    "face_resize_scale": 0.25,
    "face_state_timeout": 3.0,
    "cursor_alpha": 0.22,
    "click_cooldown": 0.80,
    "scroll_cooldown": 0.10,
    "scroll_strength": 120,
    "window_name": "Secure HCI System [VIDEO TEST]",
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


def push_latest(q: queue.Queue, frame: np.ndarray) -> None:
    if q.full():
        try:
            q.get_nowait()
        except queue.Empty:
            pass
    try:
        q.put_nowait(frame)
    except queue.Full:
        pass


def video_thread(video_path: str) -> None:
    global display_frame

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Cannot open video file: %s", video_path)
        stop_event.set()
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_delay = 1.0 / video_fps
    paused = False

    logger.info("Video loaded: %s  (%.1f fps)", video_path, video_fps)

    while not stop_event.is_set():
        if paused:
            time.sleep(0.05)
            continue

        ok, frame = cap.read()
        if not ok:
            logger.info("End of video — looping back to start.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (CONFIG["frame_width"], CONFIG["frame_height"]))

        with display_lock:
            display_frame = frame.copy()

        push_latest(cam_to_gesture, frame.copy())
        push_latest(cam_to_face, frame.copy())

        time.sleep(frame_delay)

    cap.release()
    logger.info("Video thread stopped.")


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

    cv2.putText(frame, f"FPS: {state.fps:4.1f}", (14, 28), cv2.FONT_HERSHEY_DUPLEX, 0.7, (225, 255, 225), 1, cv2.LINE_AA)
    cv2.putText(frame, f"User: {state.face.name}", (14, 58), cv2.FONT_HERSHEY_DUPLEX, 0.7, auth_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Authorized: {'YES' if state.face.authorized else 'NO'}", (14, 88), cv2.FONT_HERSHEY_DUPLEX, 0.7, auth_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Gesture: {state.gesture.state.name}", (330, 28), cv2.FONT_HERSHEY_DUPLEX, 0.7, gesture_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Faces: {state.face.face_count}", (330, 58), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 220, 140), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Control: {'ENABLED' if control_enabled else 'DISABLED'}", (330, 88), cv2.FONT_HERSHEY_DUPLEX, 0.7, control_color, 1, cv2.LINE_AA)
    cv2.putText(frame, "ESC Exit  |  F Freeze  |  SPACE Pause Video", (width - 370, height - 14), cv2.FONT_HERSHEY_DUPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
    return frame


def apply_controls(runtime, smoother, click_debouncer, scroll_debouncer) -> None:
    gesture = runtime.gesture
    if not runtime.face.authorized or runtime.manual_freeze or not gesture.hand_present or gesture.state == GestureState.PAUSE:
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
    elif gesture.state == GestureState.SCROLL and scroll_debouncer.is_ready():
        amt = int(np.clip(gesture.scroll_delta * CONFIG["scroll_strength"], -500, 500))
        if amt:
            pyautogui.scroll(amt)
            scroll_debouncer.trigger()


def main(video_path: str) -> None:
    runtime = RuntimeState()

    video_worker = threading.Thread(target=video_thread, args=(video_path,), daemon=True, name="VideoThread")

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

    video_worker.start()
    gesture_controller.start(cam_to_gesture)
    face_authenticator.start(cam_to_face)

    smoother = CursorSmoother(alpha=CONFIG["cursor_alpha"])
    click_debouncer = Debouncer(CONFIG["click_cooldown"])
    scroll_debouncer = Debouncer(CONFIG["scroll_cooldown"])

    cv2.namedWindow(CONFIG["window_name"], cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONFIG["window_name"], CONFIG["frame_width"], CONFIG["frame_height"])

    fps_frames = 0
    fps_started = time.time()
    paused = False

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
            elif key == ord(" "):
                paused = not paused
                logger.info("Video %s.", "paused" if paused else "resumed")
            elif key in (ord("f"), ord("F")):
                runtime.manual_freeze = not runtime.manual_freeze
                logger.info("Manual freeze: %s.", runtime.manual_freeze)

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        stop_event.set()
        gesture_controller.stop()
        face_authenticator.stop()
        cv2.destroyAllWindows()
        logger.info("System shut down.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HCI system with a video file.")
    parser.add_argument("--video", required=True, help="Path to the test video file (mp4, avi, etc.)")
    args = parser.parse_args()
    main(args.video)
